from __future__ import annotations

import os
import platform
import tempfile
import threading
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException

from .models import PUBLIC_MODELS, is_discoverable_model

AUTHORIZE_URL: Final = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL: Final = "https://oauth2.googleapis.com/token"
BOOTSTRAP_URL: Final = "https://cloudcode-pa.googleapis.com"
RUNTIME_URL: Final = "https://daily-cloudcode-pa.googleapis.com"
DISCOVERY_URLS: Final = tuple(
    f"{base}/v1internal:{method}"
    for method in ("fetchAvailableModels", "models")
    for base in (
        RUNTIME_URL,
        BOOTSTRAP_URL,
        "https://daily-cloudcode-pa.sandbox.googleapis.com",
    )
)
SCOPES: Final = tuple(
    f"https://www.googleapis.com/auth/{scope}"
    for scope in ("cloud-platform", "userinfo.email", "userinfo.profile", "cclog", "experimentsandconfigs")
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_AUTH_LOCK: Final = threading.RLock()


class AntigravityError(BaseLLMException):
    pass


class ReauthenticationRequired(AntigravityError):
    pass


class Credentials(BaseModel):
    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str = Field(min_length=1, repr=False)
    expires_at: float
    client_id: str
    client_secret: str | None = Field(default=None, repr=False)
    project_id: str = ""
    email: str = ""
    requires_login: bool = False

    model_config = ConfigDict(frozen=True)


class TokenResponse(BaseModel):
    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    expires_in: int = Field(default=3600, gt=0)


def content_headers(access_token: str) -> MappingProxyType[str, str]:
    version: Final = os.getenv("ANTIGRAVITY_CLIENT_VERSION", "2.1.1")
    return MappingProxyType(
        {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": f"antigravity/ide/{version} {platform.system().lower()}/{platform.machine().lower()}",
        }
    )


def bootstrap_metadata() -> MappingProxyType[str, int]:
    host: Final = platform.system()
    arm: Final = platform.machine().lower() in ("arm64", "aarch64")
    platform_id: Final = (2 if arm else 1) if host == "Darwin" else (4 if arm else 3) if host == "Linux" else 5
    return MappingProxyType({"ideType": 9, "platform": platform_id, "pluginType": 2})


def bootstrap_headers(access_token: str) -> MappingProxyType[str, str]:
    version: Final = os.getenv("ANTIGRAVITY_CLIENT_VERSION", "2.1.1")
    return MappingProxyType(
        {
            **content_headers(access_token),
            "User-Agent": f"antigravity/{version} {platform.system().lower()}/{platform.machine().lower()} google-api-nodejs-client/10.3.0",
            "X-Goog-Api-Client": "gl-node/22.21.1",
        }
    )


def _project(data: Mapping[str, JsonValue]) -> str:
    value: Final = data.get("cloudaicompanionProject")
    identifier: Final = value.get("id") if isinstance(value, dict) else value
    return identifier.strip() if isinstance(identifier, str) else ""


def _discovered_models(data: Mapping[str, JsonValue]) -> tuple[str, ...]:
    models: Final = data.get("models")
    if isinstance(models, dict):
        return tuple(
            sorted(model_id for model_id, information in models.items() if is_discoverable_model(model_id, information))
        )
    if not isinstance(models, list):
        return ()
    identifiers: Final = tuple(
        information.get("id", information.get("name", information.get("model")))
        for information in models
        if isinstance(information, dict)
    )
    return tuple(
        sorted(
            model_id
            for model_id, information in zip(identifiers, (item for item in models if isinstance(item, dict)))
            if isinstance(model_id, str) and is_discoverable_model(model_id, information)
        )
    )


class Authenticator:
    def __init__(self, directory: Path, client: httpx.Client) -> None:
        self.directory = directory
        self.client = client

    def _path(self, account: str) -> Path:
        if (
            not account
            or len(account) > 80
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in account)
        ):
            raise AntigravityError(status_code=400, message="Invalid Antigravity account name")
        return self.directory / f"{account}.json"

    def read(self, account: str = "default") -> Credentials | None:
        path: Final = self._path(account)
        try:
            return Credentials.model_validate_json(path.read_text())
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            raise AntigravityError(
                status_code=401, message="Cannot read Antigravity credentials. Sign in again"
            ) from None

    def save(self, credentials: Credentials, account: str = "default") -> None:
        path: Final = self._path(account)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".credentials-")
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(credentials.model_dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _json(self, response: httpx.Response, operation: str) -> Mapping[str, JsonValue]:
        if response.is_error:
            raise AntigravityError(
                status_code=response.status_code,
                message=f"Antigravity {operation} failed (HTTP {response.status_code})",
            )
        try:
            return _JSON_OBJECT.validate_python(response.json())
        except ValueError:
            raise AntigravityError(
                status_code=502, message=f"Antigravity {operation} returned an invalid response"
            ) from None

    def _tokens(self, data: Mapping[str, str]) -> TokenResponse:
        try:
            response: Final = self.client.post(TOKEN_URL, data=data)
            if (
                response.status_code in (400, 401)
                and _JSON_OBJECT.validate_json(response.content).get("error") == "invalid_grant"
            ):
                raise ReauthenticationRequired(
                    status_code=401, message="Google authorization expired or was revoked. Sign in again"
                )
            return TokenResponse.model_validate(self._json(response, "token exchange"))
        except httpx.HTTPError:
            raise AntigravityError(status_code=502, message="Could not reach the Google token endpoint") from None
        except ValueError:
            raise AntigravityError(status_code=502, message="Google returned an invalid token response") from None

    def authorize_url(self, redirect_uri: str, state: str, challenge: str) -> str:
        client_id: Final = os.getenv("ANTIGRAVITY_OAUTH_CLIENT_ID")
        if not client_id:
            raise AntigravityError(status_code=503, message="Set ANTIGRAVITY_OAUTH_CLIENT_ID before signing in")
        return (
            AUTHORIZE_URL
            + "?"
            + urlencode(
                MappingProxyType(
                    {
                        "client_id": client_id,
                        "redirect_uri": redirect_uri,
                        "response_type": "code",
                        "scope": " ".join(SCOPES),
                        "state": state,
                        "access_type": "offline",
                        "prompt": "consent",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    }
                )
            )
        )

    def exchange(self, code: str, redirect_uri: str, verifier: str, account: str = "default") -> Credentials:
        client_id: Final = os.environ["ANTIGRAVITY_OAUTH_CLIENT_ID"]
        client_secret: Final = os.getenv("ANTIGRAVITY_OAUTH_CLIENT_SECRET")
        tokens: Final = self._tokens(
            MappingProxyType(
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "code_verifier": verifier,
                    "client_id": client_id,
                    **(MappingProxyType({"client_secret": client_secret}) if client_secret else MappingProxyType({})),
                }
            )
        )
        if not tokens.refresh_token:
            raise AntigravityError(
                status_code=401, message="Google did not return a refresh token. Sign in again and grant offline access"
            )
        credentials: Final = Credentials(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            expires_at=time.time() + tokens.expires_in,
            client_id=client_id,
            client_secret=client_secret,
        )
        with _AUTH_LOCK:
            self.save(credentials, account)
        return self.discover_project(account)

    def credentials(self, account: str = "default", require_project: bool = True) -> Credentials:
        with _AUTH_LOCK:
            stored: Final = self.read(account)
            if stored is None or stored.requires_login:
                raise AntigravityError(status_code=401, message="Sign in to Antigravity on the local login page first")
            active: Final = stored if stored.expires_at > time.time() + 60 else self._refresh(stored, account)
            if require_project and not active.project_id:
                raise AntigravityError(
                    status_code=422,
                    message="Antigravity needs a Google Cloud project. Use the local login page to discover or set it",
                )
            return active

    def _refresh(self, stored: Credentials, account: str) -> Credentials:
        try:
            return self._refresh_and_save(stored, account)
        except ReauthenticationRequired:
            self.save(stored.model_copy(update=MappingProxyType({"requires_login": True})), account)
            raise

    def _refresh_and_save(self, stored: Credentials, account: str) -> Credentials:
        tokens: Final = self._tokens(
            MappingProxyType(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": stored.refresh_token,
                    "client_id": stored.client_id,
                    **(
                        MappingProxyType({"client_secret": stored.client_secret})
                        if stored.client_secret
                        else MappingProxyType({})
                    ),
                }
            )
        )
        active: Final = stored.model_copy(
            update=MappingProxyType(
                {
                    "access_token": tokens.access_token,
                    "refresh_token": tokens.refresh_token or stored.refresh_token,
                    "expires_at": time.time() + tokens.expires_in,
                }
            )
        )
        self.save(active, account)
        return active

    def discover_project(self, account: str = "default", project_id: str = "") -> Credentials:
        with _AUTH_LOCK:
            credentials: Final = self.credentials(account, require_project=False)
            metadata: Final = bootstrap_metadata().copy()
            headers: Final = bootstrap_headers(credentials.access_token)
            body: Final = MappingProxyType(
                {
                    "metadata": metadata,
                    **(
                        MappingProxyType({"cloudaicompanionProject": project_id})
                        if project_id
                        else MappingProxyType({})
                    ),
                }
            )
            try:
                loaded: Final = self._json(
                    self.client.post(f"{BOOTSTRAP_URL}/v1internal:loadCodeAssist", headers=headers, json=body.copy()),
                    "project discovery",
                )
                discovered: Final = _project(loaded) or self._onboard_project(loaded, headers, body)
            except httpx.HTTPError:
                raise AntigravityError(
                    status_code=502, message="Could not reach Antigravity project discovery. Retry from the login page"
                ) from None
            updated: Final = credentials.model_copy(update=MappingProxyType({"project_id": discovered or project_id}))
            self.save(updated, account)
            return updated

    def _onboard_project(
        self, loaded: Mapping[str, JsonValue], headers: Mapping[str, str], body: MappingProxyType[str, object]
    ) -> str:
        tiers: Final = loaded.get("allowedTiers")
        default_tier: Final = (
            next((t.get("id") for t in tiers if isinstance(t, dict) and t.get("isDefault")), "legacy-tier")
            if isinstance(tiers, list)
            else "legacy-tier"
        )
        onboarded: Final = self._json(
            self.client.post(
                f"{BOOTSTRAP_URL}/v1internal:onboardUser",
                headers=headers,
                json=MappingProxyType({"tier_id": default_tier, **body}).copy(),
            ),
            "project onboarding",
        )
        nested: Final = onboarded.get("response")
        retry: Final = self._json(
            self.client.post(f"{BOOTSTRAP_URL}/v1internal:loadCodeAssist", headers=headers, json=body.copy()),
            "project discovery",
        )
        return _project(retry) or _project(nested if isinstance(nested, dict) else onboarded)

    def models(self, account: str = "default") -> tuple[str, ...]:
        credentials: Final = self.credentials(account)
        headers: Final = content_headers(credentials.access_token)
        for url in DISCOVERY_URLS:
            try:
                response = self.client.post(  # rebind-ok: each endpoint produces a new response
                    url,
                    headers=headers,
                    json=_JSON_OBJECT.validate_python(MappingProxyType({})),
                )
                if response.is_error:
                    continue
                data = _JSON_OBJECT.validate_python(  # rebind-ok: each endpoint has an independent catalog
                    response.json()
                )
                discovered = _discovered_models(data)  # rebind-ok: each endpoint has an independent catalog
                if discovered:
                    return discovered
            except (httpx.HTTPError, ValueError):
                continue
        return PUBLIC_MODELS


@lru_cache(maxsize=1)
def get_authenticator() -> Authenticator:
    directory: Final = Path(os.getenv("ANTIGRAVITY_AUTH_DIR", "~/.config/litellm/antigravity")).expanduser()
    return Authenticator(directory, httpx.Client(timeout=30))
