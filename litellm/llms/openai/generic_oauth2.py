from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.caching.in_memory_cache import InMemoryCache

from .common_utils import OpenAIError

_OAUTH2_TOKEN_CACHE: Final = InMemoryCache()
_OAUTH2_REFRESH_LOCK: Final = threading.Lock()
_EXPIRY_SAFETY_BUFFER_SECONDS: Final = 10
_DEFAULT_EXPIRES_IN_SECONDS: Final = 3600


class GenericOAuth2Error(OpenAIError):
    pass


@dataclass(frozen=True, slots=True)
class GenericOAuth2Config:
    token_endpoint: str
    refresh_token: str
    client_id: str | None
    client_secret: str | None
    scope: str | None
    auth_style: Literal["basic", "body"] | None


class _RawOAuth2Params(BaseModel):
    oauth2_token_endpoint: str | None = None
    oauth2_refresh_token: str | None = None
    oauth2_client_id: str | None = None
    oauth2_client_secret: str | None = None
    oauth2_scope: str | None = None
    oauth2_auth_style: Literal["basic", "body"] | None = None

    model_config = ConfigDict(extra="ignore")


class _TokenResponse(BaseModel):
    access_token: str
    expires_in: int = _DEFAULT_EXPIRES_IN_SECONDS

    model_config = ConfigDict(extra="ignore")


def resolve_generic_oauth2_config(litellm_params: Mapping[str, object]) -> GenericOAuth2Config | None:
    raw: Final = _RawOAuth2Params.model_validate(litellm_params)
    if raw.oauth2_token_endpoint is None or raw.oauth2_refresh_token is None:
        return None
    return GenericOAuth2Config(
        token_endpoint=raw.oauth2_token_endpoint,
        refresh_token=raw.oauth2_refresh_token,
        client_id=raw.oauth2_client_id,
        client_secret=raw.oauth2_client_secret,
        scope=raw.oauth2_scope,
        auth_style=raw.oauth2_auth_style,
    )


def get_generic_oauth2_bearer_token(config: GenericOAuth2Config) -> str:
    cache_key: Final = _cache_key(config)
    cached: Final = _OAUTH2_TOKEN_CACHE.get_cache(cache_key)
    if cached is not None:
        return cached

    with _OAUTH2_REFRESH_LOCK:
        locked_cached: Final = _OAUTH2_TOKEN_CACHE.get_cache(cache_key)
        if locked_cached is not None:
            return locked_cached
        return _refresh_token(config, cache_key)


def _cache_key(config: GenericOAuth2Config) -> str:
    authorization_context: Final = json.dumps(
        (
            config.token_endpoint,
            config.refresh_token,
            config.client_id,
            config.client_secret,
            config.scope,
            config.auth_style,
        ),
        separators=(",", ":"),
    )
    return hashlib.sha256(authorization_context.encode()).hexdigest()


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    encoded: Final = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {encoded}"


def _build_request(config: GenericOAuth2Config) -> tuple[Mapping[str, str], Mapping[str, str]]:
    required_data: Final[Mapping[str, str]] = MappingProxyType(
        {"grant_type": "refresh_token", "refresh_token": config.refresh_token}
    )
    base_data: Final[Mapping[str, str]] = (
        MappingProxyType({**required_data, "scope": config.scope}) if config.scope else required_data
    )
    base_headers: Final[Mapping[str, str]] = MappingProxyType(
        {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
    )

    if config.client_id and config.client_secret and config.auth_style != "body":
        headers: Final = MappingProxyType(
            {**base_headers, "Authorization": _basic_auth_header(config.client_id, config.client_secret)}
        )
        return base_data, headers
    if config.client_id and config.client_secret:
        data: Final = MappingProxyType(
            {**base_data, "client_id": config.client_id, "client_secret": config.client_secret}
        )
        return data, base_headers
    if config.client_id:
        return MappingProxyType({**base_data, "client_id": config.client_id}), base_headers
    return base_data, base_headers


def _refresh_token(config: GenericOAuth2Config, cache_key: str) -> str:
    client: Final = litellm.module_level_client
    data, headers = _build_request(config)

    try:
        response: Final = client.post(
            config.token_endpoint,
            data=dict(data),  # mutable-ok: HTTPHandler.post() requires a concrete dict, not a Mapping
            headers=dict(headers),  # mutable-ok: HTTPHandler.post() requires a concrete dict, not a Mapping
        )
    except litellm.Timeout as exc:
        raise GenericOAuth2Error(
            status_code=504,
            message=f"OAuth2 token refresh against {config.token_endpoint} timed out",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise GenericOAuth2Error(
            status_code=exc.response.status_code,
            message=f"OAuth2 token refresh failed against {config.token_endpoint}: HTTP {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        raise GenericOAuth2Error(
            status_code=500,
            message=f"OAuth2 token refresh request to {config.token_endpoint} failed: network error",
        ) from exc

    try:
        token_response: Final = _TokenResponse.model_validate(response.json())
    except ValueError as exc:
        raise GenericOAuth2Error(
            status_code=502,
            message=f"OAuth2 token endpoint {config.token_endpoint} returned an invalid token response",
        ) from exc

    ttl: Final = max(token_response.expires_in - _EXPIRY_SAFETY_BUFFER_SECONDS, 1)
    _OAUTH2_TOKEN_CACHE.set_cache(key=cache_key, value=token_response.access_token, ttl=ttl)
    return token_response.access_token
