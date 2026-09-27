from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .authenticator import AntigravityError, Authenticator
from .usage import get_usage


@dataclass(frozen=True, slots=True)
class LoginAttempt:
    state: str
    verifier: str
    created_at: float
    return_to: str
    account: str


class LoginSessions:
    def __init__(self) -> None:
        self.attempts: MappingProxyType[str, LoginAttempt] = MappingProxyType({})
        self.lock = threading.Lock()

    def create(self, account: str = "default", return_to: str = "") -> LoginAttempt:
        with self.lock:
            safe_return: Final = return_to if return_to.startswith("/ui") and not return_to.startswith("//") else ""
            attempt: Final = LoginAttempt(
                secrets.token_urlsafe(32), secrets.token_urlsafe(64), time.monotonic(), safe_return, account
            )
            recent: Final = tuple(a for a in self.attempts.values() if time.monotonic() - a.created_at < 600)[-31:]
            self.attempts = MappingProxyType({a.state: a for a in (*recent, attempt)})
            return attempt

    def consume(self, state: str, cookie: str) -> LoginAttempt:
        with self.lock:
            attempt: Final = self.attempts.get(state)
            if (
                not state
                or not secrets.compare_digest(state, cookie)
                or attempt is None
                or time.monotonic() - attempt.created_at >= 600
            ):
                raise HTTPException(status_code=400, detail="Login expired or state did not match. Start sign-in again")
            self.attempts = MappingProxyType({key: value for key, value in self.attempts.items() if key != state})
            return attempt


class ProjectRequest(BaseModel):
    project_id: str = Field(default="", max_length=128, pattern=r"^[a-zA-Z0-9:._-]*$")


class ConnectionStatus(TypedDict):
    signed_in: ReadOnly[bool]
    project_id: ReadOnly[str]


class ProjectStatus(TypedDict):
    project_id: ReadOnly[str]


class ModelCatalog(TypedDict):
    models: ReadOnly[tuple[str, ...]]


def create_login_router(authenticator: Authenticator, origin: str) -> APIRouter:
    sessions: Final = LoginSessions()
    redirect_uri: Final = origin + "/antigravity/callback"

    router: Final = APIRouter(prefix="/antigravity")

    def page() -> HTMLResponse:
        return HTMLResponse(
            Path(__file__).with_name("login.html").read_text(),
            # Preserve Origin on the local form POST while omitting cross-origin referrers.
            headers=MappingProxyType({"Cache-Control": "no-store", "Referrer-Policy": "same-origin"}),
        )

    def status(
        account: str = "default", _user: UserAPIKeyAuth = Depends(user_api_key_auth)
    ) -> JSONResponse:
        credentials: Final = authenticator.read(account)
        payload: Final[ConnectionStatus] = {
            "signed_in": credentials is not None and not credentials.requires_login,
            "project_id": credentials.project_id if credentials else "",
        }
        return JSONResponse(
            payload,
            headers=MappingProxyType({"Cache-Control": "no-store"}),
        )

    def login(request: Request, account: str = "default", return_to: str = "") -> RedirectResponse:
        if request.headers.get("origin") != origin:
            raise HTTPException(status_code=403, detail="Cross-origin login prohibited")
        attempt: Final = sessions.create(account, return_to)
        challenge: Final = (
            base64.urlsafe_b64encode(hashlib.sha256(attempt.verifier.encode()).digest()).rstrip(b"=").decode()
        )
        try:
            url: Final = authenticator.authorize_url(redirect_uri, attempt.state, challenge)
        except AntigravityError as error:
            raise HTTPException(status_code=error.status_code, detail=error.message) from None
        response: Final = RedirectResponse(url, status_code=303)
        response.set_cookie(
            "antigravity_login_state", attempt.state, httponly=True, samesite="lax", max_age=600, path="/antigravity"
        )
        return response

    def finish_login(code: str, verifier: str, account: str) -> str:
        try:
            authenticator.exchange(code, redirect_uri, verifier, account)
            return "connected"
        except AntigravityError:
            return "needs_attention"

    def callback(request: Request, state: str = "", code: str = "", error: str = "") -> RedirectResponse:
        attempt: Final = sessions.consume(state, request.cookies.get("antigravity_login_state", ""))
        if error or not code:
            return RedirectResponse(origin + "/antigravity?login=cancelled", status_code=303)
        outcome: Final = finish_login(code, attempt.verifier, attempt.account)
        destination: Final = attempt.return_to or "/antigravity?login=" + outcome
        response: Final = RedirectResponse(origin + destination, status_code=303)
        response.delete_cookie("antigravity_login_state", path="/antigravity")
        return response

    def project(
        body: ProjectRequest,
        account: str = "default",
        _user: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> JSONResponse:
        try:
            credentials: Final = authenticator.discover_project(account=account, project_id=body.project_id)
            payload: Final[ProjectStatus] = {"project_id": credentials.project_id}
            return JSONResponse(payload)
        except AntigravityError as error:
            raise HTTPException(status_code=error.status_code, detail=error.message) from None

    def models(
        account: str = "default", _user: UserAPIKeyAuth = Depends(user_api_key_auth)
    ) -> JSONResponse:
        try:
            payload: Final[ModelCatalog] = {"models": authenticator.models(account)}
            return JSONResponse(payload)
        except AntigravityError as error:
            raise HTTPException(status_code=error.status_code, detail=error.message) from None

    def usage(
        account: str = "default",
        refresh: bool = False,
        _user: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> JSONResponse:
        try:
            result: Final = get_usage(authenticator, account=account, force_refresh=refresh)
            return JSONResponse(
                result.model_dump(mode="json"),
                headers=MappingProxyType({"Cache-Control": "no-store"}),
            )
        except AntigravityError as error:
            raise HTTPException(status_code=error.status_code, detail=error.message) from None

    router.get("")(page)
    router.get("/status")(status)
    router.post("/login")(login)
    router.get("/callback")(callback)
    router.post("/project")(project)
    router.get("/models")(models)
    router.get("/usage")(usage)
    return router
