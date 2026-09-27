from typing import Final, TypedDict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .authenticator import Authenticator
from .common_utils import CHATGPT_DEVICE_VERIFY_URL, GetAccessTokenError, GetDeviceCodeError


class ConnectionStatus(TypedDict):
    signed_in: bool


class DeviceLogin(TypedDict):
    user_code: str
    verification_uri: str
    interval: int

def create_login_router(authenticator: Authenticator, origin: str) -> APIRouter:
    del origin
    router: Final = APIRouter(prefix="/chatgpt")

    def status(
        account: str = "default", _user: UserAPIKeyAuth = Depends(user_api_key_auth)
    ) -> JSONResponse:
        payload: Final[ConnectionStatus] = {"signed_in": authenticator.is_signed_in(account)}
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    def finish_login(device_code: dict[str, str], account: str) -> None:
        try:
            authenticator.finish_device_login(device_code, account)
        except GetAccessTokenError as error:
            from litellm._logging import verbose_logger

            verbose_logger.warning("ChatGPT device login failed for profile %s: %s", account, error)

    def login(
        background_tasks: BackgroundTasks,
        account: str = "default",
        _user: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> JSONResponse:
        try:
            device_code: Final = authenticator.start_device_login(account)
        except (GetAccessTokenError, GetDeviceCodeError) as error:
            raise HTTPException(status_code=error.status_code, detail=error.message) from None

        background_tasks.add_task(finish_login, device_code, account)
        payload: Final[DeviceLogin] = {
            "user_code": device_code["user_code"],
            "verification_uri": CHATGPT_DEVICE_VERIFY_URL,
            "interval": int(device_code["interval"]),
        }
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    router.get("/status")(status)
    router.post("/login")(login)
    return router
