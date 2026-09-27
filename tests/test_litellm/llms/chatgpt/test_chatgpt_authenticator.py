import base64
import json
import os
import time
from unittest.mock import mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"

    def test_named_account_isolated_and_private(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        authenticator = Authenticator()
        authenticator._write_auth_file({"access_token": "work-token"}, "work")

        assert authenticator._read_auth_file("work") == {"access_token": "work-token"}
        assert authenticator._read_auth_file() is None
        assert os.stat(authenticator.get_auth_file("work")).st_mode & 0o777 == 0o600

    @pytest.mark.parametrize("account", ["../escape", "", "a" * 81, "name/child"])
    def test_invalid_account_name_rejected(self, authenticator, account):
        with pytest.raises(GetAccessTokenError, match="Invalid"):
            authenticator.get_auth_file(account)

    def test_device_login_records_named_account(self, authenticator):
        device_code = {"device_auth_id": "device", "user_code": "CODE", "interval": "5"}
        tokens = {"access_token": "access", "refresh_token": "refresh", "id_token": "id"}
        with (
            patch.object(authenticator, "_read_auth_file", return_value=None),
            patch.object(authenticator, "_request_device_code", return_value=device_code),
            patch.object(authenticator, "_record_device_code_request") as record,
            patch.object(authenticator, "_poll_for_authorization_code", return_value={}),
            patch.object(authenticator, "_exchange_code_for_tokens", return_value=tokens),
            patch.object(authenticator, "_write_auth_file"),
        ):
            authenticator._login_device_code("work")

        record.assert_called_once_with("work")
