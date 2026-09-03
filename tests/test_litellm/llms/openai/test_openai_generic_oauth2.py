"""
Tests for generic OAuth2 refresh_token support (litellm/llms/openai/generic_oauth2.py)
"""

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import respx

from litellm.llms.openai.generic_oauth2 import (
    _OAUTH2_TOKEN_CACHE,
    GenericOAuth2Config,
    GenericOAuth2Error,
    get_generic_oauth2_bearer_token,
    resolve_generic_oauth2_config,
)

TOKEN_URL = "https://idp.example.com/oauth/token"


@pytest.fixture(autouse=True)
def _clean_oauth2_cache():
    _OAUTH2_TOKEN_CACHE.flush_cache()
    yield
    _OAUTH2_TOKEN_CACHE.flush_cache()


def _config(**overrides) -> GenericOAuth2Config:
    defaults = {
        "token_endpoint": TOKEN_URL,
        "refresh_token": "r1",
        "client_id": None,
        "client_secret": None,
        "scope": None,
        "auth_style": None,
    }
    defaults.update(overrides)
    return GenericOAuth2Config(**defaults)


def _form_body(request: httpx.Request) -> dict[str, str]:
    return dict(item.split("=", 1) for item in request.content.decode().split("&"))


class TestResolveGenericOAuth2Config:
    def test_returns_none_when_no_oauth2_fields(self):
        assert resolve_generic_oauth2_config({}) is None
        assert resolve_generic_oauth2_config({"api_key": "sk-x"}) is None

    def test_returns_none_when_only_token_endpoint_set(self):
        assert resolve_generic_oauth2_config({"oauth2_token_endpoint": TOKEN_URL}) is None

    def test_returns_none_when_only_refresh_token_set(self):
        assert resolve_generic_oauth2_config({"oauth2_refresh_token": "r1"}) is None

    def test_returns_config_when_both_present(self):
        config = resolve_generic_oauth2_config(
            {
                "oauth2_token_endpoint": TOKEN_URL,
                "oauth2_refresh_token": "r1",
                "oauth2_client_id": "client-1",
                "oauth2_client_secret": "secret-1",
                "oauth2_scope": "llm.read",
                "oauth2_auth_style": "body",
                "api_key": "sk-should-be-ignored",
                "unrelated_field": "ignored",
            }
        )
        assert config == _config(client_id="client-1", client_secret="secret-1", scope="llm.read", auth_style="body")


class TestGetGenericOAuth2BearerToken:
    @respx.mock
    def test_sends_refresh_token_grant(self):
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        )
        assert get_generic_oauth2_bearer_token(_config()) == "at-1"
        body = _form_body(route.calls.last.request)
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "r1"
        assert "Authorization" not in route.calls.last.request.headers

    @respx.mock
    def test_includes_scope_when_set(self):
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        )
        get_generic_oauth2_bearer_token(_config(scope="llm.read"))
        assert _form_body(route.calls.last.request)["scope"] == "llm.read"

    @respx.mock
    def test_uses_basic_auth_when_client_id_and_secret_present(self):
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        )
        get_generic_oauth2_bearer_token(_config(client_id="client-1", client_secret="secret-1"))
        request = route.calls.last.request
        assert request.headers["Authorization"].startswith("Basic ")
        body = _form_body(request)
        assert "client_id" not in body
        assert "client_secret" not in body

    @respx.mock
    def test_uses_body_client_id_when_only_client_id_set(self):
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        )
        get_generic_oauth2_bearer_token(_config(client_id="public-client"))
        request = route.calls.last.request
        assert "Authorization" not in request.headers
        assert _form_body(request)["client_id"] == "public-client"

    @respx.mock
    def test_uses_body_creds_when_auth_style_body_explicit(self):
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        )
        get_generic_oauth2_bearer_token(
            _config(client_id="client-1", client_secret="secret-1", auth_style="body")
        )
        request = route.calls.last.request
        assert "Authorization" not in request.headers
        body = _form_body(request)
        assert body["client_id"] == "client-1"
        assert body["client_secret"] == "secret-1"

    @respx.mock
    def test_cache_hit_avoids_second_http_call(self):
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        )
        config = _config()
        assert get_generic_oauth2_bearer_token(config) == "at-1"
        assert get_generic_oauth2_bearer_token(config) == "at-1"
        assert route.call_count == 1

    @respx.mock
    def test_refetches_after_ttl_expiry(self):
        route = respx.post(TOKEN_URL).mock(
            side_effect=[
                httpx.Response(200, json={"access_token": "at-1", "expires_in": 12}),
                httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600}),
            ]
        )
        config = _config()
        assert get_generic_oauth2_bearer_token(config) == "at-1"
        time.sleep(2.5)
        assert get_generic_oauth2_bearer_token(config) == "at-2"
        assert route.call_count == 2

    @respx.mock
    def test_raises_generic_oauth2_error_on_401_and_redacts_secrets(self):
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(401, text="invalid_grant: refresh-token-secret-xyz")
        )
        config = _config(
            refresh_token="refresh-token-secret-xyz",
            client_id="client-1",
            client_secret="super-secret-value",
        )
        with pytest.raises(GenericOAuth2Error) as excinfo:
            get_generic_oauth2_bearer_token(config)
        assert "refresh-token-secret-xyz" not in str(excinfo.value)
        assert "super-secret-value" not in str(excinfo.value)
        assert excinfo.value.status_code == 401

    @respx.mock
    def test_raises_generic_oauth2_error_on_missing_access_token(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"token_type": "Bearer"}))
        with pytest.raises(GenericOAuth2Error):
            get_generic_oauth2_bearer_token(_config())

    @respx.mock
    def test_raises_generic_oauth2_error_on_non_json_response(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, text="not json"))
        with pytest.raises(GenericOAuth2Error):
            get_generic_oauth2_bearer_token(_config())

    @respx.mock
    def test_concurrent_calls_hit_token_endpoint_once(self):
        def _slow_response(request: httpx.Request) -> httpx.Response:
            time.sleep(0.2)
            return httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})

        route = respx.post(TOKEN_URL).mock(side_effect=_slow_response)
        config = _config()

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: get_generic_oauth2_bearer_token(config), range(8)))

        assert all(r == "at-1" for r in results)
        assert route.call_count == 1
