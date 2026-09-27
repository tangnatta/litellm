import json
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.llms.antigravity.authenticator import (
    AUTHORIZE_URL,
    BOOTSTRAP_URL,
    DISCOVERY_URLS,
    RUNTIME_URL,
    TOKEN_URL,
    AntigravityError,
    Authenticator,
    Credentials,
    ReauthenticationRequired,
    bootstrap_metadata,
    content_headers,
    get_authenticator,
)
from litellm.llms.antigravity.chat.transformation import unwrap_lines
from litellm.llms.antigravity.login import LoginSessions, create_login_router
from litellm.llms.antigravity.models import (
    PUBLIC_MODELS,
    is_discoverable_model,
    output_token_cap,
    resolve_model_id,
)
from litellm.llms.antigravity.usage import (
    _number,
    _plan,
    _quota,
    _reset_time,
    get_usage,
)


@pytest.fixture
def auth(tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("ANTIGRAVITY_AUTH_DIR", str(tmp_path))
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_CLIENT_SECRET", "test-secret")
    get_authenticator.cache_clear()
    authenticator = get_authenticator()
    authenticator.save(
        Credentials(
            access_token="test-access",
            refresh_token="test-refresh",
            expires_at=time.time() + 3600,
            client_id="test-client",
            client_secret="test-secret",
            project_id="test-project",
        )
    )
    yield authenticator
    authenticator.client.close()
    get_authenticator.cache_clear()


def sse():
    frames = (
        {"response": {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hello"}]}}]}},
        {
            "response": {
                "candidates": [
                    {"index": 0, "content": {"role": "model", "parts": [{"text": " world"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
            }
        },
    )
    return "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)


@pytest.mark.parametrize("stream", [False, True])
@respx.mock
def test_completion_calls_code_assist_directly(auth, stream):
    route = respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(
        200, text=sse(), headers={"content-type": "text/event-stream"}
    )
    result = litellm.completion(
        model="antigravity/test-model", messages=[{"role": "user", "content": "Hi"}], stream=stream
    )
    response = litellm.stream_chunk_builder(list(result)) if stream else result
    assert response.choices[0].message.content == "Hello world"
    assert response.usage.total_tokens == 6
    body = json.loads(route.calls.last.request.content)
    assert body["project"] == "test-project"
    assert body["model"] == "test-model"
    assert body["request"]["contents"][0]["parts"][0]["text"] == "Hi"
    assert body["request"]["sessionId"].startswith("-")
    assert body["requestId"].startswith("agent/")
    assert "stream" not in body and "stream" not in body["request"]
    assert route.calls.last.request.headers["authorization"] == "Bearer test-access"
    assert "test-refresh" not in route.calls.last.request.content.decode()


@respx.mock
def test_model_discovery_filters_internal_and_non_chat_models(auth):
    route = respx.post(DISCOVERY_URLS[0]).respond(
        200,
        json={
            "models": {
                "chat_20706": {"isInternal": True},
                "gemini-3.1-flash-image": {},
                "tab_flash_lite_preview": {},
                "gemini-2.5-pro": {},
                "gemini-pro-agent": {"displayName": "Gemini Pro"},
            }
        },
    )
    assert auth.models() == ("gemini-pro-agent",)
    assert json.loads(route.calls.last.request.content) == {}


@respx.mock
def test_model_discovery_falls_back_across_endpoints_and_shapes(auth):
    respx.post(DISCOVERY_URLS[0]).respond(503)
    respx.post(DISCOVERY_URLS[1]).respond(
        200,
        json={"models": [{"id": "chat_20706", "isInternal": True}, {"name": "gemini-3.8-flash-tiered"}]},
    )
    assert auth.models() == ("gemini-3.8-flash-tiered",)


@respx.mock
def test_model_discovery_uses_curated_fallback(auth):
    for url in DISCOVERY_URLS:
        respx.post(url).respond(503)
    assert auth.models() == PUBLIC_MODELS


@respx.mock
def test_usage_combines_live_catalog_and_weekly_quotas(auth):
    respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").respond(
        200, json={"currentTier": {"id": "standard-tier", "name": "Antigravity"}}
    )
    respx.post(DISCOVERY_URLS[0]).respond(
        200,
        json={
            "models": {
                "gemini-pro-agent": {"quotaInfo": {"remainingFraction": 0.8, "resetTime": "2030-01-01T00:00:00Z"}},
                "claude-sonnet-4-6": {"quotaInfo": {"remainingFraction": 0.4}},
                "chat_20706": {"quotaInfo": {"remainingFraction": 1}},
            }
        },
    )
    respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuota").respond(
        200,
        json={
            "buckets": [{"modelId": "gemini-pro-agent", "remainingFraction": 0.25, "resetTime": "2030-01-02T00:00:00Z"}]
        },
    )
    respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuotaSummary").respond(
        200,
        json={
            "groups": [
                {
                    "displayName": "Gemini Models",
                    "buckets": [{"bucketId": "weekly", "remainingFraction": 0.1, "resetTime": 1893542400000}],
                }
            ]
        },
    )
    result = get_usage(auth, force_refresh=True)
    quotas = {quota.id: quota for quota in result.quotas}
    assert result.plan == "Business"
    assert result.project_id == "test-project"
    assert quotas["gemini-pro-agent"].remaining_percentage == 25
    assert quotas["gemini-pro-agent"].source == "retrieveUserQuota"
    assert quotas["claude-sonnet-4-6"].unlimited is False
    assert quotas["gemini_weekly"].remaining == 100
    assert "chat_20706" not in quotas


@respx.mock
def test_model_alias_uses_callable_upstream_id(auth):
    route = respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(200, text=sse())
    litellm.completion(model="antigravity/gemini-3.1-pro-high", messages=[{"role": "user", "content": "Hi"}])
    assert json.loads(route.calls.last.request.content)["model"] == "gemini-pro-agent"


@respx.mock
def test_request_normalization_matches_antigravity_contract(auth):
    route = respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(200, text=sse())
    litellm.completion(
        model="antigravity/gemini-pro-agent",
        messages=[
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Prefill"},
        ],
        max_tokens=100000,
    )
    request = json.loads(route.calls.last.request.content)["request"]
    assert len(request["contents"]) == 1
    assert [part["text"] for part in request["contents"][0]["parts"]] == ["First", "Second"]
    assert request["generationConfig"]["topK"] == 40
    assert request["generationConfig"]["topP"] == 1.0
    assert request["generationConfig"]["maxOutputTokens"] == 65535


@respx.mock
def test_claude_output_is_capped_for_antigravity(auth):
    route = respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(200, text=sse())
    litellm.completion(
        model="antigravity/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100000,
    )
    generation = json.loads(route.calls.last.request.content)["request"]["generationConfig"]
    assert generation["maxOutputTokens"] == 16384
    assert "thinkingConfig" not in generation


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@respx.mock
async def test_async_completion(auth, stream):
    respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(200, text=sse())
    result = await litellm.acompletion(
        model="antigravity/test-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    )
    response = litellm.stream_chunk_builder([chunk async for chunk in result]) if stream else result
    assert response.choices[0].message.content == "Hello world"
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
@respx.mock
async def test_proxy_wildcard_deployment_routes_native_provider(auth):
    route = respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(200, text=sse())
    router = litellm.Router(
        model_list=[{"model_name": "antigravity/*", "litellm_params": {"model": "antigravity/*", "max_retries": 0}}]
    )
    response = await router.acompletion(model="antigravity/test-model", messages=[{"role": "user", "content": "Hi"}])
    assert response.choices[0].message.content == "Hello world"
    assert json.loads(route.calls.last.request.content)["model"] == "test-model"


@respx.mock
def test_refresh_rotates_and_persists_with_original_client(auth):
    stored = auth.read()
    auth.save(stored.model_copy(update={"expires_at": 0, "client_id": "original-client"}))
    route = respx.post(TOKEN_URL).respond(
        200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = tuple(executor.map(lambda _: auth.credentials(), range(5)))
    assert all(value.access_token == "new-access" for value in results)
    assert route.call_count == 1
    assert parse_qs(route.calls.last.request.content.decode())["client_id"] == ["original-client"]
    assert auth.read().refresh_token == "new-refresh"
    assert stat.S_IMODE((auth.directory / "default.json").stat().st_mode) == 0o600


@respx.mock
def test_failed_refresh_preserves_credentials_and_redacts(auth):
    stored = auth.read().model_copy(update={"expires_at": 0})
    auth.save(stored)
    route = respx.post(TOKEN_URL).respond(400, json={"error": "invalid_grant", "detail": "test-refresh"})
    with pytest.raises(AntigravityError) as error:
        auth.credentials()
    assert "test-refresh" not in str(error.value)
    assert auth.read().refresh_token == stored.refresh_token
    assert auth.read().requires_login is True
    with pytest.raises(AntigravityError):
        auth.credentials()
    assert route.call_count == 1


@respx.mock
def test_project_discovery_after_onboarding(auth):
    respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").mock(
        side_effect=[
            httpx.Response(200, json={"allowedTiers": [{"id": "free-tier", "isDefault": True}]}),
            httpx.Response(200, json={"cloudaicompanionProject": {"id": "discovered"}}),
        ]
    )
    route = respx.post(BOOTSTRAP_URL + "/v1internal:onboardUser").respond(200, json={"done": True})
    result = auth.discover_project()
    assert result.project_id == "discovered"
    assert json.loads(route.calls.last.request.content)["tier_id"] == "free-tier"
    assert auth.read().project_id == "discovered"


def test_sse_multiline_and_invalid_payload():
    assert list(
        unwrap_lines(iter([": heartbeat", 'data: {"response":', 'data: {"candidates": []}}', "", "data: [DONE]", ""]))
    ) == ['{"candidates": []}']
    with pytest.raises(AntigravityError):
        list(unwrap_lines(iter(["data: broken", ""])))


@respx.mock
def test_tool_call_signature_round_trip(auth):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Read weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    frame = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "weather", "args": {"city": "Bangkok"}},
                                "thoughtSignature": "provider-signature",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    }
    route = respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").mock(
        side_effect=[httpx.Response(200, text="data: " + json.dumps(frame) + "\n\n"), httpx.Response(200, text=sse())]
    )
    first = litellm.completion(
        model="antigravity/test-model", messages=[{"role": "user", "content": "Weather in Bangkok?"}], tools=tools
    )
    assert first.choices[0].finish_reason == "tool_calls"
    assistant = first.choices[0].message
    call = assistant.tool_calls[0]
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Bangkok"}
    litellm.completion(
        model="antigravity/test-model",
        messages=[
            {"role": "user", "content": "Weather in Bangkok?"},
            assistant.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "Sunny"},
        ],
        tools=tools,
    )
    body = json.loads(route.calls.last.request.content)
    parts = [part for content in body["request"]["contents"] for part in content["parts"]]
    assert any(part.get("thoughtSignature") == "provider-signature" for part in parts)
    assert any(
        part.get("functionResponse", part.get("function_response", {})).get("name") == "weather" for part in parts
    ), parts


@respx.mock
def test_upstream_failure_is_not_an_empty_success(auth):
    respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(
        200, text='data: {"response":{"error":{"code":429,"message":"Quota exhausted"}}}\n\n'
    )
    with pytest.raises(litellm.RateLimitError):
        litellm.completion(model="antigravity/test-model", messages=[{"role": "user", "content": "Hi"}], num_retries=0)


@respx.mock
def test_http_error_surfaces_upstream_detail(auth):
    respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(
        400, json={"error": {"message": "Model not found: chat_20706"}}
    )
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(model="antigravity/test-model", messages=[{"role": "user", "content": "Hi"}], num_retries=0)
    assert "Model not found: chat_20706" in str(exc_info.value)


@respx.mock
def test_browser_login_pkce_state_and_replay(auth):
    app = FastAPI()
    app.include_router(create_login_router(auth, "http://localhost:4000"))
    with TestClient(
        app, base_url="http://localhost:4000", follow_redirects=False, client=("127.0.0.1", 50000)
    ) as client:
        page = client.get("/antigravity")
        assert page.headers["referrer-policy"] == "same-origin"
        assert client.post("/antigravity/login").status_code == 403
        assert client.post("/antigravity/login", headers={"origin": "null"}).status_code == 403
        assert client.post("/antigravity/login", headers={"origin": "https://example.com"}).status_code == 403
        login = client.post(
            "/antigravity/login?return_to=/ui/models-and-endpoints%3Fantigravity%3Dconnected",
            headers={"origin": "http://localhost:4000"},
        )
        params = parse_qs(urlparse(login.headers["location"]).query)
        assert params["code_challenge_method"] == ["S256"]
        assert params["access_type"] == ["offline"]
        assert client.get("/antigravity/callback?state=wrong&code=x").status_code == 400
        token = respx.post(TOKEN_URL).respond(
            200, json={"access_token": "signed-in", "refresh_token": "offline", "expires_in": 3600}
        )
        respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").respond(
            200, json={"cloudaicompanionProject": "signed-in-project"}
        )
        callback = "/antigravity/callback?state=" + params["state"][0] + "&code=authorization-code"
        result = client.get(callback)
        assert result.status_code == 303
        assert result.headers["location"] == "http://localhost:4000/ui/models-and-endpoints?antigravity=connected"
        assert auth.read().project_id == "signed-in-project"
        form = parse_qs(token.calls.last.request.content.decode())
        assert form["redirect_uri"] == params["redirect_uri"]
        assert form["code_verifier"]
        assert client.get(callback).status_code == 400
        assert "access_token" not in client.get("/antigravity/status").text


class TestModels:

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("gemini-3.7-flash", "gemini-3.7-flash-tiered"),
            ("gemini-3.7-flash-high", "gemini-3.7-flash-tiered"),
            ("gemini-3.1-pro-high", "gemini-pro-agent"),
            ("gpt-oss-120b", "gpt-oss-120b-medium"),
            ("gemini-claude-sonnet-4-5", "claude-sonnet-4-6"),
        ],
    )
    def test_resolve_model_id_applies_aliases(self, alias, expected):
        assert resolve_model_id(alias) == expected

    def test_resolve_model_id_passes_unknown_through(self):
        assert resolve_model_id("unknown-model") == "unknown-model"

    @pytest.mark.parametrize(
        "model_id,information,expected",
        [
            ("gemini-pro-agent", {"displayName": "Gemini Pro"}, True),
            ("chat_20706", {"isInternal": True}, False),
            ("gemini-3.1-flash-image", {}, False),
            ("tab_flash_lite_preview", {}, False),
            ("gemini-2.5-pro", {}, False),
            ("gemini-3.5-flash", {}, False),
            ("some-imagen-model", {}, False),
            ("some-tts-model", {}, False),
            ("embedding-model", {}, False),
            ("chat_12345", {}, False),
            ("", {}, False),
        ],
    )
    def test_is_discoverable_model(self, model_id, information, expected):
        assert is_discoverable_model(model_id, information) is expected

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("claude-sonnet-4-6", 16384),
            ("claude-opus-4-6-thinking", 16384),
            ("gpt-oss-120b-medium", 32768),
            ("gemini-3.7-flash-tiered", 65535),
            ("gemini-pro-agent", 65535),
            ("unknown-model", 65536),
        ],
    )
    def test_output_token_cap(self, model_id, expected):
        assert output_token_cap(model_id) == expected


class TestAuthenticator:

    def test_read_returns_none_for_missing_file(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        assert authenticator.read("nonexistent") is None
        authenticator.client.close()

    def test_save_and_read_round_trip(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        credentials = Credentials(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() + 3600,
            client_id="cid",
            project_id="proj",
        )
        authenticator.save(credentials)
        loaded = authenticator.read()
        assert loaded is not None
        assert loaded.access_token == "tok"
        assert loaded.refresh_token == "ref"
        assert loaded.client_id == "cid"
        assert loaded.project_id == "proj"
        authenticator.client.close()

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "deep" / "dir"
        authenticator = Authenticator(nested, httpx.Client(timeout=5))
        credentials = Credentials(
            access_token="tok", refresh_token="ref", expires_at=time.time() + 3600, client_id="cid"
        )
        authenticator.save(credentials)
        assert nested.exists()
        assert authenticator.read() is not None
        authenticator.client.close()

    def test_invalid_account_name_raises(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        with pytest.raises(AntigravityError, match="Invalid"):
            authenticator.read("../escape")
        with pytest.raises(AntigravityError, match="Invalid"):
            authenticator.read("")
        with pytest.raises(AntigravityError, match="Invalid"):
            authenticator.read("a" * 81)
        authenticator.client.close()

    def test_credentials_raises_when_not_signed_in(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        with pytest.raises(AntigravityError, match="Sign in"):
            authenticator.credentials()
        authenticator.client.close()

    def test_credentials_raises_when_requires_login_flag_set(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        authenticator.save(
            Credentials(
                access_token="tok",
                refresh_token="ref",
                expires_at=time.time() + 3600,
                client_id="cid",
                project_id="proj",
                requires_login=True,
            )
        )
        with pytest.raises(AntigravityError, match="Sign in"):
            authenticator.credentials()
        authenticator.client.close()

    def test_credentials_raises_when_project_missing_and_required(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        authenticator.save(
            Credentials(
                access_token="tok",
                refresh_token="ref",
                expires_at=time.time() + 3600,
                client_id="cid",
                project_id="",
            )
        )
        with pytest.raises(AntigravityError, match="project"):
            authenticator.credentials()
        authenticator.client.close()

    def test_credentials_allows_no_project_when_not_required(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        authenticator.save(
            Credentials(
                access_token="tok",
                refresh_token="ref",
                expires_at=time.time() + 3600,
                client_id="cid",
                project_id="",
            )
        )
        result = authenticator.credentials(require_project=False)
        assert result.access_token == "tok"
        authenticator.client.close()

    @respx.mock
    def test_credentials_auto_refreshes_when_expired(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        authenticator.save(
            Credentials(
                access_token="old-tok",
                refresh_token="ref",
                expires_at=time.time() - 100,
                client_id="cid",
                project_id="proj",
            )
        )
        respx.post(TOKEN_URL).respond(
            200, json={"access_token": "new-tok", "refresh_token": "new-ref", "expires_in": 3600}
        )
        result = authenticator.credentials()
        assert result.access_token == "new-tok"
        assert result.refresh_token == "new-ref"
        persisted = authenticator.read()
        assert persisted is not None
        assert persisted.access_token == "new-tok"
        authenticator.client.close()

    @respx.mock
    def test_refresh_marks_requires_login_on_invalid_grant(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        authenticator.save(
            Credentials(
                access_token="old-tok",
                refresh_token="ref",
                expires_at=time.time() - 100,
                client_id="cid",
                project_id="proj",
            )
        )
        respx.post(TOKEN_URL).respond(400, json={"error": "invalid_grant"})
        with pytest.raises(ReauthenticationRequired):
            authenticator.credentials()
        stored = authenticator.read()
        assert stored is not None
        assert stored.requires_login is True
        authenticator.client.close()

    def test_authorize_url_raises_without_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTIGRAVITY_OAUTH_CLIENT_ID", raising=False)
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        with pytest.raises(AntigravityError, match="ANTIGRAVITY_OAUTH_CLIENT_ID"):
            authenticator.authorize_url("http://localhost/callback", "state", "challenge")
        authenticator.client.close()

    def test_authorize_url_includes_correct_params(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTIGRAVITY_OAUTH_CLIENT_ID", "test-client-id")
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=5))
        url = authenticator.authorize_url("http://localhost/callback", "test-state", "test-challenge")
        assert url.startswith(AUTHORIZE_URL)
        params = parse_qs(urlparse(url).query)
        assert params["client_id"] == ["test-client-id"]
        assert params["redirect_uri"] == ["http://localhost/callback"]
        assert params["response_type"] == ["code"]
        assert params["state"] == ["test-state"]
        assert params["access_type"] == ["offline"]
        assert params["code_challenge"] == ["test-challenge"]
        assert params["code_challenge_method"] == ["S256"]
        authenticator.client.close()

    def test_content_headers_include_bearer_token(self):
        headers = content_headers("my-token")
        assert headers["Authorization"] == "Bearer my-token"
        assert headers["Content-Type"] == "application/json"
        assert "antigravity" in headers["User-Agent"].lower()

    def test_bootstrap_metadata_returns_valid_ids(self):
        meta = bootstrap_metadata()
        assert "ideType" in meta
        assert "platform" in meta
        assert meta["ideType"] == 9


class TestUsageHelpers:

    @pytest.mark.parametrize(
        "value,expected",
        [
            (42, 42.0),
            (3.14, 3.14),
            ("100", 100.0),
            ("bad", -1),
            (True, -1),
            (None, -1),
            ([], -1),
        ],
    )
    def test_number(self, value, expected):
        assert _number(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (0, None),
            (-1, None),
            (True, None),
            ("", None),
            ("2030-01-01T00:00:00Z", "2030-01-01T00:00:00Z"),
            (1893456000, "2030-01-01T00:00:00Z"),
            (1893456000000, "2030-01-01T00:00:00Z"),
        ],
    )
    def test_reset_time(self, value, expected):
        assert _reset_time(value) == expected

    def test_quota_returns_none_for_missing_fraction(self):
        assert _quota("test-model", {}, "test-source") is None

    def test_quota_returns_populated_entry(self):
        result = _quota("test-model", {"remainingFraction": 0.5, "resetTime": "2030-01-01T00:00:00Z"}, "catalog")
        assert result is not None
        assert result.id == "test-model"
        assert result.remaining_percentage == 50.0
        assert result.unlimited is False
        assert result.reset_at == "2030-01-01T00:00:00Z"

    def test_quota_unlimited_when_full_and_no_reset(self):
        result = _quota("test-model", {"remainingFraction": 1.0}, "catalog")
        assert result is not None
        assert result.unlimited is True

    @pytest.mark.parametrize(
        "subscription,expected",
        [
            ({"currentTier": {"id": "ultra-tier"}}, "Ultra"),
            ({"currentTier": {"name": "Pro Plan"}}, "Pro"),
            ({"paidTier": "premium-tier"}, "Pro"),
            ({"subscriptionTier": "google_one_plan"}, "Pro"),
            ({"currentTier": "enterprise-suite"}, "Enterprise"),
            ({"currentTier": "business-standard"}, "Business"),
            ({"currentTier": "standard-tier"}, "Business"),
            ({"currentTier": "plus-plan"}, "Plus"),
            ({"currentTier": "lite-plan"}, "Lite"),
            ({}, "Free"),
            ({"currentTier": "unknown"}, "Free"),
        ],
    )
    def test_plan_detection(self, subscription, expected):
        assert _plan(subscription) == expected


class TestUsageCache:

    @respx.mock
    def test_usage_cache_returns_cached_result(self, auth):
        respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").respond(200, json={})
        respx.post(DISCOVERY_URLS[0]).respond(200, json={"models": {}})
        respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuota").respond(200, json={})
        respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuotaSummary").respond(200, json={})
        first = get_usage(auth)
        second = get_usage(auth)
        assert first.checked_at == second.checked_at

    @respx.mock
    def test_usage_force_refresh_bypasses_cache(self, auth):
        respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").respond(200, json={})
        respx.post(DISCOVERY_URLS[0]).respond(200, json={"models": {}})
        respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuota").respond(200, json={})
        respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuotaSummary").respond(200, json={})
        first = get_usage(auth)
        refreshed = get_usage(auth, force_refresh=True)
        assert refreshed.checked_at >= first.checked_at


class TestLoginRouter:

    def _app(self, auth):
        app = FastAPI()
        app.include_router(create_login_router(auth, "http://localhost:4000"))
        return app

    def test_status_returns_signed_in_state(self, auth):
        with TestClient(
            self._app(auth), base_url="http://localhost:4000", client=("127.0.0.1", 50000)
        ) as client:
            response = client.get("/antigravity/status")
            data = response.json()
            assert data["signed_in"] is True
            assert data["project_id"] == "test-project"

    def test_status_returns_not_signed_in(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTIGRAVITY_AUTH_DIR", str(tmp_path))
        monkeypatch.setenv("ANTIGRAVITY_OAUTH_CLIENT_ID", "test-client")
        get_authenticator.cache_clear()
        authenticator = get_authenticator()
        try:
            with TestClient(
                self._app(authenticator), base_url="http://localhost:4000", client=("127.0.0.1", 50000)
            ) as client:
                response = client.get("/antigravity/status")
                data = response.json()
                assert data["signed_in"] is False
                assert data["project_id"] == ""
        finally:
            authenticator.client.close()
            get_authenticator.cache_clear()

    @respx.mock
    def test_models_endpoint_returns_discovered_models(self, auth):
        respx.post(DISCOVERY_URLS[0]).respond(
            200, json={"models": {"gemini-pro-agent": {"displayName": "Pro"}}}
        )
        with TestClient(
            self._app(auth), base_url="http://localhost:4000", client=("127.0.0.1", 50000)
        ) as client:
            response = client.get("/antigravity/models")
            assert response.status_code == 200
            data = response.json()
            assert "gemini-pro-agent" in data["models"]

    @respx.mock
    def test_project_endpoint_discovers_project(self, auth):
        respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").respond(
            200, json={"cloudaicompanionProject": "discovered-project"}
        )
        with TestClient(
            self._app(auth), base_url="http://localhost:4000", client=("127.0.0.1", 50000)
        ) as client:
            response = client.post(
                "/antigravity/project",
                json={"project_id": ""},
                headers={"origin": "http://localhost:4000"},
            )
            assert response.status_code == 200
            assert response.json()["project_id"] == "discovered-project"

    @respx.mock
    def test_usage_endpoint(self, auth):
        respx.post(BOOTSTRAP_URL + "/v1internal:loadCodeAssist").respond(200, json={})
        respx.post(DISCOVERY_URLS[0]).respond(200, json={"models": {}})
        respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuota").respond(200, json={})
        respx.post(RUNTIME_URL + "/v1internal:retrieveUserQuotaSummary").respond(200, json={})
        with TestClient(
            self._app(auth), base_url="http://localhost:4000", client=("127.0.0.1", 50000)
        ) as client:
            response = client.get("/antigravity/usage")
            assert response.status_code == 200
            data = response.json()
            assert "plan" in data
            assert "quotas" in data

    def test_page_returns_html(self, auth):
        with TestClient(
            self._app(auth), base_url="http://localhost:4000", client=("127.0.0.1", 50000)
        ) as client:
            response = client.get("/antigravity")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_login_cancelled_redirects(self, auth):
        with TestClient(
            self._app(auth), base_url="http://localhost:4000", follow_redirects=False, client=("127.0.0.1", 50000)
        ) as client:
            login = client.post(
                "/antigravity/login", headers={"origin": "http://localhost:4000"}
            )
            params = parse_qs(urlparse(login.headers["location"]).query)
            state = params["state"][0]
            callback = client.get(f"/antigravity/callback?state={state}&error=access_denied")
            assert callback.status_code == 303
            assert "cancelled" in callback.headers["location"]


class TestLoginSessions:

    def test_create_returns_attempt_with_state(self):
        sessions = LoginSessions()
        attempt = sessions.create()
        assert len(attempt.state) > 20
        assert len(attempt.verifier) > 40

    def test_consume_removes_attempt(self):
        sessions = LoginSessions()
        attempt = sessions.create()
        consumed = sessions.consume(attempt.state, attempt.state)
        assert consumed.state == attempt.state
        with pytest.raises(Exception):
            sessions.consume(attempt.state, attempt.state)

    def test_consume_rejects_wrong_cookie(self):
        sessions = LoginSessions()
        attempt = sessions.create()
        with pytest.raises(Exception):
            sessions.consume(attempt.state, "wrong-cookie")

    def test_create_sanitizes_return_to(self):
        sessions = LoginSessions()
        attempt = sessions.create(return_to="https://evil.com")
        assert attempt.return_to == ""

    def test_create_allows_ui_return_to(self):
        sessions = LoginSessions()
        attempt = sessions.create(return_to="/ui/models")
        assert attempt.return_to == "/ui/models"

    def test_create_rejects_double_slash_return_to(self):
        sessions = LoginSessions()
        attempt = sessions.create(return_to="//evil.com")
        assert attempt.return_to == ""


class TestSSEParsing:

    def test_unwrap_lines_skips_empty_and_non_data(self):
        lines = [
            "",
            ": comment",
            "data: {\"response\":{\"candidates\":[]}}",
            "",
            "data: {\"response\":{\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}}",
        ]
        results = list(unwrap_lines(iter(lines)))
        assert len(results) == 2

    def test_unwrap_lines_rejects_invalid_json(self):
        lines = ['data: not-json']
        with pytest.raises(AntigravityError):
            list(unwrap_lines(iter(lines)))


class TestConcurrentRefresh:

    @respx.mock
    def test_concurrent_credentials_hit_token_once(self, tmp_path):
        authenticator = Authenticator(tmp_path, httpx.Client(timeout=30))
        authenticator.save(
            Credentials(
                access_token="expired",
                refresh_token="ref",
                expires_at=time.time() - 100,
                client_id="cid",
                project_id="proj",
            )
        )
        call_count = {"n": 0}

        def slow_token(request):
            call_count["n"] += 1
            time.sleep(0.1)
            return httpx.Response(200, json={"access_token": "refreshed", "expires_in": 3600})

        respx.post(TOKEN_URL).mock(side_effect=slow_token)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(authenticator.credentials) for _ in range(4)]
            results = [f.result() for f in futures]

        assert all(r.access_token == "refreshed" for r in results)
        assert call_count["n"] <= 2
        authenticator.client.close()


class TestAsyncCompletion:

    @respx.mock
    @pytest.mark.asyncio
    async def test_acompletion_calls_code_assist(self, auth):
        respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(
            200, text=sse(), headers={"content-type": "text/event-stream"}
        )
        result = await litellm.acompletion(
            model="antigravity/test-model", messages=[{"role": "user", "content": "Hi"}]
        )
        assert result.choices[0].message.content == "Hello world"
        assert result.usage.total_tokens == 6

    @respx.mock
    @pytest.mark.asyncio
    async def test_acompletion_stream(self, auth):
        respx.post(RUNTIME_URL + "/v1internal:streamGenerateContent?alt=sse").respond(
            200, text=sse(), headers={"content-type": "text/event-stream"}
        )
        result = await litellm.acompletion(
            model="antigravity/test-model", messages=[{"role": "user", "content": "Hi"}], stream=True
        )
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        combined = litellm.stream_chunk_builder(chunks)
        assert combined.choices[0].message.content == "Hello world"
