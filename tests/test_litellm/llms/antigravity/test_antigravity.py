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
    BOOTSTRAP_URL,
    RUNTIME_URL,
    TOKEN_URL,
    AntigravityError,
    Authenticator,
    Credentials,
    get_authenticator,
)
from litellm.llms.antigravity.chat.transformation import unwrap_lines
from litellm.llms.antigravity.login import create_login_router


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
    assert "stream" not in body and "stream" not in body["request"]
    assert route.calls.last.request.headers["authorization"] == "Bearer test-access"
    assert "test-refresh" not in route.calls.last.request.content.decode()


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
def test_browser_login_pkce_state_and_replay(auth):
    app = FastAPI()
    app.include_router(create_login_router(auth, "http://localhost:4000"))
    with TestClient(
        app, base_url="http://localhost:4000", follow_redirects=False, client=("127.0.0.1", 50000)
    ) as client:
        assert client.post("/antigravity/login").status_code == 403
        login = client.post("/antigravity/login", headers={"origin": "http://localhost:4000"})
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
        assert auth.read().project_id == "signed-in-project"
        form = parse_qs(token.calls.last.request.content.decode())
        assert form["redirect_uri"] == params["redirect_uri"]
        assert form["code_verifier"]
        assert client.get(callback).status_code == 400
        assert "access_token" not in client.get("/antigravity/status").text
