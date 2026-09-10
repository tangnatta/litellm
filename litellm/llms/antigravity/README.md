# Antigravity pilot

This provider connects LiteLLM directly to Google Code Assist using an Antigravity OAuth account. It does not require an OmniRoute server

The pilot includes browser login with OAuth state and PKCE, local credential storage, refresh-token rotation, project discovery/onboarding, model discovery, and synchronous/asynchronous chat with streaming and tool calls

## Run locally

Install LiteLLM's proxy dependencies, configure `ANTIGRAVITY_OAUTH_CLIENT_ID` and `ANTIGRAVITY_OAUTH_CLIENT_SECRET` for the public desktop OAuth client, then run:

```sh
python -m litellm.llms.antigravity --port 4000
```

Open [the login page](http://localhost:4000/antigravity) in your regular browser and choose **Sign in with Google**. After consent, the page discovers your project and available models. If Google requires an existing Cloud project, enter its project ID and choose **Discover or save project**. Select a model and send a message from the same page

The callback is `http://localhost:4000/antigravity/callback`. Use the same port and browser throughout login. A pending login expires after ten minutes or a server restart

Credentials are stored in `~/.config/litellm/antigravity/default.json` with owner-only permissions. Set `ANTIGRAVITY_AUTH_DIR` to choose a different directory. The OAuth client that issued the credentials is saved with them and reused during refresh

## API and SDK

The pilot runs the actual LiteLLM proxy with a wildcard deployment. Use a model ID discovered on the login page:

```sh
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"antigravity/<discovered-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

```python
import litellm

response = litellm.completion(
    model="antigravity/<discovered-model-id>",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Use `litellm.acompletion` for asynchronous calls and `stream=True` for streaming. Request `stream_options={"include_usage": True}` to receive token usage in the stream. Preserve the complete assistant tool-call message when sending tool results back so Google thought signatures survive

## Pilot scope

The bundled launcher binds to loopback, runs one worker, and exposes a local login page alongside the proxy. It is intended for one account on one computer. Credential refresh is coordinated within that process; shared credential files across multiple workers are not supported yet

Runtime requests use the daily Cloud Code endpoint by default. A deployment can set `api_base: https://cloudcode-pa.googleapis.com` to use the standard endpoint. Automatic endpoint failover, distributed account storage, remote login, multiple-account selection, and the remaining OmniRoute providers are follow-up work

OAuth client values must be supplied through the environment. The provider does not read another application's account files or embed client credentials. Live Google consent, account eligibility, and generation must be verified using the signed-in account; mocked tests do not establish live-provider compatibility
