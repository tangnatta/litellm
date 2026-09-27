# Antigravity pilot

This provider connects LiteLLM directly to Google Code Assist using an Antigravity OAuth account. It does not require an OmniRoute server

The pilot includes browser login with OAuth state and PKCE, local credential storage, refresh-token rotation, project discovery/onboarding, model discovery, and synchronous/asynchronous chat with streaming and tool calls

Model discovery follows the Antigravity hosts and both catalog methods used by OmniRoute. It filters internal, retired, opaque, and non-chat entries, and uses a curated callable catalog when live discovery is unavailable. The request adapter also applies Antigravity model aliases, native request/session identity, generation limits, conversation cleanup, tool validation, and model-specific thinking rules

The login page also reports subscription and quota data from the same Antigravity endpoints used by OmniRoute. It combines live per-model quota, catalog quota fallbacks, and weekly model-family limits. Results are cached for one minute; **Refresh usage** bypasses that cache

## Run locally

From the repository root, install LiteLLM and its proxy dependencies:

```sh
uv sync --extra proxy
```

Configure the public desktop OAuth client used for Antigravity. These are application credentials, not the user's Google account credentials:

```sh
export ANTIGRAVITY_OAUTH_CLIENT_ID='<desktop-oauth-client-id>'
export ANTIGRAVITY_OAUTH_CLIENT_SECRET='<desktop-oauth-client-secret>'
```

Set a master key when using the standard LiteLLM dashboard. Sign in to `/ui` with username `admin` and this value as the password:

```sh
export LITELLM_MASTER_KEY='sk-change-this-local-key'
```

Build validation and tests can be run before starting the service:

```sh
uv run ruff check litellm/llms/antigravity
LITELLM_LOCAL_MODEL_COST_MAP=True uv run pytest \
  tests/test_litellm/llms/antigravity/test_antigravity.py -q
```

Start the local LiteLLM proxy and login page:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True \
  uv run python -m litellm.llms.antigravity --port 4000
```

To store account tokens somewhere other than the default directory, set `ANTIGRAVITY_AUTH_DIR` before starting the service.

Open [the login page](http://localhost:4000/antigravity) in your regular browser and choose **Sign in with Google**. After consent, the page discovers your project and available models. If Google requires an existing Cloud project, enter its project ID and choose **Discover or save project**. Select a model and send a message from the same page

The same connection controls are integrated into the standard LiteLLM dashboard at [Models & Endpoints](http://localhost:4000/ui/models-and-endpoints). Open the **Antigravity** tab to sign in, discover or change the project, refresh the callable model list, and inspect provider quota. OAuth returns directly to that dashboard tab

The callback is `http://localhost:4000/antigravity/callback`. Use the same port and browser throughout login. A pending login expires after ten minutes or a server restart

Credentials are stored in `~/.config/litellm/antigravity/default.json` with owner-only permissions. Set `ANTIGRAVITY_AUTH_DIR` to choose a different directory. The OAuth client that issued the credentials is saved with them and reused during refresh

## Docker deployment

The repository includes `docker-compose.oauth.yml`, which builds the production LiteLLM image from this branch and starts it with PostgreSQL. PostgreSQL enables the standard admin dashboard, while a separate named volume persists Antigravity and ChatGPT OAuth credentials across container replacements.

Create the deployment environment from the example and replace every `change-me` value:

```sh
cp .env.oauth.example .env.oauth
```

For a remote deployment, set `LITELLM_PUBLIC_URL` to the externally reachable HTTPS origin, without a trailing slash. Register this callback with the Antigravity OAuth application:

```text
https://your-litellm-host.example/antigravity/callback
```

Build and start the full deployment:

```sh
docker compose --env-file .env.oauth -f docker-compose.oauth.yml up -d --build
docker compose --env-file .env.oauth -f docker-compose.oauth.yml ps
```

Open `LITELLM_PUBLIC_URL/ui`, sign in with username `admin` and the configured `LITELLM_MASTER_KEY`, then use the **Antigravity** or **ChatGPT** tab under **Models & Endpoints**. OAuth tokens are stored in the `oauth_credentials` Docker volume. Dashboard and model configuration data are stored in `oauth_postgres`.

To inspect startup or OAuth errors:

```sh
docker compose --env-file .env.oauth -f docker-compose.oauth.yml logs -f litellm
```

To rebuild after pulling updates while preserving both volumes:

```sh
git pull
docker compose --env-file .env.oauth -f docker-compose.oauth.yml up -d --build
```

To stop the deployment without deleting credentials or database data:

```sh
docker compose --env-file .env.oauth -f docker-compose.oauth.yml down
```

Do not add `--volumes` to that command unless you intend to permanently delete saved OAuth accounts and dashboard data.

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

The local management endpoints are:

| Endpoint | Purpose |
| --- | --- |
| `GET /antigravity/status` | Connection and project status |
| `POST /antigravity/login` / `GET /antigravity/callback` | OAuth login and callback |
| `POST /antigravity/project` | Discover or save the Cloud project |
| `GET /antigravity/models` | Callable model catalog with host fallback |
| `GET /antigravity/usage` | Subscription, model quota, and weekly quota |
| `GET /antigravity/usage?refresh=true` | Force a fresh quota check |

LiteLLM's existing `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, model-list, and Gemini-compatible proxy routes provide the public inference facades that OmniRoute exposes. The Antigravity provider handles the provider-specific request envelope and OAuth credentials behind those routes

## Pilot scope

The bundled launcher binds to loopback, runs one worker, and exposes a local login page alongside the proxy. It is intended for one account on one computer. Credential refresh is coordinated within that process; shared credential files across multiple workers are not supported yet

Runtime requests use the daily Cloud Code endpoint by default. A deployment can set `api_base: https://cloudcode-pa.googleapis.com` to use the standard endpoint. OmniRoute's database-backed multi-account routing and credit accounting, distributed account storage, and remote login are application infrastructure rather than provider endpoints and are outside this single-account local pilot

OAuth client values must be supplied through the environment. The provider does not read another application's account files or embed client credentials. Live Google consent, account eligibility, and generation must be verified using the signed-in account; mocked tests do not establish live-provider compatibility
