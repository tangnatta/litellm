import argparse
import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field


class Arguments(BaseModel):
    port: int = Field(default=4000, ge=1, le=65535)


def main() -> None:
    import uvicorn
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    parser: Final = argparse.ArgumentParser(description="Run the local LiteLLM Antigravity login pilot")
    parser.add_argument("--port", type=int, default=4000)
    arguments: Final = Arguments.model_validate(vars(parser.parse_args()))
    origin: Final = f"http://localhost:{arguments.port}"
    os.environ["CONFIG_FILE_PATH"] = str(Path(__file__).with_name("pilot.yaml"))
    os.environ["LITELLM_PUBLIC_URL"] = origin
    from litellm.proxy.proxy_server import app

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=("localhost", "127.0.0.1"))
    uvicorn.run(app, host="127.0.0.1", port=arguments.port, access_log=False, proxy_headers=False)


if __name__ == "__main__":
    main()
