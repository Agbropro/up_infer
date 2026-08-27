"""Run UP Infer using values from config.yaml."""

import uvicorn

from app.application.config import load_config


def start_server() -> None:
    """Start the configured Uvicorn server."""

    config = load_config()
    server = config.get("server", {})
    domain = str(server.get("domain", "127.0.0.1"))
    port = int(server.get("port", 8000))
    reload = bool(server.get("reload", False))
    uvicorn.run("app.main:app", host=domain, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
