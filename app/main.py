"""Start the UP Infer FastAPI application."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from app.interfaces.api import router


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="UP Infer", version="1.0.0")
app.include_router(router)
home_page = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
style_sheet = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
browser_code = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")


@app.get("/", include_in_schema=False)
async def show_home() -> HTMLResponse:
    """Serve the UP Infer interface."""

    return HTMLResponse(home_page)


@app.get("/static/styles.css", include_in_schema=False)
async def show_styles() -> Response:
    """Serve the UP Infer style sheet."""

    return Response(style_sheet, media_type="text/css")


@app.get("/static/app.js", include_in_schema=False)
async def show_script() -> Response:
    """Serve the UP Infer browser code."""

    return Response(browser_code, media_type="text/javascript")


@app.get("/health", include_in_schema=False)
async def get_health() -> dict[str, str]:
    """Report basic application health."""

    return {"status": "ok"}
