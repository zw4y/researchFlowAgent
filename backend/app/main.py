import contextlib
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.container import AppContainer
from app.core.config import get_settings
from app.core.errors import install_exception_handlers
from app.core.logging import configure_logging
from app.mcp_server import configure_mcp, mcp

settings = get_settings()
configure_logging(settings.log_level)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = AppContainer(settings)
    await container.start()
    configure_mcp(container)
    app.state.container = container
    async with mcp.session_manager.run():
        yield
    await container.close()


app = FastAPI(
    title="ResearchFlow Agent API",
    version="0.1.0",
    description="A traceable research workflow agent for papers and technical documents.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


install_exception_handlers(app)
app.include_router(router, prefix=settings.api_prefix)
app.mount("/mcp", mcp.streamable_http_app())
