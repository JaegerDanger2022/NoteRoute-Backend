import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.exceptions import http_exception_handler
from app.core.middleware import AuthMiddleware, RequestLoggingMiddleware
from app.services.firebase import init_firebase
from app.services.mongodb import close_db, init_db
from app.services import vector_svc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NoteRoute Backend...")
    init_firebase()
    await init_db()
    vector_svc.init_vector_store()
    logger.info("Connected to MongoDB, Firebase, and S3 Vectors")
    yield
    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="NoteRoute API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware — order: outermost first
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method,
        request.url.path,
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


app.add_exception_handler(Exception, _unhandled_exception_handler)

# Routers
from app.api.v1.router import router as v1_router
app.include_router(v1_router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    from datetime import datetime, timezone
    return {
        "status": "ok",
        "service": "noteroute-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
