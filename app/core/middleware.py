import uuid
import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.services.firebase import verify_id_token

logger = logging.getLogger(__name__)

# Paths that don't require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                content='{"error":"Missing or invalid Authorization header"}',
                status_code=401,
                media_type="application/json",
            )

        token = auth_header.removeprefix("Bearer ").strip()
        try:
            claims = verify_id_token(token)
        except Exception:
            return Response(
                content='{"error":"Invalid or expired token"}',
                status_code=401,
                media_type="application/json",
            )

        request.state.firebase_uid = claims["uid"]
        request.state.firebase_email = claims.get("email", "")
        return await call_next(request)
