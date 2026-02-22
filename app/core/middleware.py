import uuid
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp, Receive, Scope, Send

from app.services.firebase import verify_id_token

logger = logging.getLogger(__name__)

# Paths that don't require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}
# Path prefixes that don't require authentication (OAuth callbacks come from provider servers)
_PUBLIC_PREFIXES = (
    "/api/v1/integrations/notion/callback",
    "/api/v1/integrations/google/callback",
    "/api/v1/integrations/slack/callback",
    "/api/v1/integrations/todoist/callback",
    "/api/v1/integrations/trello/token",
    "/api/v1/users/internal/",
    "/api/v1/slots/internal/",
    "/api/v1/deliver",
    "/api/v1/admin/internal/",
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "%s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response


class AuthMiddleware:
    """Pure ASGI auth middleware — avoids BaseHTTPMiddleware 502 bug."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive, send)

        if request.url.path in _PUBLIC_PATHS or request.url.path.startswith(_PUBLIC_PREFIXES) or request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            response = Response(
                content='{"error":"Missing or invalid Authorization header"}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        token = auth_header.removeprefix("Bearer ").strip()
        try:
            claims = verify_id_token(token)
        except Exception:
            response = Response(
                content='{"error":"Invalid or expired token"}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        scope["state"] = scope.get("state", {})
        scope["state"]["firebase_uid"] = claims["uid"]
        scope["state"]["firebase_email"] = claims.get("email", "")
        await self.app(scope, receive, send)
