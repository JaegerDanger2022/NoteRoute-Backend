from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class AuthError(HTTPException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=401, detail=detail)


class ForbiddenError(HTTPException):
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(status_code=403, detail=detail)


class NotFoundError(HTTPException):
    def __init__(self, detail: str = "Not found"):
        super().__init__(status_code=404, detail=detail)


class TierLimitError(HTTPException):
    def __init__(self, detail: str = "Plan limit reached"):
        super().__init__(status_code=403, detail=detail)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None) or {},
    )
