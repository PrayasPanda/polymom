"""Error envelope schema, used for OpenAPI documentation."""

from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_example(
    status: int, code: str, message: str, description: str
) -> dict[int | str, dict[str, Any]]:
    """Build an OpenAPI ``responses`` entry for an error status."""
    return {
        status: {
            "model": ErrorResponse,
            "description": description,
            "content": {
                "application/json": {
                    "example": {"error": {"code": code, "message": message, "details": None}}
                }
            },
        }
    }
