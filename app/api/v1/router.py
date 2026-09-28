"""Aggregate router for API v1."""

from fastapi import APIRouter

from app.api.v1.routes import health, meetings

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meetings.router)
