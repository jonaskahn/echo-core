"""API package exports.

Exposes:
- `router`: FastAPI APIRouter with all endpoints
- `initialize_api()`: wiring for orchestrator, plugin manager, and services
"""

from .routes import router, initialize_api

__all__ = ["router", "initialize_api"]
