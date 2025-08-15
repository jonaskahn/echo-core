"""Application entrypoint and FastAPI app factory for ECHO.

Defines the `EchoApplication` which:

- Configures logging and CORS middleware
- Initializes API services during app lifespan (LLM factory, plugin manager,
  orchestrator via `initialize_api()`)
- Registers HTTP routes from `src/echo/api/routes.py`

Documentation-only changes; runtime behavior remains the same.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router, initialize_api
from .config.settings import Settings


class EchoApplication:
    """Create and run the ECHO FastAPI application.

    Responsibilities:
    - Provide lifecycle hooks to initialize services
    - Configure CORS and include API routes
    - Expose `create_app()` and `run()` helpers
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.app: FastAPI | None = None
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _setup_logging() -> None:
        """Configure application logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _create_lifespan_manager(self):
        """Create an async lifespan manager that initializes services.

        Returns:
            Callable: An async context manager suitable for FastAPI's
            `lifespan` parameter.
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.logger.info("Starting ECHO Multi-Agent System...")
            initialize_api(self.settings)
            self.logger.info("ECHO Multi-Agent System started successfully")
            yield
            self.logger.info("Shutting down ECHO Multi-Agent System...")

        return lifespan

    def _configure_middleware(self) -> None:
        """Configure middleware such as permissive CORS for bin.

        Uses origins from `Settings.cors_origins`.
        """
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self) -> None:
        """Register application routes including the root health/info route."""
        self.app.include_router(router)

        @self.app.get("/")
        async def root() -> Dict[str, Any]:
            """Root endpoint providing system information."""
            return {
                "message": "ECHO Multi-Agent System",
                "version": "1.0.0",
                "status": "running",
            }

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application instance.

        Returns:
            FastAPI: Configured app with routes, middleware, and lifespan.
        """
        self.app = FastAPI(
            title="ECHO Multi-Agent System",
            description="Plugin-based multi-agent chatbot system",
            version="1.0.0",
            lifespan=self._create_lifespan_manager(),
        )

        self._configure_middleware()
        self._register_routes()

        return self.app

    def run(self) -> None:
        """Run the application server with Uvicorn.

        Ensures the app is created before starting. Honors host/port/reload
        from `Settings`.
        """
        if not self.app:
            self.create_app()

        if self.app is None:
            raise RuntimeError("Failed to create FastAPI application")

        uvicorn.run(
            self.app,
            host=self.settings.api_host,
            port=self.settings.api_port,
            reload=self.settings.debug,
            log_level="info" if not self.settings.debug else "debug",
        )


# Global application instance
echo_app = EchoApplication()
app = echo_app.create_app()


def main() -> None:
    """Main entry point for the application."""
    echo_app.run()


if __name__ == "__main__":
    main()
