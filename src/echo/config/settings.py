"""Configuration settings for the ECHO Multi-Agent System.

This module defines a Pydantic ``BaseSettings`` model used to configure the
application via environment variables and a ``.bin`` file. Environment
variables are read with the ``ECHO_`` prefix (case-insensitive), and field
descriptions serve as the authoritative documentation for each setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support and validation.

    Notes:
    - Values can be provided via environment variables with prefix ``ECHO_``
      (e.g., ``ECHO_API_PORT=8080``), or from a ``.bin`` file.
    - Configuration is case-insensitive and validates assignments at runtime.
    - See ``model_config`` for environment loading behavior.
    """

    app_name: str = Field(default="ECHO Multi-Agent System", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")

    default_llm_provider: str = Field(default="openai", description="Default LLM provider")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

    plugins_dir: str = Field(default="./src/echo-plugins", description="Plugin directory path")
    enable_git_plugins: bool = Field(default=True, description="Enable Git-based plugins")
    enable_plugin_validation: bool = Field(default=True, description="Enable plugin validation")

    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")

    session_timeout: int = Field(default=3600, ge=60, description="Session timeout in seconds")
    max_session_history: int = Field(default=100, ge=1, description="Maximum session history length")

    max_hops: int = Field(default=10, ge=1, le=50, description="Maximum processing hops")

    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        """Validate that the configured default LLM provider is supported.

        Args:
            value: Provider name supplied via settings/bin (e.g., "openai").

        Returns:
            The validated provider name.

        Raises:
            ValueError: If the provider is not one of the supported options.
        """
        supported_providers = ["openai", "anthropic", "google"]
        if value not in supported_providers:
            raise ValueError(
                f"Unsupported LLM provider: {value}. " f"Supported providers: {', '.join(supported_providers)}"
            )
        return value

    @field_validator("plugins_dir")
    @classmethod
    def validate_plugins_dir(cls, value: str) -> str:
        """Ensure the plugins directory exists or can be created.

        Args:
            value: Filesystem path to the plugins directory.

        Returns:
            The original path after ensuring it exists.
        """
        plugins_path = Path(value)
        if not plugins_path.exists():
            plugins_path.mkdir(parents=True, exist_ok=True)
        return value

    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Return the API key for the specified provider.

        Provider aliases are supported: "claude" maps to Anthropic, and
        "gemini" maps to Google.

        Args:
            provider: Provider name or alias (e.g., "openai", "anthropic",
                "claude", "google", "gemini").

        Returns:
            The API key string if configured; otherwise, ``None``.
        """
        provider_key_mapping = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "claude": self.anthropic_api_key,
            "google": self.google_api_key,
            "gemini": self.google_api_key,
        }
        return provider_key_mapping.get(provider)

    def validate_provider_credentials(self, provider: str) -> bool:
        """Check whether credentials exist for the specified provider.

        Args:
            provider: Provider name or alias.

        Returns:
            True if a non-empty API key is configured; otherwise, False.
        """
        api_key = self.get_api_key_for_provider(provider)
        return api_key is not None and len(api_key.strip()) > 0

    model_config = {
        "env_file": ".bin",
        "env_prefix": "ECHO_",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore",
    }
