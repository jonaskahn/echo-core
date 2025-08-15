"""Factory utilities for creating and managing chat models.

Provides:
- Caching of model instances via ``ModelCacheManager``.
- Provider management via ``ProviderRegistry`` (OpenAI, Anthropic/Claude, Google/Gemini).
- Integration with ``Settings`` for resolving API keys.
- Helpers for binding tools to models for agent execution.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool

from .providers import (
    ModelConfig,
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleGenAIProvider,
)
from echo_sdk.base.loggable import Loggable
from ..config.settings import Settings


class ModelCacheManager(Loggable):
    """Manage cached ``BaseChatModel`` instances for performance.

    The cache key combines provider, model name, and temperature to avoid
    rebuilding identical model configurations repeatedly.
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[str, BaseChatModel] = {}

    @staticmethod
    def get_cache_key(config: ModelConfig) -> str:
        """Return the cache key for a model configuration.

        Args:
            config: Model configuration.

        Returns:
            A stable key in the form ``"{provider}:{model_name}:{temperature}"``.
        """
        return f"{config.provider}:{config.model_name}:{config.temperature}"

    def get_cached_model(self, cache_key: str) -> Optional[BaseChatModel]:
        """Return a cached model if present.

        Args:
            cache_key: Key previously produced by ``get_cache_key()``.

        Returns:
            The cached ``BaseChatModel`` if available; otherwise, ``None``.
        """
        return self._cache.get(cache_key)

    def cache_model(self, cache_key: str, model: BaseChatModel) -> None:
        """Store a model in the cache.

        Args:
            cache_key: Key under which to store the model.
            model: The instantiated ``BaseChatModel`` to cache.
        """
        self._cache[cache_key] = model
        self.logger.info(f"Cached model: {cache_key}")

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self._cache.clear()
        self.logger.info("Cleared model cache")

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics.

        Returns:
            A dictionary with basic metrics, e.g., ``{"cached_models": N}``.
        """
        return {"cached_models": len(self._cache)}


class ProviderRegistry(Loggable):
    """Registry for managing LLM providers.

    Pre-registers OpenAI, Anthropic (aliased as ``"anthropic"`` and
    ``"claude"``), and Google GenAI (aliased as ``"google"`` and
    ``"gemini"``).
    """

    def __init__(self):
        super().__init__()
        self._providers: Dict[str, BaseLLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "claude": AnthropicProvider(),
            "google": GoogleGenAIProvider(),
            "gemini": GoogleGenAIProvider(),
        }
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Return provider by name.

        Args:
            provider_name: Provider key or alias.

        Returns:
            The provider instance if registered; otherwise, ``None``.
        """
        return self._providers.get(provider_name)

    def register_provider(self, name: str, provider: BaseLLMProvider) -> None:
        """Register a new provider.

        Args:
            name: Provider name under which it will be registered.
            provider: Concrete ``BaseLLMProvider`` implementation.
        """
        self._providers[name] = provider
        self.logger.info(f"Registered provider: {name}")

    def get_available_providers(self) -> List[str]:
        """Return the list of available provider names."""
        return list(self._providers.keys())

    def is_provider_available(self, provider_name: str) -> bool:
        """Return whether a provider is registered and available."""
        return provider_name in self._providers


class LLMModelFactory(Loggable):
    """Create and manage chat models with provider and cache handling.

    Responsibilities:
    - Resolve provider credentials from ``Settings`` when missing in ``ModelConfig``.
    - Cache constructed models to avoid redundant instantiation.
    - Bind tools for agent-specific execution.
    """

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.cache_manager = ModelCacheManager()
        self.provider_registry = ProviderRegistry()

    def create_base_model(self, config: ModelConfig) -> BaseChatModel:
        """Create a base model without tools.

        Args:
            config: Model configuration (provider, model, temperature, etc.).

        Returns:
            A provider-specific ``BaseChatModel`` instance.

        Raises:
            ValueError: If the provider is unknown or credentials are missing.
        """
        cache_key = self.cache_manager.get_cache_key(config)

        cached_model = self.cache_manager.get_cached_model(cache_key)
        if cached_model:
            return cached_model

        provider = self.provider_registry.get_provider(config.provider)
        if not provider:
            raise ValueError(f"Unknown provider: {config.provider}")

        self._ensure_api_key(config)

        model = provider.create_model(config)
        self.cache_manager.cache_model(cache_key, model)
        self.logger.info(f"Created model: {cache_key}")
        return model

    def _ensure_api_key(self, config: ModelConfig) -> None:
        """Ensure an API key is present in the configuration.

        If ``config.api_key`` is empty, resolve it via ``Settings``. Raises
        ``ValueError`` if a key cannot be found for the given provider.
        """
        if not config.api_key:
            config.api_key = self.settings.get_api_key_for_provider(config.provider)
            if not config.api_key:
                raise ValueError(f"No API key found for provider: {config.provider}")

    def create_agent_model(
        self, agent_name: str, tools: List[Tool], config: Optional[ModelConfig] = None
    ) -> BaseChatModel:
        """Create a model with tools bound for an agent.

        Args:
            agent_name: Logical name of the agent requesting the model.
            tools: Tools to bind to the model.
            config: Optional explicit configuration; if omitted, a default is used.

        Returns:
            A ``BaseChatModel`` with tools bound.

        Raises:
            ValueError: If the provider is not registered.
        """
        if not config:
            config = self._create_default_config()

        base_model = self.create_base_model(config)
        provider = self.provider_registry.get_provider(config.provider)

        if not provider:
            raise ValueError(f"Provider not found: {config.provider}")

        bound_model = provider.bind_tools(base_model, tools)
        self.logger.info(f"Created agent model for {agent_name} with {len(tools)} tools")
        return bound_model

    def _create_default_config(self) -> ModelConfig:
        """Create a default model configuration.

        Returns:
            A ``ModelConfig`` using the default provider from settings and a
            reasonable default model name for that provider.
        """
        model_name = "gpt-4o" if self.settings.default_llm_provider == "openai" else "claude-3-opus-20240229"

        return ModelConfig(
            provider=self.settings.default_llm_provider,
            model_name=model_name,
            api_key=self.settings.get_api_key_for_provider(self.settings.default_llm_provider),
        )

    def get_available_providers(self) -> List[str]:
        """Return the list of available provider names."""
        return self.provider_registry.get_available_providers()

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.cache_manager.clear_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics from the cache manager."""
        return self.cache_manager.get_cache_stats()

    def register_provider(self, name: str, provider: BaseLLMProvider) -> None:
        """Register a new LLM provider.

        Args:
            name: Provider name.
            provider: Concrete implementation to register.
        """
        self.provider_registry.register_provider(name, provider)
