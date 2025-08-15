"""Provider interfaces and implementations for chat LLMs.

Includes:
- ``BaseLLMProvider`` abstract interface.
- Concrete providers for OpenAI, Anthropic/Claude, and Google/Gemini.
- ``ModelConfig`` dataclass for model configuration and optional params.

Optional third-party packages are imported with graceful fallbacks; if a
provider package is missing, a clear ``ImportError`` is raised when used.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool

try:
    from langchain_openai import ChatOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False
    ChatGoogleGenerativeAI = None


@dataclass
class ModelConfig:
    """Configuration for constructing and tuning chat models.

    Fields:
    - provider: Provider key (e.g., "openai", "anthropic", "google").
    - model_name: Provider-specific model identifier.
    - temperature: Sampling temperature (0.0–1.0 typical range).
    - max_tokens: Maximum tokens to generate.
    - api_key: API key for provider (may be resolved elsewhere).
    - additional_params: Extra provider-specific parameters.
    """

    provider: str
    model_name: str
    temperature: float = 0.8
    max_tokens: int = 1024
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize mutable defaults after initialization."""
        if self.additional_params is None:
            self.additional_params = {}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def create_model(self, config: ModelConfig) -> BaseChatModel:
        """Create a provider-specific chat model.

        Args:
            config: Model configuration parameters.

        Returns:
            A concrete ``BaseChatModel`` instance.
        """
        pass

    @abstractmethod
    def bind_tools(self, model: BaseChatModel, tools: List[Tool]) -> BaseChatModel:
        """Bind tools to the given model.

        Args:
            model: Base chat model to augment with tools.
            tools: List of ``Tool`` objects to bind.

        Returns:
            A ``BaseChatModel`` with tools bound.
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def create_model(self, config: ModelConfig) -> BaseChatModel:
        """Create an OpenAI chat model.

        Raises:
            ImportError: If ``langchain-openai`` is not installed.
        """
        if not HAS_OPENAI:
            raise ImportError("langchain-openai is not installed. Install with: pip install langchain-openai")

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            **config.additional_params,
        )

    def bind_tools(self, model: BaseChatModel, tools: List[Tool]) -> BaseChatModel:
        """Bind tools to an OpenAI chat model."""
        return model.bind_tools(tools)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) provider implementation."""

    def create_model(self, config: ModelConfig) -> BaseChatModel:
        """Create an Anthropic (Claude) chat model.

        Raises:
            ImportError: If ``langchain-anthropic`` is not installed.
        """
        if not HAS_ANTHROPIC:
            raise ImportError("langchain-anthropic is not installed. Install with: pip install langchain-anthropic")

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            **config.additional_params,
        )

    def bind_tools(self, model: BaseChatModel, tools: List[Tool]) -> BaseChatModel:
        """Bind tools to an Anthropic chat model."""
        return model.bind_tools(tools)


class GoogleGenAIProvider(BaseLLMProvider):
    """Google Generative AI (Gemini) provider implementation."""

    def create_model(self, config: ModelConfig) -> BaseChatModel:
        """Create a Google Generative AI (Gemini) chat model.

        Raises:
            ImportError: If ``langchain-google-genai`` is not installed.
        """
        if not HAS_GOOGLE:
            raise ImportError(
                "langchain-google-genai is not installed. Install with: pip install langchain-google-genai"
            )

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            google_api_key=config.api_key,
            **config.additional_params,
        )

    def bind_tools(self, model: BaseChatModel, tools: List[Tool]) -> BaseChatModel:
        """Bind tools to a Google Generative AI chat model."""
        return model.bind_tools(tools)
