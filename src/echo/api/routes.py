"""FastAPI routes and service wiring for the ECHO system.

Exposes HTTP endpoints for:

- Chatting with the multi-agent orchestrator (`POST /chat`)
- Listing plugins and fetching plugin details
- Reloading and health-checking plugins
- Reporting overall system status
- Managing in-memory chat sessions

Also provides `initialize_api()` to construct the LLM factory, plugin
manager, and orchestrator, and to wire them into a shared service container.
Documentation-only changes; no runtime logic is modified.
"""

import logging
import traceback
import uuid
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from echo_sdk.base.loggable import Loggable
from ..config.settings import Settings
from ..core.orchestrator import MultiAgentOrchestrator
from ..core.state import AgentState
from ..llm.factory import LLMModelFactory
from ..plugins.sdk_manager import SDKPluginManager


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint.

    Attributes:
        message: User's input text.
        session_id: Optional conversation session identifier. If omitted, a
            new session is created.
        metadata: Optional request-scoped metadata for downstream use.
    """

    message: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response payload for the chat endpoint.

    Attributes:
        response: Final assistant message text.
        session_id: The session identifier used for this turn.
        plugin_used: Backward-compat field for the last plugin involved.
        agents_used: Agents involved in this turn (for multi-agent flows).
        hops: Number of state transitions in the orchestrator.
        metadata: Additional diagnostics (message_count, routing history, etc.).
    """

    response: str
    session_id: str
    plugin_used: Optional[str] = None
    agents_used: Optional[List[str]] = None
    hops: int
    metadata: Optional[Dict[str, Any]] = None


class PluginInfo(BaseModel):
    """Public information about a plugin bundle.

    Attributes:
        name: Plugin name.
        version: Semantic version string.
        description: Human-readable description.
        capabilities: High-level list of supported capabilities.
        status: Health status (e.g., "healthy" or "failed").
    """

    name: str
    version: str
    description: str
    capabilities: List[str]
    status: str


class SystemStatus(BaseModel):
    """Aggregated system status snapshot.

    Attributes:
        status: Overall API status string.
        available_plugins: Names of successfully loaded plugins.
        healthy_plugins: Subset of available plugins passing health checks.
        failed_plugins: Plugins that failed to load or failed health checks.
        total_sessions: In-memory active session count.
    """

    status: str
    available_plugins: List[str]
    healthy_plugins: List[str]
    failed_plugins: List[str]
    total_sessions: int


class SessionManager:
    """In-memory chat session tracking.

    Maintains per-session message histories used as input to the orchestrator.
    """

    def __init__(self):
        self._sessions: Dict[str, List[Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, List[Any]]:
        """Return an existing session or create a new one.

        Args:
            session_id: Optional pre-existing session identifier.

        Returns:
            tuple[str, List[Any]]: The session id and its message history.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self.logger.info(f"Created new session: {session_id}")

        return session_id, self._sessions[session_id]

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session's history if it exists.

        Args:
            session_id: Identifier of the session to remove.

        Returns:
            bool: True if the session existed and was cleared; False otherwise.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self.logger.info(f"Cleared session: {session_id}")
            return True
        return False

    def get_total_sessions(self) -> int:
        """Return the number of active in-memory sessions."""
        return len(self._sessions)

    def add_message_to_session(self, session_id: str, message: Any) -> None:
        """Append a message to a session's history if the session exists.

        Args:
            session_id: Target session identifier.
            message: Message object to record (e.g., LangChain message).
        """
        if session_id in self._sessions:
            self._sessions[session_id].append(message)


class APIServiceContainer(Loggable):
    """Container for API services and dependencies.

    Holds initialized instances of the orchestrator, plugin manager, and an
    in-memory `SessionManager`. Accessors raise HTTP 503 if not initialized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.orchestrator: Optional[MultiAgentOrchestrator] = None
        self.plugin_manager: Optional[SDKPluginManager] = None
        self.session_manager = SessionManager()

    def initialize(self, orchestrator: MultiAgentOrchestrator, plugin_manager: SDKPluginManager) -> None:
        """Initialize all API services.

        Args:
            orchestrator: The multi-agent orchestrator instance.
            plugin_manager: The plugin manager instance.
        """
        self.orchestrator = orchestrator
        self.plugin_manager = plugin_manager
        self.logger.info("API services initialized")

    def get_orchestrator(self) -> MultiAgentOrchestrator:
        """Return the orchestrator instance or raise HTTP 503 if unavailable."""
        if not self.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        return self.orchestrator

    def get_plugin_manager(self) -> SDKPluginManager:
        """Return the plugin manager instance or raise HTTP 503 if unavailable."""
        if not self.plugin_manager:
            raise HTTPException(status_code=503, detail="Plugin manager not initialized")
        return self.plugin_manager


# Global service container
service_container = APIServiceContainer()
router = APIRouter(prefix="/api/v1")


def get_orchestrator() -> MultiAgentOrchestrator:
    """FastAPI dependency providing the initialized orchestrator instance."""
    return service_container.get_orchestrator()


def get_plugin_manager() -> SDKPluginManager:
    """FastAPI dependency providing the initialized plugin manager instance."""
    return service_container.get_plugin_manager()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, orch: MultiAgentOrchestrator = Depends(get_orchestrator)):
    """Chat with the ECHO orchestrator.

    Accepts a message and optional `session_id`, appends it to the session
    history, runs the orchestrator, and returns the last AI message content
    plus diagnostics.

    Args:
        request: Chat payload containing user input and optional metadata.
        orch: Injected orchestrator instance.

    Returns:
        ChatResponse: Final assistant message and conversation metadata.
    """
    try:
        session_id, session_messages = service_container.session_manager.get_or_create_session(request.session_id)

        user_msg = HumanMessage(content=request.message)
        service_container.session_manager.add_message_to_session(session_id, user_msg)

        state: AgentState = {
            "messages": session_messages,
            "hops": 0,
            "session_id": session_id,
            "plugin_context": {},
        }

        result = await orch.ainvoke(state)

        service_container.logger.info(f"Total messages in result: {len(result['messages'])}")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            content = getattr(msg, "content", str(msg))[:100]  # First 100 chars
            service_container.logger.info(f"Message {i}: Type={msg_type}, Content={content}")

        response_text = "No response generated"
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and not msg.content.startswith("Tool"):
                if isinstance(msg, AIMessage) and msg.content:
                    response_text = msg.content
                    service_container.logger.info(f"Found response: {response_text}")
                    break

        for msg in result["messages"]:
            service_container.session_manager.add_message_to_session(session_id, msg)

        plugin_context = result.get("plugin_context", {})
        plugin_used = plugin_context.get("last_plugin", None)

        agents_used = None
        multi_agent_analysis = plugin_context.get("multi_agent_analysis", {})
        if multi_agent_analysis.get("is_multi_agent"):
            agents_used = multi_agent_analysis.get("required_agents", [])
        elif plugin_used:
            agents_used = [plugin_used]

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            plugin_used=plugin_used,
            agents_used=agents_used,
            hops=result.get("hops", 0),
            metadata={
                "message_count": len(result["messages"]),
                "routing_history": plugin_context.get("routing_history", []),
                "multi_agent": multi_agent_analysis.get("is_multi_agent", False),
            },
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        service_container.logger.error(f"Chat error: {e}\nTraceback:\n{error_trace}")
        return ChatResponse(
            response=f"I encountered an error processing your request. Error: {str(e)}",
            session_id=session_id,
            plugin_used=None,
            agents_used=None,
            hops=1,
            metadata={
                "message_count": 1,
                "routing_history": [],
                "multi_agent": False,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


@router.get("/plugins", response_model=List[PluginInfo])
async def list_plugins(pm: SDKPluginManager = Depends(get_plugin_manager)):
    """List available plugin bundles and their status."""
    plugins = []

    for plugin_name in pm.get_available_plugins():
        bundle = pm.get_plugin_bundle(plugin_name)
        if bundle:
            metadata = bundle.metadata
            status = "healthy" if plugin_name in pm.healthy_plugins else "failed"

            plugins.append(
                PluginInfo(
                    name=metadata.name,
                    version=metadata.version,
                    description=metadata.description,
                    capabilities=metadata.capabilities,
                    status=status,
                )
            )

    return plugins


@router.get("/plugins/{plugin_name}", response_model=PluginInfo)
async def get_plugin(plugin_name: str, pm: SDKPluginManager = Depends(get_plugin_manager)):
    """Return metadata and status for a specific plugin bundle.

    Raises:
        HTTPException: 404 if the plugin is not found.
    """
    bundle = pm.get_plugin_bundle(plugin_name)

    if not bundle:
        raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} not found")

    metadata = bundle.metadata
    status = "healthy" if plugin_name in pm.healthy_plugins else "failed"

    return PluginInfo(
        name=metadata.name,
        version=metadata.version,
        description=metadata.description,
        capabilities=metadata.capabilities,
        status=status,
    )


@router.post("/plugins/reload")
async def reload_plugins(pm: SDKPluginManager = Depends(get_plugin_manager)):
    """Reload all plugins and refresh health status.

    Returns:
        dict: Summary of loaded/healthy/failed plugin names.
    """
    try:
        pm.reload_plugins()

        return {
            "status": "success",
            "loaded": list(pm.get_available_plugins()),
            "healthy": list(pm.healthy_plugins),
            "failed": list(pm.failed_plugins),
        }
    except Exception as e:
        service_container.logger.error(f"Plugin reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=SystemStatus)
async def system_status(pm: SDKPluginManager = Depends(get_plugin_manager)):
    """Return a snapshot of overall system status."""
    return SystemStatus(
        status="operational",
        available_plugins=pm.get_available_plugins(),
        healthy_plugins=list(pm.healthy_plugins),
        failed_plugins=list(pm.failed_plugins),
        total_sessions=service_container.session_manager.get_total_sessions(),
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session's history.

    Raises:
        HTTPException: 404 if the session id is unknown.
    """
    if service_container.session_manager.clear_session(session_id):
        return {"status": "cleared", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@router.get("/health")
async def health_check():
    """Simple liveness probe for the API service."""
    return {"status": "healthy"}


def initialize_api(settings: Settings) -> None:
    """Initialize API components and wire services into the container.

    Constructs the `LLMModelFactory`, `PluginManager` (loading and health
    checking plugins), and the `MultiAgentOrchestrator`. Registers them in the
    global `service_container` for dependency injection in routes.

    Args:
        settings: Application settings with plugin and provider configuration.
    """
    llm_factory = LLMModelFactory(settings)

    plugin_manager = SDKPluginManager(settings.plugins_dir, llm_factory)
    plugin_manager.discover_and_load_plugins()
    plugin_manager.perform_health_checks()

    orchestrator = MultiAgentOrchestrator(plugin_manager, llm_factory, settings)
    service_container.initialize(orchestrator, plugin_manager)
    service_container.logger.info("API initialized successfully")
    service_container.logger.info(f"Loaded plugins: {plugin_manager.get_available_plugins()}")
