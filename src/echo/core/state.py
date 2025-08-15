"""State management for ECHO Multi-Agent System."""

import logging
from typing import TYPE_CHECKING, Sequence, Optional, Dict, Any, List, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from typing import TypedDict
else:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        from typing import TypedDict

logger = logging.getLogger(__name__)

MAX_HOPS: int = 10


class AgentState(TypedDict, total=False):
    """TypedDict representing the conversation state tracked by the orchestrator.

    Fields:
    - messages: Sequence of LangChain messages; aggregated via
      ``langgraph.graph.message.add_messages``.
    - current_agent: Identifier of the active agent/plugin for the current hop.
    - hops: Hop counter used to guard against infinite loops.
    - last_tool_call: Name of the last tool call issued by the assistant, if any.
    - session_id: Optional session identifier used to group requests.
    - metadata: Arbitrary metadata associated with the session or request.
    - agents_used: Ordered list of agents that have contributed so far.
    - parallel_results: Optional container for intermediate results produced
      outside the main turn.
    - routing_decision: Label of the most recent routing decision
      (e.g., next agent/tool).
    - plugin_context: Ephemeral plugin-specific context preserved across hops.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: Optional[str]
    hops: int
    last_tool_call: Optional[str]
    session_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

    agents_used: List[str]
    parallel_results: Optional[Dict[str, Any]]
    routing_decision: Optional[str]

    plugin_context: Optional[Dict[str, Any]]


def _inc_hops(state: AgentState) -> int:
    """Increment the hop counter.

    Args:
        state: Current ``AgentState`` snapshot.

    Returns:
        The incremented hop count.
    """
    return state.get("hops", 0) + 1


def _last_assistant_tool_call_name(state: AgentState) -> Optional[str]:
    """Return the name of the last tool call issued by the assistant.

    Scans the state's messages in reverse chronological order and returns the
    tool name from the most recent assistant message that contains
    ``tool_calls``.

    Args:
        state: Current ``AgentState`` snapshot.

    Returns:
        The tool name if found; otherwise, ``None``.
    """
    messages = state.get("messages", [])
    for message in reversed(messages):
        if hasattr(message, "tool_calls") and message.tool_calls:
            return message.tool_calls[-1]["name"]
    return None
