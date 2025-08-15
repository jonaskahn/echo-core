"""Orchestrates multi-agent conversations for the ECHO system using LangGraph.

Builds a sequential, tool-routed graph:
  coordinator -> control_tools -> {plugin}_agent -> {plugin}_tools -> coordinator (repeat) -> finalizer -> END

Plugins register their nodes and edges via `PluginManager`. The orchestrator exposes
sync/async entry points and guards against infinite loops using a hop counter in
`AgentState`.
"""

from typing import Dict, Any

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import AgentState, MAX_HOPS, _inc_hops
from echo_sdk.base.loggable import Loggable
from ..config.settings import Settings
from ..llm.factory import LLMModelFactory
from ..plugins.sdk_manager import SDKPluginManager


class MultiAgentOrchestrator(Loggable):
    """Coordinates the sequential, tool-routed multi-agent workflow.

    Graph shape:
      coordinator -> control_tools -> {plugin}_agent -> {plugin}_tools -> coordinator ... -> finalizer -> END

    State:
      Uses `AgentState` with a `messages` history and a `hops` counter to prevent loops.

    Attributes:
      - plugin_manager: Provides coordinator tools and plugin bundles (nodes/edges).
      - llm_factory: Builds the base LLM used by the coordinator.
      - settings: Global runtime configuration.
      - coordinator_model: LLM bound with coordinator routing tools.
      - graph: Compiled LangGraph ready to invoke.
    """

    def __init__(
        self,
        plugin_manager: SDKPluginManager,
        llm_factory: LLMModelFactory,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.plugin_manager = plugin_manager
        self.llm_factory = llm_factory
        self.settings = settings

        self.coordinator_model = self._create_coordinator_model()
        self.graph = self._build_graph()

    def _create_coordinator_model(self):
        """Build the coordinator LLM and bind routing tools.

        - Chooses the model from `settings.default_llm_provider`.
        - Uses low temperature for deterministic routing.
        - Binds coordinator tools with `parallel_tool_calls=False` (one tool per step).

        Returns:
            Any: Runnable LLM with `invoke()` that may emit tool calls.
        """
        from ..llm.providers import ModelConfig

        control_tools = self.plugin_manager.get_coordinator_tools()
        config = ModelConfig(
            provider=self.settings.default_llm_provider,
            model_name=("gpt-4.1" if self.settings.default_llm_provider == "openai" else "claude-3-opus-20240229"),
            temperature=0.0,
        )

        base_model = self.llm_factory.create_base_model(config)
        return base_model.bind_tools(control_tools, parallel_tool_calls=False)

    def _build_graph(self) -> StateGraph:
        """Assemble and compile the LangGraph with dynamic plugin nodes/edges.

        Nodes:
          - coordinator: chooses the next agent or finalization via tools
          - control_tools: executes the coordinator's tool call
          - finalizer: produces the final answer

        Edges:
          - coordinator -> control_tools when last message has tool_calls, else -> END
          - control_tools -> mapping: 'end' -> 'finalizer'; '{plugin}_agent' -> '{plugin}_agent'
          - finalizer -> END

        Also registers all plugin-provided nodes and direct/conditional edges.

        Returns:
            StateGraph: Compiled graph runnable.
        """
        graph = StateGraph(AgentState)

        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("control_tools", ToolNode(tools=self.plugin_manager.get_coordinator_tools()))
        graph.add_node("finalizer", self._finalizer_node)

        for plugin_name, bundle in self.plugin_manager.plugin_bundles.items():
            nodes = bundle.get_graph_nodes()
            edges = bundle.get_graph_edges()

            for node_name, node_func in nodes.items():
                graph.add_node(node_name, node_func)

            for edge_def in edges["direct_edges"]:
                graph.add_edge(edge_def[0], edge_def[1])

            for node_name, edge_info in edges["conditional_edges"].items():
                graph.add_conditional_edges(node_name, edge_info["condition"], edge_info["mapping"])

        graph.set_entry_point("coordinator")

        graph.add_conditional_edges(
            "coordinator",
            lambda s: ("continue" if getattr(s["messages"][-1], "tool_calls", None) else "end"),
            {"continue": "control_tools", "end": END},
        )

        route_mapping = {"end": "finalizer"}
        for plugin_name in self.plugin_manager.get_available_plugins():
            route_mapping[f"{plugin_name}_agent"] = f"{plugin_name}_agent"

        graph.add_conditional_edges("control_tools", self._route_after_control_tools, route_mapping)
        graph.add_edge("finalizer", END)

        return graph.compile()

    def _coordinator_node(self, state: AgentState) -> AgentState:
        """Coordinator decision step.

        - Stops early when `MAX_HOPS` is reached.
        - Builds a `SystemMessage` that lists available `goto_{plugin}` tools and `finalize`.
        - Invokes the coordinator model to produce a single tool call or a normal reply.

        Args:
            state: Current `AgentState`.

        Returns:
            AgentState: State updates with the coordinator's `AIMessage` and incremented hops.
        """
        if state.get("hops", 0) >= MAX_HOPS:
            final_msg = SystemMessage(content=f"Max hops ({MAX_HOPS}) reached. Finalize now.")
            return {"messages": [final_msg], "hops": _inc_hops(state)}

        plugin_info = self.plugin_manager.get_plugin_routing_info()
        available_plugins = []

        for plugin_name, description in plugin_info.items():
            available_plugins.append(f"- goto_{plugin_name}: {description}")

        goto_options = " | ".join(f"goto_{name}" for name in plugin_info.keys())

        system_content = f"""You are the Coordinator for a multi-agent system. Your role is to analyze queries and route to appropriate agents.
**AVAILABLE ROUTING OPTIONS**
{chr(10).join(available_plugins)}
- finalize: Use ONLY when all necessary information has been gathered to answer the query

**IMPORTANT**
- You can only call ONE tool at a time. The system will invoke you again after each agent completes.
**CURRENT DECISION**
- Choose ONE of: {goto_options} | finalize"""

        system = SystemMessage(content=system_content)
        response = self.coordinator_model.invoke([system] + state["messages"])

        return {"messages": [response], "hops": _inc_hops(state)}

    def _finalizer_node(self, state: AgentState) -> AgentState:
        """Produce the final user-facing answer.

        Uses the coordinator model to summarize the accumulated `messages` into
        a concise, coherent response while preserving the user's language.

        Args:
            state: Current `AgentState`.

        Returns:
            AgentState: State with a final `AIMessage` appended.
        """
        messages = state.get("messages", [])
        summary_prompt = SystemMessage(
            content="""You are creating the final response for a multi-agent conversation.
CRITICAL REQUIREMENTS:
1. Be comprehensive but concise
2. Maintain the language used in the original query
3. Connect all the work done by different agents into a coherent answer"""
        )
        final_response = self.coordinator_model.invoke([summary_prompt] + messages)

        return {
            "messages": [final_response],
            "hops": state.get("hops", 0),
        }

    @staticmethod
    def _analyze_conversation_context(state: AgentState) -> str:
        """Extract a compact summary of the conversation so far.

        This helper can inform routing or summarization prompts by mentioning
        used agents, the last tool response (truncated), and the original query.

        Args:
            state: Current `AgentState`.

        Returns:
            str: Human-readable summary of prior context.
        """
        messages = state.get("messages", [])
        if len(messages) <= 1:
            return "This is the initial query - no prior agent work completed."

        context_parts = []
        agents_used = set()

        for msg in messages:
            if hasattr(msg, "name") and msg.name:
                if "_agent" in msg.name:
                    agent_name = msg.name.replace("_agent", "")
                    agents_used.add(agent_name)

        if agents_used:
            context_parts.append(f"Agents already used: {', '.join(agents_used)}")

        tool_responses = [msg for msg in messages if hasattr(msg, "type") and msg.type == "tool" and msg.content]

        if tool_responses:
            last_tool_response = tool_responses[-1].content[:200]
            context_parts.append(f"Last tool response: {last_tool_response}")

        original_query = messages[0].content if messages else "No query"
        context_parts.append(f"Original query: {original_query}")

        return " | ".join(context_parts) if context_parts else "No prior context"

    def _route_after_control_tools(self, state: AgentState) -> str:
        """Map the output of coordinator tools to the next graph node.

        Reads the last tool message content and routes as follows:
          - 'final' or unknown value -> 'end' (mapped to 'finalizer')
          - exact plugin name or 'goto_{plugin}' -> '{plugin}_agent'

        Args:
            state: Current `AgentState`.

        Returns:
            str: Next route label understood by the conditional edges.
        """
        last_message = state["messages"][-1] if state["messages"] else None

        if not last_message or not hasattr(last_message, "content"):
            self.logger.warning("No valid tool message found in routing")
            return "end"

        tool_result = last_message.content

        self.logger.info(f"Routing decision: tool_result='{tool_result}'")

        if tool_result == "final":
            return "end"
        elif tool_result in self.plugin_manager.plugin_bundles:
            target_agent = f"{tool_result}_agent"
            self.logger.info(f"Routing to agent: {target_agent}")
            return target_agent
        elif tool_result and tool_result.startswith("goto_"):
            plugin_name = tool_result[5:]
            if plugin_name in self.plugin_manager.plugin_bundles:
                target_agent = f"{plugin_name}_agent"
                self.logger.info(f"Routing to agent via goto: {target_agent}")
                return target_agent

        self.logger.warning(f"Unknown routing result: '{tool_result}', ending conversation")
        return "end"

    @staticmethod
    def _should_finalize_with_context(state: AgentState) -> bool:
        """Heuristically decide whether a finalization pass is appropriate.

        Counts agent responses and message length; not wired into the graph but
        available for experimentation.

        Args:
            state: Current `AgentState`.

        Returns:
            bool: True if we likely have enough context to finalize.
        """
        messages = state.get("messages", [])

        agent_responses = 0
        for msg in messages:
            if hasattr(msg, "content") and msg.content and hasattr(msg, "id") and "run-" in str(msg.id):
                agent_responses += 1

        return agent_responses > 0 and len(messages) > 3

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestrator synchronously.

        Expected input:
          - messages: list[BaseMessage] (Human/AI/Tool etc.)
          - hops (optional): int

        Returns:
          - dict with updated 'messages' and 'hops'.

        On error, logs and returns a friendly `AIMessage` plus incremented hops.
        """
        try:
            result = self.graph.invoke(input_data)
            return result
        except Exception as e:
            self.logger.error(f"Orchestrator invoke error: {e}")
            return {
                "messages": [AIMessage(content="I encountered an error processing your request.")],
                "hops": input_data.get("hops", 0) + 1,
            }

    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestrator asynchronously.

        Expected input:
          - messages: list[BaseMessage] (Human/AI/Tool etc.)
          - hops (optional): int

        Returns:
          - dict with updated 'messages' and 'hops'. Includes 'error' on failures.
        """
        try:
            result = await self.graph.ainvoke(input_data)
            return result
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            self.logger.error(f"Orchestrator AI invoke error: {e}\nTraceback:\n{error_trace}")
            return {
                "messages": [AIMessage(content=f"I encountered an error processing your request. Error: {str(e)}")],
                "hops": input_data.get("hops", 0) + 1,
                "error": str(e),
            }
