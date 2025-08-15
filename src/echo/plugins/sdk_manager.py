"""SDK-based plugin manager for ECHO using the echo-plugin-sdk.

This replaces the direct plugin loading approach with SDK-based discovery,
enabling true decoupling between core and plugins.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from langchain_core.tools import Tool, tool
from langgraph.prebuilt import ToolNode

# Import from SDK instead of local modules
try:
    from echo_sdk import discover_plugins, PluginContract, AgentState
    from echo_sdk.utils import validate_plugin_structure

    SDK_AVAILABLE = True
except ImportError:
    # Fallback for when SDK is not available
    discover_plugins = None
    PluginContract = None
    AgentState = None
    validate_plugin_structure = None
    SDK_AVAILABLE = False

from echo_sdk.base.loggable import Loggable
from ..llm.factory import LLMModelFactory


class SDKPluginBundle(Loggable):
    """SDK-based plugin bundle using PluginContract interface.

    This replaces the old PluginBundle with SDK-based contracts,
    maintaining the same interface for the orchestrator.
    """

    def __init__(
        self,
        contract: "PluginContract",
        agent,
        bound_model,
        tools: List[Tool],
    ):
        super().__init__()
        self.contract = contract
        self.metadata = contract.get_metadata()
        self.agent = agent
        self.bound_model = bound_model
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.agent_node = agent.create_agent_node()

    def get_graph_nodes(self) -> Dict[str, Any]:
        """Return LangGraph node callables for this plugin bundle."""
        agent_name = self.metadata.name
        return {
            f"{agent_name}_agent": self.agent_node,
            f"{agent_name}_tools": self.tool_node,
        }

    def get_graph_edges(self) -> Dict[str, Any]:
        """Return LangGraph edge definitions for this plugin bundle."""
        agent_name = self.metadata.name
        return {
            "conditional_edges": {
                f"{agent_name}_agent": {
                    "condition": self.agent.should_continue,
                    "mapping": {
                        "continue": f"{agent_name}_tools",
                        "back": "coordinator",
                    },
                }
            },
            "direct_edges": [(f"{agent_name}_tools", f"{agent_name}_agent")],
        }


class SDKPluginManager(Loggable):
    """SDK-based plugin manager using echo-plugin-sdk for discovery.

    This replaces the old PluginManager with SDK-based plugin discovery,
    enabling true decoupling between core and plugins.
    """

    def __init__(self, plugins_dir: str, llm_factory: LLMModelFactory):
        super().__init__()
        self.plugins_dir = Path(plugins_dir)
        self.llm_factory = llm_factory

        if not SDK_AVAILABLE:
            raise RuntimeError("echo-plugin-sdk is not available. " "Please install it: pip install echo-plugin-sdk")

        self.plugin_bundles: Dict[str, SDKPluginBundle] = {}
        self.plugin_contracts: Dict[str, "PluginContract"] = {}

        self.healthy_plugins: Set[str] = set()
        self.failed_plugins: Set[str] = set()

    def discover_plugin_directories(self) -> List[Path]:
        """Discover plugin directories containing plugin packages."""
        plugins = []
        if not self.plugins_dir.exists():
            self.logger.warning(f"Plugins directory not found: {self.plugins_dir}")
            return plugins

        for plugin_path in self.plugins_dir.iterdir():
            if plugin_path.is_dir() and (plugin_path / "__init__.py").exists():
                plugins.append(plugin_path)

        self.logger.info(f"Discovered {len(plugins)} plugin directories")
        return plugins

    def load_plugin_packages(self) -> None:
        """Load plugin packages to trigger SDK registration."""
        plugin_paths = self.discover_plugin_directories()

        for plugin_path in plugin_paths:
            try:
                self._load_plugin_package(plugin_path)
            except Exception as e:
                self.logger.error(f"Failed to load plugin package {plugin_path}: {e}")
                self.failed_plugins.add(plugin_path.name)

    def _load_plugin_package(self, plugin_path: Path) -> None:
        """Load a plugin package to trigger auto-registration."""
        plugin_name = plugin_path.name

        # Add plugin directory to path temporarily
        sys.path.insert(0, str(plugin_path.parent))
        try:
            # Import the plugin package - this should trigger auto-registration
            importlib.import_module(plugin_name)
            self.logger.info(f"Loaded plugin package: {plugin_name}")
        except Exception as e:
            self.logger.error(f"Error loading plugin package {plugin_name}: {e}")
            raise
        finally:
            # Clean up sys.path
            if str(plugin_path.parent) in sys.path:
                sys.path.remove(str(plugin_path.parent))

    def discover_and_load_plugins(self) -> None:
        """Discover SDK-registered plugins and create bundles."""
        # First load plugin packages to trigger registration
        self.load_plugin_packages()

        # Then discover plugins from SDK registry
        contracts = discover_plugins()

        self.logger.info(f"Discovered {len(contracts)} SDK-registered plugins")

        for contract in contracts:
            try:
                self._create_plugin_bundle(contract)
            except Exception as e:
                self.logger.error(f"Failed to create bundle for {contract.name}: {e}")
                self.failed_plugins.add(contract.name)

    def _create_plugin_bundle(self, contract: "PluginContract") -> bool:
        """Create a plugin bundle from an SDK contract."""
        try:
            metadata = contract.get_metadata()
            plugin_name = metadata.name

            self.logger.info(f"Creating plugin bundle for: {plugin_name}")

            # Validate plugin structure
            if validate_plugin_structure:
                errors = validate_plugin_structure(contract.plugin_class)
                if errors:
                    self.logger.error(f"Plugin validation failed for {plugin_name}: {errors}")
                    return False

            # Validate dependencies
            dep_errors = contract.validate_dependencies()
            if dep_errors:
                self.logger.error(f"Plugin dependencies failed for {plugin_name}: {dep_errors}")
                return False

            # Create agent instance
            agent = contract.create_agent()

            # Create model configuration
            model_config = self._create_model_config(metadata)
            base_model = self.llm_factory.create_base_model(model_config)

            # Get tools and bind model
            tools = agent.get_tools()
            bound_model = agent.bind_model(base_model)

            # Initialize agent
            agent.initialize()

            # Create bundle
            bundle = SDKPluginBundle(contract=contract, agent=agent, bound_model=bound_model, tools=tools)

            self.plugin_bundles[plugin_name] = bundle
            self.plugin_contracts[plugin_name] = contract
            self.healthy_plugins.add(plugin_name)

            self.logger.info(f"Successfully created plugin bundle: {plugin_name} v{metadata.version}")
            self.logger.info(f"  - Agent: {type(agent).__name__}")
            self.logger.info(f"  - Tools: {len(tools)} tools")
            self.logger.info(f"  - Capabilities: {metadata.capabilities}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to create plugin bundle for {contract.name}: {e}")
            return False

    def _create_model_config(self, metadata):
        """Create a model configuration from plugin metadata."""
        from ..llm.providers import ModelConfig

        # Get model config from metadata
        model_config = metadata.get_model_config()

        return ModelConfig(
            provider=model_config.provider,
            model_name=model_config.model_name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            additional_params=model_config.additional_params,
        )

    def get_plugin_bundle(self, name: str) -> Optional[SDKPluginBundle]:
        """Get a plugin bundle by name."""
        return self.plugin_bundles.get(name)

    def get_available_plugins(self) -> List[str]:
        """List names of successfully loaded plugin bundles."""
        return list(self.plugin_bundles.keys())

    def get_all_plugin_tools(self) -> Dict[str, List[Tool]]:
        """Return all registered tools organized by plugin name."""
        result = {}
        for name, bundle in self.plugin_bundles.items():
            result[name] = bundle.tools
        return result

    def get_plugin_routing_info(self) -> Dict[str, str]:
        """Return short routing descriptions for coordinator prompts."""
        result = {}
        for name, bundle in self.plugin_bundles.items():
            metadata = bundle.metadata
            result[name] = f"{metadata.description}. Capabilities: {', '.join(metadata.capabilities)}"
        return result

    def perform_health_checks(self) -> Dict[str, bool]:
        """Perform health checks on all plugin bundles."""
        results = {}

        for plugin_name, contract in self.plugin_contracts.items():
            try:
                # Use SDK contract health check
                health_status = contract.health_check()
                is_healthy = health_status.get("healthy", False)

                results[plugin_name] = is_healthy

                if is_healthy:
                    self.healthy_plugins.add(plugin_name)
                    self.failed_plugins.discard(plugin_name)
                else:
                    self.failed_plugins.add(plugin_name)
                    self.healthy_plugins.discard(plugin_name)

                self.logger.info(f"Health check for {plugin_name}: {health_status}")

            except Exception as e:
                self.logger.error(f"Health check failed for {plugin_name}: {e}")
                results[plugin_name] = False
                self.failed_plugins.add(plugin_name)

        return results

    def get_coordinator_tools(self) -> List[Tool]:
        """Create routing tools used by the coordinator for control flow."""
        control_tools: List[Tool] = []

        # Create goto tools for each plugin
        for plugin_name in self.get_available_plugins():
            metadata = self.plugin_contracts[plugin_name].get_metadata()

            # Create closure to capture plugin_name
            def make_goto_tool(name: str, desc: str) -> Tool:
                func_name = f"goto_{name}"
                func_code = f'''
def {func_name}():
    """Route to {name} plugin for {desc}"""
    return "{name}"
'''
                local_vars = {}
                exec(func_code, {}, local_vars)
                return tool(local_vars[func_name])

            control_tools.append(make_goto_tool(plugin_name, metadata.description))

        # Add finalize tool
        @tool
        def finalize():
            """Signal that the task is complete and provide final response."""
            return "final"

        control_tools.append(finalize)
        return control_tools

    def reload_plugins(self) -> None:
        """Reload all plugins by clearing and rediscovering."""
        self.logger.info("Reloading all plugins...")

        # Clear current state
        self.plugin_bundles.clear()
        self.plugin_contracts.clear()
        self.healthy_plugins.clear()
        self.failed_plugins.clear()

        # Rediscover and load
        self.discover_and_load_plugins()
        self.perform_health_checks()

        self.logger.info(f"Plugin reload complete. Loaded {len(self.plugin_bundles)} plugins")
