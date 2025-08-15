# Echo Core

Echo Core is the main orchestration and coordination engine for the ECHO multi-agent system.

## Architecture

Echo Core provides:

- **Multi-Agent Orchestration**: Coordinates multiple AI agents using LangGraph
- **Plugin System**: SDK-based plugin discovery and lifecycle management
- **LLM Factory**: Provider-agnostic model creation and caching
- **REST API**: FastAPI-based HTTP interface
- **State Management**: Conversation state tracking and session management

## Configuration

Configuration is managed through environment variables with the `ECHO_` prefix:

```bash
# LLM Provider
ECHO_DEFAULT_LLM_PROVIDER=openai
ECHO_OPENAI_API_KEY=your_key_here

# API Server
ECHO_API_HOST=0.0.0.0
ECHO_API_PORT=8000

# Plugins
ECHO_PLUGINS_DIR=./plugins
```

See `src/echo_core/config/settings.py` for all available settings.

## Local Development

To set up the project for local development with the SDK and plugins:

```bash
# Install dependencies with local SDK and plugins
poetry install --with local --without production
```

This command:

- Installs the main project dependencies
- Includes the local `echo-sdk` and `echo-plugins` packages for development
- Excludes production dependencies that would fetch from remote repositories

## Project Structure

```
echo_core/
├── src/echo_core/
│   ├── api/              # FastAPI routes and service wiring
│   ├── config/           # Application configuration
│   ├── core/             # Orchestration and state management
│   ├── llm/              # LLM factory and providers
│   ├── plugins/          # SDK-based plugin management
│   └── main.py           # Application entrypoint
├── bin/                  # Binary scripts
├── tests/                # Test suite
└── pyproject.toml        # Project configuration and dependencies
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
