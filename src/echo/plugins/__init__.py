"""Plugin system exports.

Exposes:
- `SDKPluginManager`, `SDKPluginBundle`: SDK-based plugin discovery and lifecycle
"""

from .sdk_manager import SDKPluginManager, SDKPluginBundle

__all__ = [
    "SDKPluginManager",
    "SDKPluginBundle",
]
