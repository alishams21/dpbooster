from importlib.metadata import version

try:
    __version__ = version("dpbooster")
except Exception:
    __version__ = "unknown"

from .rag_provider import registry

__all__ = ["registry"]