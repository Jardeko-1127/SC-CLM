"""SC-CLM V6c — Cross-Attention Mass Conditioning variant (append injection + optional MLP)."""

from .config import V6cConfig
from .model import V6cSCLM

__all__ = ["V6cConfig", "V6cSCLM"]
