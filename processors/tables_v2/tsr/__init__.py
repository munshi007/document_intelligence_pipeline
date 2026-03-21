"""TSR Engine implementations."""

from .base import TSREngine, CellPx

# Lazy import TATR to avoid torch dependency at import time
def get_tatr_engine():
    from .tatr import TATREngine
    return TATREngine

__all__ = ["TSREngine", "CellPx", "get_tatr_engine"]
