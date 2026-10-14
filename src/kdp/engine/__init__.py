from __future__ import annotations

from ..config import Settings
from .base import Engine, PageInput, RawBlock, RawPage
from .replay import ReplayEngine


def create_engine(settings: Settings) -> Engine:
    if settings.parse.engine == "paddle_vl":
        from .paddle_vl import PaddleVLEngine

        return PaddleVLEngine(settings)
    raise ValueError(settings.parse.engine)


__all__ = ["Engine", "PageInput", "RawBlock", "RawPage", "ReplayEngine", "create_engine"]
