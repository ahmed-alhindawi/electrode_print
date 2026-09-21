"""Local web front end.  Run with ``python -m electrode_slicer.web``."""

from __future__ import annotations

from .server import SliceService, main, serve

__all__ = ["SliceService", "main", "serve"]
