"""Infill patterns.

Importing this package registers every built-in pattern.  A new pattern is a
subclass of :class:`~electrode_slicer.patterns.base.InfillPattern` plus a call
to :func:`register`; nothing else in the codebase needs to change.
"""

from __future__ import annotations

from .base import (
    CROSSING,
    JUNCTION,
    NONE,
    FillContext,
    InfillPattern,
    OverlapClass,
    all_patterns,
    get_pattern,
    pattern_names,
    register,
)

# Import order fixes the order patterns appear in on the command line, so it
# is deliberate rather than alphabetical and isort is told to leave it alone.
# isort: off
from .hatch import Grid, Lines, NoInfill, Solid, Stars, Triangles   # noqa: F401
from .curves import Hilbert, Serpentine                             # noqa: F401
from .honeycomb import Honeycomb                                    # noqa: F401
from .tpms import Gyroid, TpmsPattern                               # noqa: F401
# isort: on

__all__ = [
    "CROSSING",
    "JUNCTION",
    "NONE",
    "FillContext",
    "InfillPattern",
    "OverlapClass",
    "all_patterns",
    "get_pattern",
    "pattern_names",
    "register",
]
