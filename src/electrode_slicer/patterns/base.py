"""The abstract infill pattern, and the registry that finds them by name.

A pattern is asked to fill a region and is given one scalar knob to do it
with -- a line spacing, a cell size or a lattice period.  The slicer bisects
on that knob until the deposited area matches the requested density, which is
what makes "40% infill" mean the same thing for every pattern.  The only rule
an implementation must honour is that coverage falls monotonically as the knob
grows; get that wrong and calibration cannot converge.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from shapely.geometry import LineString

__all__ = [
    "FillContext",
    "InfillPattern",
    "OverlapClass",
    "register",
    "get_pattern",
    "all_patterns",
    "pattern_names",
]

# How a pattern treats its own beads.  This is the axis the electrode work
# turns on: whether the nozzle drives over material it has already laid down
# and leaves a discrete hump behind.
OverlapClass = str
NONE: OverlapClass = "none"           # no bead is laid over another
JUNCTION: OverlapClass = "junction"   # bead ends meet at lattice nodes
CROSSING: OverlapClass = "crossing"   # beads cross, piling up a hump


@dataclass(frozen=True)
class FillContext:
    """Everything a pattern needs to fill one region of one layer."""

    region: object          # shapely polygonal geometry to fill
    scale: float            # the knob: spacing, cell size or period
    angle_deg: float = 0.0
    z: float = 0.0          # height of this slice, for z-varying patterns
    line_width: float = 0.0
    resolution: float | None = None   # sampling step for field-based patterns
    stitch_ratio: float | None = None  # row spacings between cross-stitches


class InfillPattern(ABC):
    """Base class for every infill pattern."""

    #: Name used on the command line, in the SVG and in the manifest.
    name: str = ""
    #: Whether, and how, this pattern lays bead over bead.
    overlap_class: OverlapClass = NONE
    #: One line for ``--list-patterns``.
    description: str = ""
    #: Number of straight-line direction families, 0 for non-hatch patterns.
    families: int = 0
    #: True when the cross section changes with height.
    z_dependent: bool = False

    @abstractmethod
    def generate(self, ctx: FillContext) -> list[LineString]:
        """Return the fill toolpaths, already clipped to ``ctx.region``."""

    def initial_scale(self, density: float, line_width: float) -> float:
        """Analytic first guess at the knob for a target areal density.

        For k families of parallel beads of width w at spacing s the coverage
        is k*w/s, so s = k*w/density.  Patterns that are not built from
        straight-line families override this.  The guess only has to be in the
        right neighbourhood: the slicer brackets and bisects from here.
        """
        density = max(density, 1e-4)
        return max(self.families, 1) * line_width / density

    def min_scale(self, line_width: float) -> float:
        """The knob value below which this pattern is already solid.

        The slicer will not probe finer than this, which both saves work and
        keeps a calibration that cannot be satisfied from generating absurd
        amounts of geometry.  For anything whose knob is a bead spacing that
        is one bead width; patterns measuring something else override it.
        """
        return line_width

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r} {self.overlap_class}>"


_REGISTRY: dict[str, InfillPattern] = {}


def register(pattern: InfillPattern) -> InfillPattern:
    """Add a pattern to the registry, keyed by its name."""
    if not pattern.name:
        raise ValueError(f"{type(pattern).__name__} has no name")
    if pattern.name in _REGISTRY:
        raise ValueError(f"duplicate infill pattern {pattern.name!r}")
    if pattern.overlap_class not in (NONE, JUNCTION, CROSSING):
        raise ValueError(
            f"{pattern.name}: overlap_class must be one of "
            f"{NONE!r}, {JUNCTION!r}, {CROSSING!r}"
        )
    _REGISTRY[pattern.name] = pattern
    return pattern


def get_pattern(name: str) -> InfillPattern:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(pattern_names())
        raise KeyError(f"unknown infill pattern {name!r}; known: {known}") from None


def all_patterns() -> list[InfillPattern]:
    """Every registered pattern, ordered by how much it crosses itself."""
    order = {NONE: 0, JUNCTION: 1, CROSSING: 2}
    return sorted(_REGISTRY.values(), key=lambda p: (order[p.overlap_class], p.name))


def pattern_names() -> tuple[str, ...]:
    """Registered names in registration order, which is the CLI's order."""
    return tuple(_REGISTRY)


def hex_initial_scale(density: float, line_width: float) -> float:
    """Cell circumradius for a hex lattice at a target density.

    A hexagon of circumradius R has area (3*sqrt(3)/2)R^2 and, once shared
    walls are counted once, 3R of wall per cell.
    """
    return 2.0 * line_width / (math.sqrt(3.0) * max(density, 1e-4))
