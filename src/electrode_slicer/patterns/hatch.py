"""Patterns built from families of parallel straight lines.

These span most of the crossing axis on their own: one family never crosses,
two give a clean X at every cell, and three or four pile several beads onto
each node.
"""

from __future__ import annotations

from shapely.geometry import LineString

from ..geometry import hatch
from .base import CROSSING, NONE, FillContext, InfillPattern, register

__all__ = ["HatchPattern", "Lines", "Grid", "Triangles", "Stars", "Solid", "NoInfill"]


class HatchPattern(InfillPattern):
    """A fill made of `angles` line families at one shared spacing."""

    angles: tuple[float, ...] = (0.0,)

    @property
    def families(self) -> int:  # type: ignore[override]
        return len(self.angles)

    def generate(self, ctx: FillContext) -> list[LineString]:
        lines: list[LineString] = []
        for offset in self.angles:
            lines.extend(hatch(ctx.region, ctx.scale, ctx.angle_deg + offset))
        return lines


class Lines(HatchPattern):
    name = "lines"
    overlap_class = NONE
    description = "parallel passes; the flat, no-hump control"
    angles = (0.0,)


class Grid(HatchPattern):
    name = "grid"
    overlap_class = CROSSING
    description = "two families 90 deg apart; one clean X hump per cell"
    angles = (0.0, 90.0)


class Triangles(HatchPattern):
    name = "triangles"
    overlap_class = CROSSING
    description = "three families 60 deg apart; 3-way humps, denser"
    angles = (0.0, 60.0, 120.0)


class Stars(HatchPattern):
    name = "stars"
    overlap_class = CROSSING
    description = "four families 45 deg apart; the densest hump lattice"
    angles = (0.0, 45.0, 90.0, 135.0)


class Solid(HatchPattern):
    """Fully dense parallel passes, used as the 100% reference pad."""

    name = "solid"
    overlap_class = NONE
    description = "fully dense parallel passes; a flat reference pad"
    angles = (0.0,)

    def initial_scale(self, density: float, line_width: float) -> float:
        # Solid means beads laid edge to edge whatever density was asked for.
        return line_width


class NoInfill(InfillPattern):
    name = "none"
    overlap_class = NONE
    description = "perimeters only"

    def generate(self, ctx: FillContext) -> list[LineString]:
        return []


for _pattern in (Lines(), Grid(), Triangles(), Stars(), Solid(), NoInfill()):
    register(_pattern)
