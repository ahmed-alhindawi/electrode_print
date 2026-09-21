"""A hexagonal cell wall lattice."""

from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import LineString
from shapely.ops import unary_union

from ..geometry import as_linestrings
from .base import JUNCTION, FillContext, InfillPattern, hex_initial_scale, register

__all__ = ["Honeycomb"]


class Honeycomb(InfillPattern):
    name = "honeycomb"
    overlap_class = JUNCTION
    description = "hex walls; three bead ends meet at each node, mild pile-up"

    def initial_scale(self, density: float, line_width: float) -> float:
        return hex_initial_scale(density, line_width)

    def min_scale(self, line_width: float) -> float:
        # The knob is a circumradius, not a spacing.  A cell's hole closes
        # when its inradius R*sqrt(3)/2 drops below half a bead, so the
        # lattice goes solid at R = w/sqrt(3), not at R = w.
        return line_width / math.sqrt(3.0)

    def generate(self, ctx: FillContext) -> list[LineString]:
        r_cell = ctx.scale
        if r_cell <= 0 or ctx.region.is_empty:
            return []
        minx, miny, maxx, maxy = ctx.region.bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        reach = math.hypot(maxx - minx, maxy - miny) / 2.0 + 2.0 * r_cell

        step_x, step_y = 1.5 * r_cell, math.sqrt(3.0) * r_cell
        n_col = int(reach / step_x) + 2
        n_row = int(reach / step_y) + 2
        # Every cell becomes a LineString that unary_union has to node against
        # its neighbours, so the cost climbs fast; refuse rather than thrash.
        if (2 * n_col + 1) * (2 * n_row + 1) > 40_000:
            return []

        corners = [
            (r_cell * math.cos(math.radians(60 * k)),
             r_cell * math.sin(math.radians(60 * k)))
            for k in range(6)
        ]
        cells = []
        for i in range(-n_col, n_col + 1):
            ox = i * step_x
            for j in range(-n_row, n_row + 1):
                oy = j * step_y + (step_y / 2.0 if i % 2 else 0.0)
                pts = [(ox + dx, oy + dy) for dx, dy in corners]
                pts.append(pts[0])
                cells.append(LineString(pts))

        # unary_union dissolves the walls shared between neighbouring cells,
        # so each wall is traced once rather than twice.
        lattice = unary_union(cells)
        lattice = affinity.rotate(lattice, ctx.angle_deg, origin=(0, 0))
        lattice = affinity.translate(lattice, cx, cy)
        return as_linestrings(ctx.region.intersection(lattice))


register(Honeycomb())
