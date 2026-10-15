"""Cross sections of triply-periodic minimal surfaces.

Sampling the implicit field and contouring it keeps every surface on one code
path, and gives the true z-varying cross section rather than a 2D lookalike.
Adding another surface is a new subclass with a new ``field`` -- Schwarz-P is
two lines of numpy -- but only the gyroid is registered, because it is the
only one whose cross section stays connected across the useful density range.
"""

from __future__ import annotations

import math

import numpy as np
from contourpy import LineType, contour_generator
from shapely import affinity
from shapely.geometry import LineString, MultiLineString

from ..geometry import as_linestrings, robust_intersection
from .base import NONE, FillContext, InfillPattern, register

__all__ = ["TpmsPattern", "Gyroid"]


class TpmsPattern(InfillPattern):
    """Zero contour of a periodic scalar field, taken at the slice height."""

    overlap_class = NONE
    z_dependent = True
    #: Samples across one period; finer costs time, coarser rounds the curve.
    samples_per_period = 24

    def field(self, u, v, w):  # pragma: no cover - abstract in spirit
        raise NotImplementedError

    def initial_scale(self, density: float, line_width: float) -> float:
        # Rough on purpose: the slicer's bisection cleans it up.
        return 2.4 * line_width / max(density, 1e-4)

    def generate(self, ctx: FillContext) -> list[LineString]:
        period = ctx.scale
        if period <= 0 or ctx.region.is_empty:
            return []
        # Work in the pattern's own frame so the field stays axis-aligned.
        local = affinity.rotate(ctx.region, -ctx.angle_deg, origin=(0, 0))
        minx, miny, maxx, maxy = local.bounds
        # Sampling finer than a fraction of the bead buys nothing -- the
        # machine cannot render detail below its own line width -- and the
        # field arrays grow as the square of the step, so floor it against the
        # bead rather than letting a small calibration probe allocate GBs.
        floor = max(ctx.line_width / 8.0, 1e-3) if ctx.line_width else 1e-3
        res = ctx.resolution or max(period / self.samples_per_period, floor)
        pad = period * 0.1 + res

        xs = np.arange(minx - pad, maxx + pad + res, res)
        ys = np.arange(miny - pad, maxy + pad + res, res)
        if xs.size < 2 or ys.size < 2 or xs.size * ys.size > 4_000_000:
            return []

        k = 2.0 * math.pi / period
        u, v = np.meshgrid(k * xs, k * ys)
        field = self.field(u, v, k * ctx.z)

        gen = contour_generator(x=xs, y=ys, z=field, line_type=LineType.Separate)
        curves = [LineString(pts) for pts in gen.lines(0.0) if len(pts) >= 2]
        if not curves:
            return []
        back = affinity.rotate(MultiLineString(curves), ctx.angle_deg, origin=(0, 0))
        return as_linestrings(robust_intersection(ctx.region, back))


class Gyroid(TpmsPattern):
    name = "gyroid"
    description = "gyroid cross section; smooth non-crossing waves"

    def field(self, u, v, w):
        return np.sin(u) * np.cos(v) + np.sin(v) * math.cos(w) + math.sin(w) * np.cos(u)


register(Gyroid())
