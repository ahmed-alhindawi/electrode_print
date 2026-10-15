"""Single-stroke fills: one continuous path rather than many separate ones.

Concentric rings are crossing-free, which makes them a tempting control for
the hump experiment, but nested loops never touch each other, so most of the
ink ends up electrically isolated.  These two fills solve that by construction:
they are drawn as one continuous stroke, so every part of the fill is joined
to every other part, and neither ever crosses itself.

They are also gentler on a flexible substrate than a straight hatch -- the
turns are rounded rather than square, so there is no sharp corner to
concentrate strain when the electrode is bent against skin.
"""

from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import LineString, MultiLineString

from ..geometry import as_linestrings, connect_along_boundary, hatch
from .base import CROSSING, NONE, FillContext, InfillPattern, register

__all__ = ["Serpentine", "Hilbert", "Stitched"]


def _rounded(points: list[tuple[float, float]], radius: float,
             segments: int = 4) -> list[tuple[float, float]]:
    """Round off every interior corner of a polyline.

    A square turn is a strain riser on a flexible part, and the nozzle has to
    stop dead to make it.  Cutting the corner costs a few vertices and gives
    a path that bends and prints smoothly.
    """
    if radius <= 0 or len(points) < 3:
        return points
    out = [points[0]]
    for i in range(1, len(points) - 1):
        prev, here, nxt = points[i - 1], points[i], points[i + 1]
        into = (here[0] - prev[0], here[1] - prev[1])
        away = (nxt[0] - here[0], nxt[1] - here[1])
        len_in = math.hypot(*into)
        len_out = math.hypot(*away)
        if len_in < 1e-12 or len_out < 1e-12:
            continue
        # Never cut away more than half of either neighbouring segment.
        cut = min(radius, len_in / 2.0, len_out / 2.0)
        start = (here[0] - into[0] / len_in * cut, here[1] - into[1] / len_in * cut)
        end = (here[0] + away[0] / len_out * cut, here[1] + away[1] / len_out * cut)
        out.append(start)
        for step in range(1, segments):
            t = step / segments
            # Quadratic Bezier through the corner: cheap, and smooth enough
            # at the scale a 100 um bead can actually render.
            u = 1.0 - t
            out.append((
                u * u * start[0] + 2 * u * t * here[0] + t * t * end[0],
                u * u * start[1] + 2 * u * t * here[1] + t * t * end[1],
            ))
        out.append(end)
    out.append(points[-1])
    return out


class Serpentine(InfillPattern):
    """A boustrophedon: hatch rows joined end to end into one snake."""

    name = "serpentine"
    overlap_class = NONE
    description = "one continuous S-shaped stroke; no crossings, nothing stranded"
    families = 1

    def generate(self, ctx: FillContext) -> list[LineString]:
        spacing = ctx.scale
        if spacing <= 0 or ctx.region.is_empty:
            return []
        rows = hatch(ctx.region, spacing, ctx.angle_deg)
        if not rows:
            return []

        # Work axis-aligned so "the next row up" is just the next y.
        back = affinity.rotate(MultiLineString(rows), -ctx.angle_deg, origin=(0, 0))
        local_region = affinity.rotate(ctx.region, -ctx.angle_deg, origin=(0, 0))
        flat = sorted(
            (sorted(ln.coords, key=lambda p: p[0]) for ln in as_linestrings(back)),
            key=lambda pts: (round(pts[0][1], 6), pts[0][0]),
        )

        # A connector is legal only if it stays on the part.  Across the notch
        # where the tab meets the disc it would run through open air, so the
        # snake breaks there and the two halves anchor to the walls instead.
        tolerance = local_region.buffer(1e-6)
        paths: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        rightwards = True
        for pts in flat:
            run = pts if rightwards else pts[::-1]
            if current:
                link = LineString([current[-1], run[0]])
                if tolerance.contains(link):
                    current.extend(run)
                else:
                    paths.append(current)
                    current = list(run)
            else:
                current = list(run)
            rightwards = not rightwards
        if current:
            paths.append(current)

        # Round the turns by a little under half the row spacing, which is the
        # most that fits without the arc reaching the neighbouring row.
        radius = min(spacing * 0.45, spacing / 2.0)
        smoothed = [
            LineString(_rounded(path, radius)) for path in paths if len(path) >= 2
        ]
        if not smoothed:
            return []
        turned = affinity.rotate(MultiLineString(smoothed), ctx.angle_deg, origin=(0, 0))
        # Clip against a hair-widened region rather than the region itself.
        # The rows were clipped by hatch() and every connector was checked for
        # containment, so nothing here strays; but on a square the connectors
        # run exactly along the boundary, and clipping a line that lies on a
        # polygon edge shatters it into fragments on rounding error alone --
        # which would undo the single continuous stroke this pattern is for.
        return as_linestrings(ctx.region.buffer(1e-9).intersection(turned))


def _hilbert_points(x0: float, y0: float, xi: float, xj: float,
                    yi: float, yj: float, order: int, out: list) -> None:
    """Classic recursive Hilbert construction over a parallelogram frame."""
    if order <= 0:
        out.append((x0 + (xi + yi) / 2.0, y0 + (xj + yj) / 2.0))
        return
    hx, hy = xi / 2.0, xj / 2.0
    vx, vy = yi / 2.0, yj / 2.0
    _hilbert_points(x0, y0, vx, vy, hx, hy, order - 1, out)
    _hilbert_points(x0 + hx, y0 + hy, hx, hy, vx, vy, order - 1, out)
    _hilbert_points(x0 + hx + vx, y0 + hy + vy, hx, hy, vx, vy, order - 1, out)
    _hilbert_points(x0 + hx + yi, y0 + hy + yj, -vx, -vy, -hx, -hy, order - 1, out)


class Hilbert(InfillPattern):
    """A Hilbert space-filling curve, drawn at a chosen cell size."""

    name = "hilbert"
    overlap_class = NONE
    description = "space-filling curve; as few strokes as the outline allows, no crossings"
    families = 1
    #: Refuse to build a curve with more cells than this.
    max_cells = 300_000
    #: The longest run along the region's edge, in cells, used to rejoin the
    #: pieces a clip cuts the curve into.  Beyond that a join would lay a
    #: long bead hugging the wall, which is no longer this pattern.
    max_join = 4.0

    def min_scale(self, line_width: float) -> float:
        # A cell of one bead width leaves a notch on the outside of every
        # rounded turn, so the curve tops out near 89% rather than solid.
        # It closes up by about 0.6 of a bead, which is where the search is
        # allowed to stop.
        return line_width * 0.5

    def generate(self, ctx: FillContext) -> list[LineString]:
        cell = ctx.scale
        if cell <= 0 or ctx.region.is_empty:
            return []
        local = affinity.rotate(ctx.region, -ctx.angle_deg, origin=(0, 0))
        minx, miny, maxx, maxy = local.bounds
        span = max(maxx - minx, maxy - miny)
        if span <= 0:
            return []

        # The curve only exists on a 2^n grid, so pick the smallest order that
        # covers the part at the requested cell size.  Keeping the cell as the
        # knob -- rather than the order -- is what lets density stay
        # continuous instead of jumping by factors of four.
        order = max(1, math.ceil(math.log2(span / cell)))
        if 4 ** order > self.max_cells:
            return []
        side = cell * (2 ** order)

        points: list[tuple[float, float]] = []
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        origin_x, origin_y = cx - side / 2.0, cy - side / 2.0
        _hilbert_points(origin_x, origin_y, side, 0.0, 0.0, side, order, points)
        if len(points) < 2:
            return []

        # Ease the right-angle turns for the same reason as the serpentine.
        curve = LineString(_rounded(points, cell * 0.3, segments=3))
        turned = affinity.rotate(curve, ctx.angle_deg, origin=(0, 0))
        # The curve is built on a square and clipped to the part, which cuts
        # it wherever it wanders out and back.  Rejoin the pieces along the
        # region's edge so it stays the one continuous stroke it is meant to be.
        pieces = as_linestrings(ctx.region.intersection(turned))
        joined = connect_along_boundary(pieces, ctx.region, max_gap=self.max_join * cell,
                                        along=turned)
        # A corner of the curve that only just dips inside, boxed in by the
        # rest so no join can reach it, is a dot of ink rather than a path:
        # a dispenser start and stop for nothing.  Leave those out.
        stub = 2.0 * ctx.line_width
        kept = [ln for ln in joined if ln.length >= stub]
        return kept or joined


class Stitched(Serpentine):
    """A compliant serpentine with a sparse cross-stitch laid over it.

    Crossings and stiffness usually arrive together: a lattice fine enough to
    make many humps is also braced into many small cells, and a braced cell's
    compliance falls with the cube of its size.  Measured on this electrode
    the penalty is a cliff rather than a slope -- the first crossings cost
    most of the compliance, and further ones are comparatively cheap.

    This tries to get the humps without paying the whole cliff, by making the
    cells long instead of small.  The serpentine rows carry the density; a
    second family crossing them every ``ratio`` rows supplies the humps.  The
    cells are then ``ratio`` times longer than they are wide, so bending
    along the stitches stays far softer than a square lattice of the same
    hump count would be.  It is anisotropic on purpose -- most skin an
    electrode sits on curves one way far more than the other.
    """

    name = "stitched"
    overlap_class = CROSSING
    description = "serpentine with a sparse cross-stitch; humps on a compliant base"
    #: How many row spacings apart the stitches run.  This is the dial on
    #: the compromise: low packs in humps at the cost of compliance, high
    #: keeps the base soft and adds humps sparingly.
    ratio = 6.0

    def _ratio(self, ctx: FillContext) -> float:
        given = ctx.stitch_ratio
        return max(1.0, given) if given else self.ratio

    def initial_scale(self, density: float, line_width: float) -> float:
        # Rows at spacing s plus stitches at ratio*s cover (1 + 1/ratio)*w/s.
        return line_width * (1.0 + 1.0 / self.ratio) / max(density, 1e-4)

    def generate(self, ctx: FillContext) -> list[LineString]:
        rows = super().generate(ctx)
        stitches = hatch(ctx.region, ctx.scale * self._ratio(ctx), ctx.angle_deg + 90.0)
        return rows + stitches


register(Serpentine())
register(Hilbert())
register(Stitched())
