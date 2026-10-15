"""Splitting a layer into print passes that a region slicer cannot misread.

The Voltera NOVA does not dispense along the strokes in a Gerber file.  It
reads the artwork as filled copper -- every stroke widened to its aperture and
merged with whatever it touches -- and hands that region to Arachne, which
lays beads to fill it.  A strip exactly one bead wide comes back as one bead
down its middle, which is what was drawn.  Anywhere strips cross or meet the
merged region is wider and oddly shaped, and Arachne improvises: lines kink
towards the junction, stop short of it, restart offset, and sprout stubs and
blobs to fill the leftover corners.  It also never lays one bead over another,
so a crossing drawn in one file is printed flat.

So the plate is written as several files per layer, printed in order, each of
which holds only strips that touch nothing else in it:

1. the walls -- perimeters plus anything meant to be solid, which Arachne is
   welcome to fill however it likes.  It is written as the area those cover
   (``LayerPaths.walls_area``), not as their strokes: the strokes' union has a
   sawtooth edge where hatch lines end and pinholes where beads meet, and
   Arachne kinks and stubs its way round every one;
2. the infill, joined into the straightest continuous runs and then split so
   that no two runs in one pass touch.  Crossing families land in different
   passes, so each crossing is printed bead over bead, as designed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

from .geometry import (
    as_polygons,
    ring_lines,
    robust_difference,
    robust_union,
    tidy_area,
    toolpath_area,
)

__all__ = ["join_straightest", "separate_touching", "print_passes", "pass_artwork",
           "PassFill", "predict_fill"]


def join_straightest(lines: list[LineString], max_turn_deg: float = 75.0,
                     tol: float = 1e-4) -> list[LineString]:
    """Join paths that meet end to end into the straightest non-branching runs.

    At every node the incident ends are paired off by how nearly straight
    they continue, best first, up to `max_turn_deg`; ends left over stop
    there.  A honeycomb comes apart into zigzags and short connectors rather
    than a separate start and stop for every wall -- each start being a
    chance for the dispenser to leave a blob.
    """
    segs = [list(ln.coords) for ln in lines if ln.length > 0]
    if not segs:
        return []

    def key(p):
        return (round(p[0] / tol), round(p[1] / tol))

    ends: dict[tuple, list[tuple[int, int]]] = {}
    for i, c in enumerate(segs):
        ends.setdefault(key(c[0]), []).append((i, 0))
        ends.setdefault(key(c[-1]), []).append((i, 1))

    def heading(i: int, end: int) -> float:
        """Direction leaving the node along path i from the given end."""
        c = segs[i]
        a, b = (c[0], c[1]) if end == 0 else (c[-1], c[-2])
        return math.atan2(b[1] - a[1], b[0] - a[0])

    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for incident in ends.values():
        pairs = []
        for x in range(len(incident)):
            for y in range(x + 1, len(incident)):
                (i, e), (j, f) = incident[x], incident[y]
                if i == j:
                    continue
                apart = abs((heading(i, e) - heading(j, f) + math.pi) % (2 * math.pi) - math.pi)
                turn = math.degrees(math.pi - apart)
                if turn <= max_turn_deg:
                    pairs.append((turn, incident[x], incident[y]))
        for _, a, b in sorted(pairs):
            if a not in partner and b not in partner:
                partner[a], partner[b] = b, a

    seen: set[int] = set()
    out: list[LineString] = []
    for start in range(len(segs)):
        if start in seen:
            continue
        # Back up to a free end, so a run is walked from its beginning.
        i, enter = start, 0
        for _ in range(len(segs)):
            back = partner.get((i, enter))
            if back is None or back[0] == start:
                break
            i, enter = back[0], 1 - back[1]
        pts: list[tuple[float, float]] = []
        while i not in seen:
            seen.add(i)
            c = segs[i] if enter == 0 else segs[i][::-1]
            pts.extend(c if not pts else c[1:])
            nxt = partner.get((i, 1 - enter))
            if nxt is None:
                break
            i, enter = nxt
        if len(pts) >= 2:
            out.append(LineString(pts))
    return out


def separate_touching(lines: list[LineString], width: float,
                      clearance: float) -> list[list[LineString]]:
    """Share the paths out into as few groups as possible, none touching within one.

    Two strips `width` wide touch when their centrelines come within `width`
    of each other; `clearance` keeps a sliver of daylight between them too,
    so the artwork cannot merge them however it rounds.  Greedy colouring,
    most-constrained path first.
    """
    lines = [ln for ln in lines if ln.length > 0]
    if not lines:
        return []
    reach = width + clearance
    tree = STRtree(lines)
    touching: list[set[int]] = [set() for _ in lines]
    for i, ln in enumerate(lines):
        for j in tree.query(ln, predicate="dwithin", distance=reach):
            j = int(j)
            if j != i:
                touching[i].add(j)
    group: dict[int, int] = {}
    for i in sorted(range(len(lines)), key=lambda k: -len(touching[k])):
        taken = {group[j] for j in touching[i] if j in group}
        g = 0
        while g in taken:
            g += 1
        group[i] = g
    out: list[list[LineString]] = [[] for _ in range(max(group.values()) + 1)]
    for i in range(len(lines)):
        out[group[i]].append(lines[i])
    return out


def print_passes(layer, width: float, clearance: float | None = None
                 ) -> list[tuple[str, list[LineString]]]:
    """A layer's toolpaths as ``(role, paths)`` passes, in the order to print them.

    The walls come first, with anything meant to be solid -- the fill outside
    the modifiers, or a pad that is solid all through -- since a solid area
    is one Arachne should fill.  The infill follows in as many passes as it
    takes for no two strips in one to touch.
    """
    clearance = 0.1 * width if clearance is None else clearance
    walls = layer.perimeters + layer.solid
    infill = layer.infill
    if layer.pattern == "solid":
        walls, infill = walls + infill, []
    out = [("walls", walls)] if walls else []
    groups = separate_touching(join_straightest(infill), width, clearance)
    out += [("infill", group) for group in groups]
    return out


def pass_artwork(layer, width: float):
    """A layer's print passes as the copper each GTL file draws: ``(role, area)``.

    The same split and the same area :meth:`SweepResult.gerber_passes` writes:
    the walls as their tidied region, each infill pass as its strokes widened
    to the bead.  It is what the NOVA's slicer is handed to fill.
    """
    out = []
    for k, (role, paths) in enumerate(print_passes(layer, width)):
        if k == 0 and role == "walls":
            area = tidy_area(layer.walls_area, width / 2.0)
        else:
            area = toolpath_area(paths, width)
        out.append((role, area))
    return out


@dataclass
class PassFill:
    """How a region slicer is likely to fill one pass with a given tip."""

    loops: list[LineString] = field(default_factory=list)
    # Per loop: is it a single bead down a strip only one tip wide?  Its ring
    # then runs out along one side and back along the other, a hair apart.
    single: list[bool] = field(default_factory=list)
    gaps: Polygon = field(default_factory=Polygon)   # artwork the tip cannot reach
    area: float = 0.0

    @property
    def length(self) -> float:
        """Nozzle travel while dispensing; a single bead is laid once, not twice."""
        return sum(ln.length / (2 if one else 1)
                   for ln, one in zip(self.loops, self.single, strict=True))

    @property
    def gap_fraction(self) -> float:
        return min(1.0, self.gaps.area / self.area) if self.area > 0 else 0.0


#: A region slicer narrows a bead this far below the tip before it gives a
#: strip up as too thin -- Arachne's minimum wall width is about 85% of the
#: nozzle by default -- so a strip drawn exactly one bead wide still gets one.
MIN_BEAD = 0.85


def predict_fill(area, tip: float, max_loops: int = 400) -> PassFill:
    """Concentric loops a `tip` wide filling `area`, and what they miss.

    An approximation of what the NOVA does with a GTL pass, not its slicer:
    loops stepped in a tip at a time from the edge, the first half a tip in,
    as long as there is room for a bead at least ``MIN_BEAD`` of the tip.
    What the beads leave uncovered -- bar slivers a tenth of a tip across,
    which a bead spreads over -- comes back as `gaps`: the places the slicer
    must improvise, which is where the kinks, stubs and blobs come from.
    """
    polys = as_polygons(area)
    fill = PassFill(area=sum(p.area for p in polys))
    if not polys or tip <= 0:
        return fill
    slack = (1.0 - MIN_BEAD) * tip / 2.0
    loops: list[LineString] = []
    for k in range(max_loops):
        inset = robust_union([p.buffer(-((k + 0.5) * tip - slack), join_style="round")
                              for p in polys])
        rings = as_polygons(inset)
        if not rings:
            break
        for ring in rings:
            lines = ring_lines([ring])
            one = ring.buffer(-1.01 * slack).is_empty
            loops.extend(lines)
            fill.single.extend([one] * len(lines))
    fill.loops = loops
    covered = toolpath_area(loops, tip)
    missed = robust_difference(robust_union(polys), covered) if loops else robust_union(polys)
    fill.gaps = tidy_area(missed, 0.2 * tip) if not missed.is_empty else Polygon()
    if fill.gaps is None or fill.gaps.is_empty:
        fill.gaps = Polygon()
    return fill
