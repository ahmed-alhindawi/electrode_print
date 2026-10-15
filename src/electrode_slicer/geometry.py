"""Shapely helpers shared by the slicer, the fill patterns and the metrics."""

from __future__ import annotations

import math
import random
import sys

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.polygon import orient
from shapely.ops import substring, unary_union

__all__ = [
    "as_linestrings",
    "as_polygons",
    "robust_intersection",
    "robust_union",
    "robust_difference",
    "toolpath_area",
    "ring_lines",
    "hatch",
    "extend_ends",
    "simplify_paths",
    "chain_paths",
    "connect_along_boundary",
    "order_paths",
    "fit_arcs",
    "region_contours",
    "tidy_area",
]


def as_linestrings(geom) -> list[LineString]:
    """Flatten any shapely geometry down to its non-degenerate LineStrings."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []
    if isinstance(geom, MultiLineString | GeometryCollection):
        out: list[LineString] = []
        for part in geom.geoms:
            out.extend(as_linestrings(part))
        return out
    return []


def as_polygons(geom) -> list[Polygon]:
    """Flatten any shapely geometry down to its non-degenerate Polygons."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    if isinstance(geom, MultiPolygon | GeometryCollection):
        out: list[Polygon] = []
        for part in geom.geoms:
            out.extend(as_polygons(part))
        return out
    return []


#: Coordinates are snapped to this grid when an overlay fails.  It is a ten
#: thousandth of a micron -- orders of magnitude below anything the machine
#: can position to -- so nothing moves anywhere that matters.
SNAP_GRID = 1e-7

#: Under Pyodide (the static web build) GEOS' C++ exceptions are not turned
#: into Python ones: a TopologyException aborts the whole interpreter instead
#: of reaching an ``except``.  There the overlays go straight to snap-rounding
#: on SNAP_GRID, which GEOS guarantees not to throw, rather than trying the
#: exact overlay first and recovering.
SNAP_FIRST = sys.platform == "emscripten"


def robust_union(geoms):
    """Union geometries, surviving GEOS' robustness failures.

    Overlaying hundreds of beads that run very nearly tangent to one another
    can defeat GEOS' exact predicates and raise a TopologyException even
    though every input is valid.  Dense space-filling curves do it readily.
    Snapping to a fine grid removes the ambiguity that trips it up.
    """
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return Polygon()
    if SNAP_FIRST:
        return shapely.union_all(geoms, grid_size=SNAP_GRID)
    try:
        return unary_union(geoms)
    except shapely.errors.GEOSException:
        pass
    snapped = [shapely.set_precision(g, SNAP_GRID) for g in geoms]
    snapped = [g for g in snapped if not g.is_empty]
    try:
        return unary_union(snapped)
    except shapely.errors.GEOSException:
        # Last resort: fold them in one at a time, so a single awkward pair
        # cannot take the whole overlay down with it.
        total = snapped[0]
        for geom in snapped[1:]:
            try:
                total = total.union(geom)
            except shapely.errors.GEOSException:
                continue
        return total


def robust_intersection(a, b):
    """Intersect two geometries, surviving GEOS' robustness failures.

    The same near-tangent bead geometry that defeats a union defeats the
    clip that follows it, so both halves of the operation need the same
    treatment or the overlay just fails one step later.
    """
    if a is None or b is None or a.is_empty or b.is_empty:
        return Polygon()
    if SNAP_FIRST:
        return shapely.intersection(a, b, grid_size=SNAP_GRID)
    try:
        return a.intersection(b)
    except shapely.errors.GEOSException:
        pass
    try:
        return shapely.set_precision(a, SNAP_GRID).intersection(
            shapely.set_precision(b, SNAP_GRID)
        )
    except shapely.errors.GEOSException:
        # Buffering by nothing rebuilds the topology from scratch, which is
        # slower but copes with cases snapping alone does not.
        try:
            return a.buffer(0).intersection(b.buffer(0))
        except shapely.errors.GEOSException:   # pragma: no cover
            return Polygon()


def robust_difference(a, b):
    """`a` less `b`, surviving GEOS' robustness failures as the others do."""
    if a is None or a.is_empty:
        return Polygon()
    if b is None or b.is_empty:
        return a
    if SNAP_FIRST:
        return shapely.difference(a, b, grid_size=SNAP_GRID)
    try:
        return a.difference(b)
    except shapely.errors.GEOSException:
        return shapely.difference(a, b, grid_size=SNAP_GRID)


def toolpath_area(lines: list[LineString], line_width: float):
    """The material a set of centrelines deposits, as a polygon."""
    if not lines:
        return Polygon()
    return robust_union([ln.buffer(line_width / 2.0, quad_segs=8) for ln in lines])


def ring_lines(polys: list[Polygon]) -> list[LineString]:
    """Every boundary ring (exterior and holes) of a set of polygons."""
    out = []
    for poly in polys:
        out.append(LineString(poly.exterior.coords))
        out.extend(LineString(ring.coords) for ring in poly.interiors)
    return out


def hatch(region, spacing: float, angle_deg: float) -> list[LineString]:
    """Parallel straight lines at `angle_deg`, clipped to `region`.

    The primitive behind every straight-line fill: one call per direction
    family, so a grid is two calls and a star is four.
    """
    if spacing <= 0 or region.is_empty:
        return []
    minx, miny, maxx, maxy = region.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    reach = math.hypot(maxx - minx, maxy - miny) / 2.0 + spacing
    count = int(math.ceil(2.0 * reach / spacing))
    if count > 100_000:  # pathological spacing; refuse rather than hang
        return []
    lines = [
        LineString([(-reach, -reach + i * spacing), (reach, -reach + i * spacing)])
        for i in range(count + 1)
    ]
    grid = affinity.rotate(MultiLineString(lines), angle_deg, origin=(0, 0))
    grid = affinity.translate(grid, cx, cy)
    return as_linestrings(robust_intersection(region, grid))


def _step_out(frm, to, distance: float):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return to
    return (to[0] + dx / norm * distance, to[1] + dy / norm * distance)


def extend_ends(
    lines: list[LineString],
    distance: float,
    region,
    limit=None,
    tol: float = 1e-6,
) -> list[LineString]:
    """Push clipped bead ends outwards so they weld into the perimeter loops.

    Slicers overlap infill into the walls to avoid a seam.  Doing it by
    extending the ends rather than by growing the infill region keeps every
    pattern's overlap proportional to how many bead ends it has, instead of
    handing closed-loop patterns like concentric a whole extra ring's worth of
    doubled-up material.

    Only ends created by clipping -- the ones lying on the region boundary --
    are extended.  A honeycomb's interior lattice nodes are bead ends too, and
    nudging those past each other would fake a crossing at every node.
    """
    if distance <= 0:
        return lines
    boundary = region.boundary
    out: list[LineString] = []
    for line in lines:
        if line.is_ring or line.length == 0:
            out.append(line)
            continue
        coords = list(line.coords)
        touched = False
        if boundary.distance(Point(coords[0])) <= tol:
            coords[0] = _step_out(coords[1], coords[0], distance)
            touched = True
        if boundary.distance(Point(coords[-1])) <= tol:
            coords[-1] = _step_out(coords[-2], coords[-1], distance)
            touched = True
        if not touched:
            out.append(line)
            continue
        grown = LineString(coords)
        if limit is None:
            out.append(grown)
            continue
        clipped = as_linestrings(robust_intersection(grown, limit))
        out.extend(clipped if clipped else [line])
    return out


def simplify_paths(lines: list[LineString], tolerance: float) -> list[LineString]:
    """Thin out toolpath vertices below the machine's resolution.

    An STL tessellates a circle into hundreds of facets, and the slice inherits
    every one.  Emitting them all makes a plate-sized SVG tens of megabytes and
    feeds the nozzle thousands of sub-micron moves, so drop the vertices that
    move the path by less than `tolerance`.  Applied before the geometry is
    measured, so the reported numbers describe what actually gets printed.
    """
    if tolerance <= 0:
        return lines
    out: list[LineString] = []
    for line in lines:
        thin = line.simplify(tolerance, preserve_topology=False)
        # A ring simplified past a triangle is no longer a loop; keep the
        # original rather than emit a degenerate path.
        if thin.is_empty or (line.is_ring and len(thin.coords) < 4):
            out.append(line)
        else:
            out.append(thin)
    return out


def chain_paths(lines: list[LineString], tolerance: float = 1e-4) -> list[LineString]:
    """Join paths that meet end to end into as few continuous strokes as possible.

    Fill patterns come out as many short pieces -- a honeycomb is hundreds of
    wall segments meeting three to a node -- and a tool that plans its own
    dispense from the artwork starts and stops at every one.  Treat each path
    as an edge between its two ends and cover the graph with Euler trails:
    every segment is drawn exactly once, so nothing is added or lost, and a
    connected piece needs only one stroke per pair of odd-degree ends.  Ends
    closer than `tolerance` mm count as the same point.
    """
    lines = [ln for ln in lines if len(ln.coords) >= 2]
    if not lines:
        return []
    ends = [pt for ln in lines for pt in (ln.coords[0], ln.coords[-1])]
    node = list(range(len(ends)))

    def root(i: int) -> int:
        while node[i] != i:
            node[i] = node[node[i]]
            i = node[i]
        return i

    for a, b in cKDTree(ends).query_pairs(tolerance):
        node[root(a)] = root(b)

    graph = nx.MultiGraph()
    for idx, ln in enumerate(lines):
        start, end = root(2 * idx), root(2 * idx + 1)
        graph.add_edge(start, end, key=idx, start=start, coords=list(ln.coords))

    out: list[LineString] = []
    for members in nx.connected_components(graph):
        piece = nx.MultiGraph(graph.subgraph(members))
        odd = [n for n in piece if piece.degree(n) % 2]
        # Pair the odd ends up through one extra node; the circuit through it
        # then splits into the fewest open trails that cover everything.
        virtual = ("virtual",)
        for n in odd:
            piece.add_edge(virtual, n, key=("virtual", n))
        source = virtual if odd else next(iter(members))
        trail: list[tuple[float, float]] = []
        for u, v, key in nx.eulerian_circuit(piece, source=source, keys=True):
            if virtual in (u, v):
                if len(trail) >= 2:
                    out.append(LineString(trail))
                trail = []
                continue
            data = piece.edges[u, v, key]
            coords = data["coords"] if data["start"] == u else data["coords"][::-1]
            trail.extend(coords if not trail else coords[1:])
        if len(trail) >= 2:
            out.append(LineString(trail))
    return out


def connect_along_boundary(lines: list[LineString], region, max_gap: float = math.inf,
                           along: LineString | None = None,
                           tol: float = 1e-6) -> list[LineString]:
    """Rejoin paths a clip cut apart, by running along the region's edge.

    Clipping a continuous path to a region leaves pieces whose ends sit on
    the region's boundary.  Ends that are neighbours along the boundary --
    no other end between them -- may be joined by the stretch of boundary
    between them.  Since no other end lies on that stretch, a connector meets
    nothing but the two pieces it joins: no crossing and no new junction.

    When every end lies on one ring the joins are chosen exactly: pieces that
    never cross nest like brackets, so a single path through all of them
    must pair the ends off around the ring in one of a few ways, and those
    are all tried (see :func:`_one_path_around`).  The shortest set of joins
    that makes one path, with none longer than `max_gap`, wins.  Otherwise
    -- ends on several rings, or no single path possible -- neighbours are
    joined greedily, shortest first, never closing a loop.  `along`, the
    path before it was clipped, puts the joins that just shortcut one of its
    excursions first in that greedy pass.
    """
    lines = [ln for ln in lines if ln.length > 0]
    if len(lines) < 2:
        return lines
    rings = [LineString(r.coords) for p in as_polygons(region)
             for r in (p.exterior, *p.interiors)]
    # Every end that lies on a ring, as (position along it, piece, end).
    on_ring: dict[int, list[tuple[float, int, int]]] = {}
    for i, ln in enumerate(lines):
        for end, pt in ((0, Point(ln.coords[0])), (1, Point(ln.coords[-1]))):
            for r, ring in enumerate(rings):
                if ring.distance(pt) <= tol:
                    on_ring.setdefault(r, []).append((ring.project(pt), i, end))
                    break
    for ends in on_ring.values():
        ends.sort()

    joins = None
    if len(on_ring) == 1:
        (r, ends), = on_ring.items()
        if len(ends) == 2 * len(lines):
            joins = _one_path_around(ends, rings[r].length, len(lines), max_gap)
            if joins is not None:
                joins = [(r, a, b) for a, b in joins]
    if joins is None:
        joins = _greedy_joins(on_ring, rings, len(lines), max_gap, lines, along)
        if len(on_ring) == 1 and len(joins) < len(lines) - 1:
            (r, ends), = on_ring.items()
            joins = _refine_joins(joins, ends, rings[r].length, len(lines), max_gap)

    connectors: list[LineString] = []
    for r, (pa, ia, ea), (pb, ib, eb) in joins:
        if r is None:
            (r,) = on_ring
        ring, total = rings[r], rings[r].length
        if pb >= pa:
            arc = list(substring(ring, pa, pb).coords)
        else:                                  # across the ring's start
            arc = list(substring(ring, pa, total).coords) + list(substring(ring, 0, pb).coords)[1:]
        # Land exactly on the pieces' ends, so the join is watertight.
        arc[0] = lines[ia].coords[-1 if ea else 0]
        arc[-1] = lines[ib].coords[-1 if eb else 0]
        connectors.append(LineString(arc))
    if not connectors:
        return lines
    return chain_paths(lines + connectors)


def _one_path_around(ends, total: float, pieces: int, max_gap: float):
    """Joins between neighbouring ends that string every piece into one path.

    `ends` is every piece end, sorted round one ring.  Each join pairs an end
    with the next one round, so a set of joins is a matching on that cycle of
    ends; one path through n pieces needs n - 1 of them, which leaves two ends
    unjoined.  With those two fixed, the joins on each arc between them are
    forced -- so every candidate is a choice of the two loose ends.  Returns
    the cheapest valid set as ``[(end, next end), ...]``, or None.
    """
    m = len(ends)
    gap = [(ends[(k + 1) % m][0] - ends[k][0]) % total for k in range(m)]

    def valid(starts):
        """The joins (k, k+1) for each k in `starts`, if they make one path."""
        group = list(range(pieces))

        def root(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        cost = 0.0
        for k in starts:
            if gap[k] > max_gap or gap[k] <= 0:
                return None
            a, b = root(ends[k][1]), root(ends[(k + 1) % m][1])
            if a == b:
                return None
            group[a] = b
            cost += gap[k]
        return cost

    best = None
    # Leave ends u and v loose; the m - 2 others pair off with their next
    # neighbour round each arc between them, which needs an even count on each.
    for u in range(m):
        for step in range(1, m, 2):
            v = (u + step) % m
            if v < u:
                continue
            starts = list(range(u + 1, v - 1, 2)) + [k % m for k in range(v + 1, u + m - 1, 2)]
            cost = valid(starts)
            if cost is not None and (best is None or cost < best[0]):
                best = (cost, starts)
    if best is None:
        return None
    return [(ends[k], ends[(k + 1) % m]) for k in best[1]]


def _refine_joins(joins, ends, total: float, pieces: int, max_gap: float):
    """Fewer separate paths, by reworking the joins round one ring.

    Joins pair each end with its neighbour round the ring, so between two
    ends left loose the joins alternate, and there are just two ways to lay
    them: shifted by one, the run takes in both loose ends and two paths
    become one -- if that closes no loop.  Repeats that while it helps,
    trying a loose end two places either way when nothing merges outright.
    """
    m = len(ends)
    index = {e: k for k, e in enumerate(ends)}
    gap = [(ends[(k + 1) % m][0] - ends[k][0]) % total for k in range(m)]
    allowed = [0 < g <= max_gap and ends[k][1] != ends[(k + 1) % m][1]
               for k, g in enumerate(gap)]
    chosen = {index[a] for _, a, _ in joins}       # join k runs from end k to k + 1

    def acyclic(arcs) -> bool:
        group = list(range(pieces))

        def root(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        for k in arcs:
            a, b = root(ends[k][1]), root(ends[(k + 1) % m][1])
            if a == b:
                return False
            group[a] = b
        return True

    def loose(arcs):
        taken = set()
        for k in arcs:
            taken.update((k, (k + 1) % m))
        return [k for k in range(m) if k not in taken]

    def merged(arcs, a, b):
        """`arcs` with the run from loose end a round to loose end b re-laid."""
        span = [(a + i) % m for i in range((b - a) % m + 1)]
        inside = set(span[:-1])
        new = {k for k in arcs if k not in inside} | set(span[0:-1:2])
        if not all(allowed[k] for k in set(span[0:-1:2])):
            return None
        return new if acyclic(new) else None

    def cost(arcs):
        return sum(gap[k] for k in arcs)

    for _ in range(4 * m):
        free = loose(chosen)
        if len(free) < 2:
            break
        best = None
        pairs = list(zip(free, free[1:] + free[:1], strict=True))
        for a, b in pairs:
            new = merged(chosen, a, b)
            if new is not None and (best is None or cost(new) < cost(best)):
                best = new
        if best is None:
            # Nothing merges as it stands: slide one loose end two places and
            # look again, keeping the first slide that unlocks a merge.
            for a in free:
                for step in (2, -2):
                    k = a if step > 0 else (a - 2) % m
                    pair = (k + 1) % m if step > 0 else (a - 1) % m
                    # The join beside the loose end moves across it.
                    if pair not in chosen or not allowed[k if step > 0 else (a - 1) % m]:
                        continue
                    slid = (chosen - {pair}) | {k if step > 0 else (a - 1) % m}
                    if not acyclic(slid):
                        continue
                    free2 = loose(slid)
                    for c, d in zip(free2, free2[1:] + free2[:1], strict=True):
                        new = merged(slid, c, d)
                        if new is not None and (best is None or cost(new) < cost(best)):
                            best = new
                if best is not None:
                    break
        if best is None:
            break
        chosen = best
    return [(None, ends[k], ends[(k + 1) % m]) for k in sorted(chosen)]


def _greedy_joins(on_ring, rings, pieces: int, max_gap: float, lines, along,
                  tries: int = 48):
    """Neighbouring ends joined so as to leave as few separate paths as it can.

    Each pass takes the candidate joins in some order and keeps every one that
    neither reuses an end nor closes a loop.  The first pass goes shortest
    first, with the shortcuts across `along`'s own excursions ahead of the
    rest; the others shuffle that order (with a fixed seed, so a part always
    slices the same).  The pass with the most joins wins, then the shortest.
    """
    consecutive: set[frozenset] = set()
    if along is not None:
        span = [(along.project(Point(ln.coords[0])), along.project(Point(ln.coords[-1])))
                for ln in lines]
        order = sorted(range(len(lines)), key=lambda i: min(span[i]))
        for i, j in zip(order, order[1:], strict=False):
            # Out at the later end of one piece, back in at the earlier end of
            # the next -- whichever way round the clip happened to return them.
            out = (i, 1 if span[i][1] >= span[i][0] else 0)
            back = (j, 0 if span[j][0] <= span[j][1] else 1)
            consecutive.add(frozenset({out, back}))
    candidates = []
    for r, ends in on_ring.items():
        total = rings[r].length
        for k, here in enumerate(ends):
            there = ends[(k + 1) % len(ends)]
            gap = (there[0] - here[0]) % total if len(ends) > 1 else 0.0
            if here[1:] != there[1:] and 0 < gap <= max_gap:
                rank = 0 if frozenset({here[1:], there[1:]}) in consecutive else 1
                candidates.append((rank, gap, r, here, there))
    candidates.sort()

    def attempt(ordered):
        group = list(range(pieces))

        def root(i: int) -> int:
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        used: set[tuple[int, int]] = set()
        joins, cost = [], 0.0
        for _, gap, r, here, there in ordered:
            a, b = here[1:], there[1:]
            if a in used or b in used or root(a[0]) == root(b[0]):
                continue
            used.update((a, b))
            group[root(a[0])] = root(b[0])
            joins.append((r, here, there))
            cost += gap
        return joins, cost

    best, best_key = None, None
    shuffle = random.Random(0)
    for t in range(max(1, tries)):
        ordered = candidates if t == 0 else shuffle.sample(candidates, len(candidates))
        joins, cost = attempt(ordered)
        key = (-len(joins), cost)
        if best_key is None or key < best_key:
            best, best_key = joins, key
        if len(joins) == pieces - 1:
            break                      # one path; nothing fewer is possible
    return best


def order_paths(lines: list[LineString]) -> list[LineString]:
    """Put paths in nearest-next order, reversing any that start further away."""
    left = list(lines)
    if not left:
        return []
    out = [left.pop(0)]
    while left:
        here = out[-1].coords[-1]
        best, flip, gap = 0, False, math.inf
        for i, ln in enumerate(left):
            for reverse, pt in ((False, ln.coords[0]), (True, ln.coords[-1])):
                d = math.dist(here, pt)
                if d < gap:
                    best, flip, gap = i, reverse, d
        chosen = left.pop(best)
        out.append(LineString(chosen.coords[::-1]) if flip else chosen)
    return out


def _circle(a, b, c):
    """Centre and radius of the circle through three points, or None if straight."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy), math.dist((ux, uy), a)


def _on_arc(pts, centre, radius, tolerance) -> float | None:
    """The signed sweep if every vertex and chord of `pts` hugs the circle."""
    sweep, turn = 0.0, 0.0
    for p, q in zip(pts, pts[1:], strict=False):
        if abs(math.dist(q, centre) - radius) > tolerance:
            return None
        # The chord bows inward by its sagitta; that must stay close too.
        if radius - math.sqrt(max(radius**2 - (math.dist(p, q) / 2) ** 2, 0.0)) > tolerance:
            return None
        step = (math.atan2(q[1] - centre[1], q[0] - centre[0])
                - math.atan2(p[1] - centre[1], p[0] - centre[0]))
        step = (step + math.pi) % (2 * math.pi) - math.pi
        if step == 0 or (turn and (step > 0) != (turn > 0)):
            return None
        turn = step
        sweep += step
    return sweep


def fit_arcs(coords, tolerance: float, min_points: int = 4, max_sweep: float = math.pi,
             straighten: float = 0.0):
    """Describe a polyline as straight moves and circular arcs.

    A sliced circle is a polygon of short chords -- nearly a hundred on a
    5 mm pad -- which a tool that plans its dispense per primitive takes as
    that many separate moves.  Runs of at least `min_points` vertices that lie
    within `tolerance` of one circle become a single arc (at most `max_sweep`
    radians, so each arc is unambiguous); everything else stays straight.
    Returns ``("line", end)`` and ``("arc", end, centre, ccw)`` moves, starting
    from ``coords[0]``.

    `straighten` thins the straight stretches between arcs to within that
    distance, after the arcs are found -- simplifying first would leave too few
    vertices on a circle to recognise it.  Arc ends are kept exactly.
    """
    pts = [tuple(map(float, p)) for p in coords]
    moves: list[tuple] = []
    i = 0
    while i < len(pts) - 1:
        best = None
        j = i + min_points - 1
        while j < len(pts):
            fit = _circle(pts[i], pts[(i + j) // 2], pts[j])
            if fit is None or fit[1] > 1e3:
                break
            sweep = _on_arc(pts[i:j + 1], fit[0], fit[1], tolerance)
            if sweep is None or abs(sweep) > max_sweep:
                break
            best = (j, fit[0], sweep > 0)
            j += 1
        if best:
            j, centre, ccw = best
            moves.append(("arc", pts[j], centre, ccw))
            i = j
        else:
            moves.append(("line", pts[i + 1]))
            i += 1
    return _straighten(moves, pts[0], straighten) if straighten > 0 else moves


def _straighten(moves, start, tolerance):
    """Each run of straight moves thinned to within `tolerance`, ends kept."""
    out: list[tuple] = []
    run = [start]
    for move in [*moves, None]:
        if move is not None and move[0] == "line":
            run.append(move[1])
            continue
        if len(run) > 2:
            run = list(LineString(run).simplify(tolerance, preserve_topology=False).coords)
        out.extend(("line", p) for p in run[1:])
        if move is not None:
            out.append(move)
            run = [move[1]]
    return out


def tidy_area(geom, feature: float):
    """`geom` without the notches and slivers narrower than `feature`.

    A closing then an opening by half of `feature`: notches and gaps
    narrower than it fill in, spurs and slivers narrower than it go.  Parts at
    least `feature` across keep their shape, bar corners rounded to that
    radius.  Use it on an area a region slicer is about to fill, where a
    detail finer than the bead only makes the slicer improvise round it.
    """
    if geom is None or geom.is_empty or feature <= 0:
        return geom
    r = feature / 2.0
    closed = geom.buffer(r, join_style="round").buffer(-r, join_style="round")
    opened = closed.buffer(-r, join_style="round").buffer(r, join_style="round")
    polys = as_polygons(opened)
    if not polys:
        return Polygon()
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def region_contours(geom, tolerance: float = 0.0) -> list[list[tuple[str, list]]]:
    """An area as Gerber region contours, its holes joined on by cut-ins.

    A Gerber region is filled solid, and a second contour inside it is just
    more copper, not a hole.  The portable way to cut one out -- the one the
    Gerber spec recommends and PCB tools write -- is a cut-in: the outline
    runs in to the hole along a bridge, round it the other way, and back out
    along the same bridge, so the bridge encloses nothing.

    Each contour is a list of ``(kind, points)`` pieces drawn end to end:
    ``"ring"`` pieces follow the boundary and may be written as arcs;
    ``"bridge"`` pieces must stay straight, so the way out retraces the way
    in exactly.  Vertices that move a ring by less than `tolerance` are dropped.
    """
    out = []
    for poly in as_polygons(geom):
        if tolerance > 0:
            thin = poly.simplify(tolerance, preserve_topology=True)
            if isinstance(thin, Polygon) and thin.is_valid and not thin.is_empty:
                poly = thin
        poly = orient(poly, 1.0)                 # outline CCW, holes CW
        pieces = [("ring", [p[:2] for p in poly.exterior.coords])]
        bridges: list[LineString] = []
        # Nearest the outline first, so each bridge is as short as it can be.
        for hole in sorted(poly.interiors, key=poly.exterior.distance):
            joined = _cut_in(pieces, [p[:2] for p in hole.coords], poly, bridges)
            if joined is not None:
                pieces = joined
        out.append(pieces)
    return out


def _cut_in(pieces, hole, poly, bridges):
    """`pieces` with `hole` spliced in along the shortest clear bridge, or None."""
    hole_pts = hole[:-1]
    anchors = [(k, i, pt) for k, (kind, pts) in enumerate(pieces) if kind == "ring"
               for i, pt in enumerate(pts)]
    tree = cKDTree([a[2] for a in anchors])
    candidates = []
    for j, h in enumerate(hole_pts):
        dists, idx = tree.query(h, k=min(8, len(anchors)))
        for d, a in zip(np.atleast_1d(dists), np.atleast_1d(idx), strict=True):
            candidates.append((d, int(a), j))
    for _, a, j in sorted(candidates):
        k, i, anchor = anchors[a]
        h = hole_pts[j]
        bridge = LineString([anchor, h])
        # Clear of the other holes, and not across a bridge already laid.
        if bridge.length == 0 or not poly.covers(bridge):
            continue
        if any(bridge.crosses(b) for b in bridges):
            continue
        bridges.append(bridge)
        ring = pieces[k][1]
        around = hole_pts[j:] + hole_pts[:j] + [h]
        spliced = [("ring", ring[:i + 1]), ("bridge", [anchor, h]), ("ring", around),
                   ("bridge", [h, anchor]), ("ring", ring[i:])]
        return pieces[:k] + [p for p in spliced if len(p[1]) >= 2] + pieces[k + 1:]
    return None
