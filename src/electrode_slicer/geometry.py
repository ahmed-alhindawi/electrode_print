"""Shapely helpers shared by the slicer, the fill patterns and the metrics."""

from __future__ import annotations

import math

import shapely
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import unary_union

__all__ = [
    "as_linestrings",
    "as_polygons",
    "robust_intersection",
    "robust_union",
    "toolpath_area",
    "ring_lines",
    "hatch",
    "extend_ends",
    "simplify_paths",
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
    return as_linestrings(region.intersection(grid))


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
        clipped = as_linestrings(grown.intersection(limit))
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
