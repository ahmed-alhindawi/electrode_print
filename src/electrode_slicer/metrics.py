"""Measuring how much a layer's toolpath runs over itself.

The working hypothesis behind this tool is that beads crossing over one
another leave discrete humps that behave like a micro-needle array at the
skin interface.  These functions measure that directly, so a pattern sweep
produces a number to regress impedance against rather than a qualitative
"this one overlaps".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from shapely import STRtree
from shapely.geometry import LineString, Point

from .geometry import as_polygons, robust_intersection, robust_union

__all__ = ["OverlapStats", "analyse_overlap", "crossing_nodes", "crossing_points"]


def _iter_points(geom):
    """Every Point in an intersection result (ignoring collinear overlaps)."""
    if geom.is_empty:
        return
    kind = geom.geom_type
    if kind == "Point":
        yield geom
    elif kind in ("MultiPoint", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_points(part)


def _is_interior(line: LineString, point, tol: float) -> bool:
    """True when the point is mid-path rather than at a bead end."""
    coords = line.coords
    return (
        math.dist(coords[0], (point.x, point.y)) > tol
        and math.dist(coords[-1], (point.x, point.y)) > tol
    )


def crossing_nodes(infill: list[LineString], snap: float = 1e-3) -> list[dict]:
    """Every place two infill beads meet, with how many meet there.

    ``transversal`` counts the pairs that genuinely cross rather than touching
    end to end, which is what separates a hump from a lattice node.
    """
    if len(infill) < 2:
        return []
    tree = STRtree(infill)
    nodes: dict[tuple[int, int], dict] = {}
    for i, a in enumerate(infill):
        for j in tree.query(a):
            j = int(j)
            if j <= i:
                continue
            b = infill[j]
            for pt in _iter_points(a.intersection(b)):
                key = (round(pt.x / snap), round(pt.y / snap))
                node = nodes.setdefault(
                    key, {"x": key[0] * snap, "y": key[1] * snap,
                          "members": set(), "transversal": 0},
                )
                node["members"].update((i, j))
                if _is_interior(a, pt, snap) and _is_interior(b, pt, snap):
                    node["transversal"] += 1
    return list(nodes.values())


def crossing_points(infill: list[LineString], snap: float = 1e-3) -> list[tuple[float, float, int]]:
    """Just the humps: ``(x, y, beads meeting)`` for each bead-over-bead crossing.

    Kept out of :class:`OverlapStats` deliberately -- a dense sweep has
    hundreds of thousands of these, and only the viewer ever wants them.
    """
    return [
        (node["x"], node["y"], len(node["members"]))
        for node in crossing_nodes(infill, snap)
        if node["transversal"]
    ]


@dataclass
class OverlapStats:
    """How much a layer's toolpath runs over itself."""

    islands: int = 1              # electrically separate blobs of ink in the layer
    connected_fraction: float = 1.0   # share of the ink in the largest blob

    crossings: int = 0            # nodes where a bead is laid over another bead
    junctions: int = 0            # nodes where >=3 bead ends meet without crossing
    crossings_per_mm2: float = 0.0
    max_multiplicity: int = 0     # most beads meeting at a single node
    mean_multiplicity: float = 0.0    # average beads stacked at a crossing
    overlap_area_mm2: float = 0.0     # doubly-deposited footprint, all paths
    overlap_fraction: float = 0.0     # as a share of the footprint laid down
    excess_volume_mm3: float = 0.0    # material that has nowhere to go but up
    crossing_excess_volume_mm3: float = 0.0  # the part of it piled at crossings
    hump_height_mm: float = 0.0       # how proud a typical hump stands

    def as_dict(self) -> dict:
        return asdict(self)


def analyse_overlap(
    infill: list[LineString],
    all_paths: list[LineString],
    line_width: float,
    layer_height: float,
    region_area: float,
    snap: float = 1e-3,
) -> OverlapStats:
    """Count self-crossings in the infill and the extra material they deposit.

    Crossings are counted on the infill alone: every pattern ends its beads on
    the perimeter loop, so including the walls would add a near-constant term
    that swamps the difference between patterns.  The volume figures cover all
    paths, because that is what the nozzle actually puts down.

    The overlap figures compare beads against each other, so a single bead that
    doubles back over itself is not counted.  None of the patterns here do that.
    """
    stats = OverlapStats()
    if not all_paths:
        return stats

    # Extra material = what we dispense minus the footprint it ends up in.
    # Anything the union swallows was laid on top of an earlier bead.
    beads = [ln.buffer(line_width / 2.0, quad_segs=8) for ln in all_paths]
    dispensed = sum(b.area for b in beads)
    merged = robust_union(beads)
    # A printed electrode only conducts through ink that is joined up.  Infill
    # that lands as separate islands is electrically dead however good it looks,
    # so count the connected blobs rather than assume there is one.
    blobs = as_polygons(merged)
    stats.islands = len(blobs)
    # The count alone understates this badly: four nested rings sounds minor,
    # but if only the outer one touches a perimeter then most of the ink is
    # dead.  Measure the share that is joined to the largest blob, which is
    # the one carrying the perimeters and the connection tab.
    footprint = sum(b.area for b in blobs)
    if footprint > 0:
        stats.connected_fraction = max(b.area for b in blobs) / footprint
    stats.overlap_area_mm2 = max(0.0, dispensed - merged.area)
    stats.overlap_fraction = stats.overlap_area_mm2 / dispensed if dispensed else 0.0
    stats.excess_volume_mm3 = stats.overlap_area_mm2 * layer_height

    points: list[tuple[float, float]] = []
    multiplicities: list[int] = []
    for node in crossing_nodes(infill, snap):
        if node["transversal"]:
            stats.crossings += 1
            multiplicities.append(len(node["members"]))
            points.append((node["x"], node["y"]))
        elif len(node["members"]) >= 3:
            stats.junctions += 1

    if region_area > 0:
        stats.crossings_per_mm2 = stats.crossings / region_area
    if multiplicities:
        stats.max_multiplicity = max(multiplicities)
        stats.mean_multiplicity = sum(multiplicities) / len(multiplicities)
        # A crossing is beads stacked on beads, so it stands one nominal layer
        # proud for each pass beyond the first.  This is the height that the
        # micro-needle argument turns on.
        stats.hump_height_mm = (stats.mean_multiplicity - 1.0) * layer_height
        # Measure the pile-up in a window around the crossings only, so it can
        # be told apart from the overlap that neighbouring parallel beads
        # produce anywhere in the layer.
        window = robust_union(
            [Point(x, y).buffer(line_width, quad_segs=8) for x, y in points]
        )
        local = [b for b in beads if b.intersects(window)]
        if local:
            local_dispensed = sum(robust_intersection(b, window).area for b in local)
            local_footprint = robust_intersection(merged, window).area
            stats.crossing_excess_volume_mm3 = (
                max(0.0, local_dispensed - local_footprint) * layer_height
            )
    return stats
