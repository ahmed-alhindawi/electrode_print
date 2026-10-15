"""Shared shapely helpers."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box

from electrode_slicer.geometry import (
    as_linestrings,
    as_polygons,
    chain_paths,
    connect_along_boundary,
    extend_ends,
    fit_arcs,
    hatch,
    order_paths,
    region_contours,
    ring_lines,
    simplify_paths,
    tidy_area,
    toolpath_area,
)


def test_flattening_digs_through_nested_collections():
    nested = GeometryCollection([
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        box(0, 0, 1, 1),
        Point(5, 5),
    ])
    assert len(as_linestrings(nested)) == 2
    assert len(as_polygons(nested)) == 1


def test_flattening_drops_zero_length_and_zero_area_pieces():
    assert as_linestrings(LineString([(0, 0), (0, 0)])) == []
    assert as_polygons(box(0, 0, 0, 0)) == []
    assert as_linestrings(None) == []


def test_hatch_spacing_and_clipping(square):
    lines = hatch(square, spacing=1.0, angle_deg=0.0)
    assert len(lines) == pytest.approx(10, abs=2)
    for line in lines:
        assert line.length == pytest.approx(10.0, abs=1e-6)
        assert square.buffer(1e-9).contains(line)


def test_hatch_angle_rotates_the_family(square):
    flat = hatch(square, 1.0, 0.0)
    diagonal = hatch(square, 1.0, 45.0)
    # A 45-degree hatch of a square runs corner to corner, so lines vary in
    # length where an axis-aligned one does not.
    assert len({round(ln.length, 6) for ln in flat}) == 1
    assert len({round(ln.length, 6) for ln in diagonal}) > 1


def test_hatch_refuses_absurd_spacing_rather_than_hanging(square):
    assert hatch(square, 0.0, 0.0) == []
    assert hatch(square, -1.0, 0.0) == []
    assert hatch(square, 1e-6, 0.0) == []      # would be millions of lines


def test_ring_lines_returns_holes_as_well_as_outlines():
    donut = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    rings = ring_lines([donut])
    assert len(rings) == 2
    assert all(r.is_ring for r in rings)


def test_toolpath_area_is_length_times_width_for_a_straight_bead():
    line = LineString([(0, 0), (10, 0)])
    area = toolpath_area([line], 0.4).area
    # A round-capped stroke: the rectangle plus the two end caps.
    assert area == pytest.approx(10 * 0.4 + 3.14159 * 0.2**2, rel=0.01)


def test_toolpath_area_of_nothing_is_empty():
    assert toolpath_area([], 0.4).is_empty


def test_only_clipped_ends_are_extended(square):
    """Interior lattice ends must stay put or every node becomes a fake crossing."""
    slice_geom = square.buffer(1.0)                  # the wall zone around the infill
    clipped = LineString([(0, 5), (10, 5)])          # both ends on the region boundary
    interior = LineString([(3, 3), (6, 6)])          # a free-floating lattice end
    out = extend_ends([clipped, interior], 0.5, square, slice_geom)
    assert out[0].length == pytest.approx(clipped.length + 1.0)
    assert out[1].length == pytest.approx(interior.length)


def test_extension_stays_inside_the_cross_section(square):
    """Beads may run into the walls but never off the edge of the part."""
    line = LineString([(0, 5), (10, 5)])
    out = extend_ends([line], 5.0, square, square.buffer(0.25))
    assert out[0].length == pytest.approx(10.5)


def test_rings_are_never_extended(square):
    ring = LineString(square.buffer(-1).exterior.coords)
    grown = extend_ends([ring], 0.5, square, square.buffer(1.0))
    assert grown[0].length == pytest.approx(ring.length)


def test_zero_extension_is_a_no_op(square):
    line = LineString([(0, 5), (10, 5)])
    assert extend_ends([line], 0.0, square)[0] is line


def test_simplify_thins_tessellated_curves_without_moving_them():
    """An STL circle arrives with hundreds of facets; the machine needs a few."""
    circle = LineString(Point(0, 0).buffer(2.5, quad_segs=360).exterior.coords)
    thin = simplify_paths([circle], 0.005)[0]
    assert len(thin.coords) < len(circle.coords) / 5
    assert thin.hausdorff_distance(circle) <= 0.005
    assert thin.length == pytest.approx(circle.length, rel=0.001)


def test_simplify_keeps_rings_closed():
    ring = LineString(Point(0, 0).buffer(1.0, quad_segs=64).exterior.coords)
    thin = simplify_paths([ring], 0.5)[0]
    assert thin.is_ring
    assert len(thin.coords) >= 4


def test_simplify_zero_is_a_no_op():
    line = LineString([(0, 0), (1, 0.0001), (2, 0)])
    assert simplify_paths([line], 0.0)[0].equals(line)


# --------------------------------------------------------------------------
# joining paths into continuous strokes
# --------------------------------------------------------------------------


def _segments(lines):
    """Every segment, direction ignored, to compare what gets drawn."""
    out = []
    for ln in lines:
        for a, b in zip(ln.coords, ln.coords[1:], strict=False):
            out.append(tuple(sorted((tuple(round(v, 6) for v in a),
                                     tuple(round(v, 6) for v in b)))))
    return sorted(out)


def test_paths_that_meet_end_to_end_become_one_stroke():
    pieces = [LineString([(0, 0), (1, 0)]), LineString([(2, 0), (1, 0)]),
              LineString([(2, 0), (2, 1)])]
    chained = chain_paths(pieces)
    assert len(chained) == 1
    assert _segments(chained) == _segments(pieces)


def test_a_loop_of_pieces_closes_into_a_ring():
    pieces = [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0, 1)]),
              LineString([(0, 1), (0, 0)])]
    (ring,) = chain_paths(pieces)
    assert ring.is_ring


def test_a_junction_needs_one_stroke_per_pair_of_odd_ends():
    # A T: three ends of degree one and a node of degree three.
    pieces = [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)]),
              LineString([(1, 0), (1, 1)])]
    chained = chain_paths(pieces)
    assert len(chained) == 2
    assert _segments(chained) == _segments(pieces)


def test_paths_that_do_not_touch_are_left_apart():
    pieces = [LineString([(0, 0), (1, 0)]), LineString([(1.01, 0), (2, 0)])]
    assert len(chain_paths(pieces)) == 2


def test_chaining_draws_a_honeycomb_in_fewer_strokes_without_changing_it(electrode, printer):
    from electrode_slicer.slicer import Slicer
    slicer = Slicer(printer)
    plan = slicer.layer_plan(electrode.height)
    zs = [z for z, _ in plan]
    layer = slicer.build(slicer.sections(electrode.mesh, zs), zs, "honeycomb", 0.4)[0]
    chained = chain_paths(layer.all_paths)
    assert len(chained) < len(layer.all_paths) / 2
    assert _segments(chained) == _segments(layer.all_paths)


def test_ordering_starts_each_path_near_where_the_last_one_ended():
    far, near = LineString([(10, 0), (11, 0)]), LineString([(2, 0), (1.5, 0)])
    ordered = order_paths([LineString([(0, 0), (1, 0)]), far, near])
    assert list(ordered[1].coords) == [(1.5, 0), (2, 0)]      # reversed to start close by
    assert ordered[2].equals(far)


def _stitched(contour):
    """A contour's pieces joined into one closed ring of points."""
    pts = [contour[0][1][0]]
    for _, piece in contour:
        assert piece[0] == pts[-1]          # each piece starts where the last ended
        pts += piece[1:]
    assert pts[0] == pts[-1]
    return pts


def _signed_area(pts):
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False))


def test_a_region_with_holes_is_one_contour_whose_bridges_enclose_nothing():
    area = box(0, 0, 10, 4).difference(Point(2, 2).buffer(1)).difference(Point(6, 2).buffer(1))
    (contour,) = region_contours(area)
    assert sum(kind == "bridge" for kind, _ in contour) == 4
    assert _signed_area(_stitched(contour)) == pytest.approx(area.area)


def test_a_cut_in_bridge_retraces_itself_exactly():
    ring = Point(0, 0).buffer(4).difference(Point(0, 0).buffer(3.5))
    bridges = [p for kind, p in region_contours(ring)[0] if kind == "bridge"]
    assert bridges[1] == bridges[0][::-1]


def test_an_island_in_a_hole_is_its_own_contour():
    frame = box(0, 0, 10, 4).difference(box(1, 1, 9, 3))
    contours = region_contours(frame.union(box(2, 1.5, 8, 2.5)))
    assert len(contours) == 2
    assert sum(_signed_area(_stitched(c)) for c in contours) == pytest.approx(24.0 + 6.0)


def test_tidying_drops_slivers_and_notches_finer_than_the_feature():
    band = box(0, 0, 4, 1)
    sliver = box(4, 0.45, 6, 0.47)               # 20 um wide spur
    notch = box(2, 0.9, 2.02, 1)                  # 20 um wide slot
    tidy = tidy_area(band.union(sliver).difference(notch), 0.05)
    assert tidy.bounds[2] == pytest.approx(4, abs=0.05)   # the spur is gone
    assert tidy.area == pytest.approx(4, rel=1e-3)


def test_tidying_keeps_a_band_one_bead_wide():
    ring = Point(0, 0).buffer(4).difference(Point(0, 0).buffer(3.9))
    assert tidy_area(ring, 0.05).area == pytest.approx(ring.area, rel=1e-2)


def test_straightening_thins_the_lines_but_keeps_the_arcs():
    # A half circle of fine chords, then a wobbly straight tail.
    arc = [(math.cos(math.pi * i / 64), math.sin(math.pi * i / 64)) for i in range(65)]
    tail = [(-1 - 0.01 * i, 0.002 * (i % 2)) for i in range(1, 41)]
    moves = fit_arcs(arc + tail, 0.005, straighten=0.025)
    assert [m[0] for m in moves].count("arc") >= 1
    assert sum(m[0] == "line" for m in moves) == 1
    assert moves[-1][1] == tail[-1]


def test_pieces_a_clip_cut_apart_are_rejoined_along_the_edge():
    # A zigzag clipped by a disc comes apart wherever it runs outside.
    disc = Point(0, 0).buffer(1, quad_segs=32)
    zigzag = LineString([(-2, y) if i % 2 else (2, y)
                         for i, y in enumerate([-0.8 + 0.2 * k for k in range(9)])])
    pieces = as_linestrings(disc.intersection(zigzag))
    assert len(pieces) > 4
    joined = connect_along_boundary(pieces, disc, along=zigzag)
    assert len(joined) == 1
    # Nothing is lost, and the connectors only add length along the edge.
    assert joined[0].length > sum(p.length for p in pieces)
    assert disc.buffer(1e-6).covers(joined[0])


def test_rejoined_strokes_neither_cross_nor_touch_one_another():
    from electrode_slicer.patterns import FillContext, get_pattern

    disc = Point(4, 4).buffer(3.5, quad_segs=64)
    ctx = FillContext(region=disc, scale=0.23, angle_deg=45.0, z=0.05, line_width=0.1)
    pattern = get_pattern("hilbert")
    paths = pattern.generate(ctx)
    clipped = as_linestrings(disc.intersection(LineString(
        [p for ln in paths for p in ln.coords])))   # a stand-in count of raw pieces
    assert len(paths) <= 8 < len(clipped)
    for i, a in enumerate(paths):
        assert a.is_simple
        for b in paths[i + 1:]:
            assert not a.intersects(b)


def test_joins_respect_the_longest_allowed_gap():
    disc = Point(0, 0).buffer(1, quad_segs=32)
    pieces = [LineString([(-1, 0), (1, 0)]), LineString([(0, -1), (0, -0.5)])]
    assert len(connect_along_boundary(pieces, disc, max_gap=0.1)) == 2
