"""Shared shapely helpers."""

from __future__ import annotations

import pytest
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box

from electrode_slicer.geometry import (
    as_linestrings,
    as_polygons,
    extend_ends,
    hatch,
    ring_lines,
    simplify_paths,
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
