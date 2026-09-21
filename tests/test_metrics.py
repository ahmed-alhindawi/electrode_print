"""Overlap measurement -- the numbers the hypothesis is judged on."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

from electrode_slicer.geometry import hatch
from electrode_slicer.metrics import analyse_overlap


def test_parallel_lines_never_cross(square):
    lines = hatch(square, spacing=1.0, angle_deg=0.0)
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.crossings == 0
    assert stats.crossing_excess_volume_mm3 == 0.0


def test_a_single_x_is_one_crossing_of_two_beads(square):
    lines = [LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)])]
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.crossings == 1
    assert stats.max_multiplicity == 2
    # Two orthogonal 0.2 mm beads share a 0.2 x 0.2 mm square.
    assert stats.overlap_area_mm2 == pytest.approx(0.04, rel=0.02)
    assert stats.hump_height_mm == pytest.approx(0.1)


def test_three_beads_through_one_point_stack_higher(square):
    lines = [
        LineString([(0, 5), (10, 5)]),
        LineString([(5, 0), (5, 10)]),
        LineString([(0, 0), (10, 10)]),
    ]
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.crossings == 1
    assert stats.max_multiplicity == 3
    assert stats.mean_multiplicity == pytest.approx(3.0)
    assert stats.hump_height_mm == pytest.approx(0.2)


def test_beads_meeting_end_to_end_are_junctions_not_crossings(square):
    """A honeycomb node is three bead ends, not a bead driven over a bead."""
    lines = [
        LineString([(5, 5), (5, 10)]),
        LineString([(5, 5), (0, 0)]),
        LineString([(5, 5), (10, 0)]),
    ]
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.crossings == 0
    assert stats.junctions == 1


def test_two_beads_merely_touching_end_to_end_are_neither(square):
    lines = [LineString([(0, 5), (5, 5)]), LineString([(5, 5), (10, 5)])]
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.crossings == 0
    assert stats.junctions == 0


def test_crossing_count_scales_with_grid_density(square):
    from electrode_slicer.patterns import FillContext, get_pattern

    def grid(scale):
        return get_pattern("grid").generate(
            FillContext(region=square, scale=scale, line_width=0.2)
        )

    coarse, fine = grid(2.0), grid(1.0)
    n_coarse = analyse_overlap(coarse, coarse, 0.2, 0.1, square.area).crossings
    n_fine = analyse_overlap(fine, fine, 0.2, 0.1, square.area).crossings
    # Halving the spacing quadruples the intersections of a square lattice.
    assert n_fine == pytest.approx(4 * n_coarse, rel=0.25)


def test_crossings_per_mm2_is_normalised_by_the_region(square):
    lines = [LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)])]
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.crossings_per_mm2 == pytest.approx(1 / 100.0)


def test_disconnected_ink_is_counted_as_islands(square):
    lines = [LineString([(1, 1), (2, 1)]), LineString([(8, 8), (9, 8)])]
    assert analyse_overlap([], lines, 0.2, 0.1, square.area).islands == 2
    joined = [LineString([(1, 1), (9, 1)]), LineString([(5, 1), (5, 9)])]
    assert analyse_overlap([], joined, 0.2, 0.1, square.area).islands == 1


def test_the_connected_share_says_how_much_of_the_ink_works(square):
    """Two islands can mean 1% stranded or 50%; the count cannot tell them apart."""
    even = [LineString([(1, 1), (5, 1)]), LineString([(1, 8), (5, 8)])]
    assert analyse_overlap([], even, 0.2, 0.1, square.area).connected_fraction == \
        pytest.approx(0.5, abs=0.02)

    speck = [LineString([(1, 1), (9, 1)]), LineString([(4, 8), (4.1, 8)])]
    stats = analyse_overlap([], speck, 0.2, 0.1, square.area)
    assert stats.islands == 2
    assert stats.connected_fraction > 0.95, "a speck alongside a long bead"


def test_joined_ink_is_wholly_connected(square):
    joined = [LineString([(1, 1), (9, 1)]), LineString([(5, 1), (5, 9)])]
    assert analyse_overlap([], joined, 0.2, 0.1, square.area).connected_fraction == 1.0


def test_an_empty_layer_is_trivially_connected():
    assert analyse_overlap([], [], 0.2, 0.1, 100.0).connected_fraction == 1.0


def test_parallel_overlap_is_reported_but_is_not_a_crossing(square):
    """Beads running alongside each other pile up without any hump."""
    lines = [LineString([(0, 5), (10, 5)]), LineString([(0, 5.1), (10, 5.1)])]
    stats = analyse_overlap(lines, lines, 0.4, 0.1, square.area)
    assert stats.crossings == 0
    # Two 0.4 mm beads 0.1 mm apart share 0.3 mm of width over 10 mm, against
    # 2 x 4 mm2 dispensed: 3/8 of everything laid down is doubled up.
    assert stats.overlap_fraction == pytest.approx(0.375, rel=0.05)
    assert stats.crossing_excess_volume_mm3 == 0.0


def test_crossing_excess_is_measured_only_around_the_crossings(square):
    """It must not swallow the overlap that parallel neighbours produce."""
    crossed = [LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)])]
    # Off to one side, clear of the vertical bead, so they only ever run
    # alongside each other and never cross anything.
    beside = [LineString([(6, 1), (10, 1)]), LineString([(6, 1.15), (10, 1.15)])]
    stats = analyse_overlap(crossed + beside, crossed + beside, 0.3, 0.1, square.area)
    assert stats.crossings == 1
    # The whole-layer figure sees both; the crossing figure sees only the X.
    assert stats.excess_volume_mm3 > 3 * stats.crossing_excess_volume_mm3
    assert stats.crossing_excess_volume_mm3 > 0


def test_an_empty_layer_measures_as_nothing():
    stats = analyse_overlap([], [], 0.2, 0.1, 100.0)
    assert stats.crossings == 0
    assert stats.overlap_area_mm2 == 0.0
    assert stats.as_dict()["islands"] == 1


def test_stats_round_trip_to_a_dict_for_the_manifest(square):
    lines = [LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)])]
    data = analyse_overlap(lines, lines, 0.2, 0.1, square.area).as_dict()
    assert data["crossings"] == 1
    assert set(data) >= {"islands", "junctions", "hump_height_mm", "overlap_fraction"}
