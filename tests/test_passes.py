"""Splitting a layer into passes a region slicer cannot misread."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, box

from electrode_slicer.passes import (
    join_straightest,
    pass_artwork,
    predict_fill,
    print_passes,
    separate_touching,
)
from electrode_slicer.slicer import LayerPaths

W = 0.1


def touching(group, width=W):
    return [(a, b) for i, a in enumerate(group) for b in group[i + 1:]
            if a.distance(b) < width]


def test_crossing_lines_go_to_different_passes():
    lines = [LineString([(0, 1), (2, 1)]), LineString([(1, 0), (1, 2)])]
    groups = separate_touching(lines, W, 0.01)
    assert len(groups) == 2


def test_lines_that_never_touch_share_a_pass():
    lines = [LineString([(0, y), (2, y)]) for y in (0, 0.5, 1.0)]
    assert len(separate_touching(lines, W, 0.01)) == 1


def test_strips_that_would_merge_are_kept_apart():
    # Centrelines exactly one bead apart: the strips meet edge to edge.
    lines = [LineString([(0, 0), (2, 0)]), LineString([(0, W), (2, W)])]
    assert len(separate_touching(lines, W, 0.01)) == 2


def test_no_pass_ever_holds_two_touching_strips():
    grid = [LineString([(x, 0), (x, 3)]) for x in (0.5, 1, 1.5, 2, 2.5)]
    grid += [LineString([(0, y), (3, y)]) for y in (0.5, 1, 1.5, 2, 2.5)]
    for group in separate_touching(grid, W, 0.01):
        assert not touching(group)


def test_a_line_split_at_its_nodes_is_joined_back_straight():
    pieces = [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)]),
              LineString([(1, 0), (1, 1)])]      # a branch off the middle
    runs = join_straightest(pieces)
    assert sorted(round(r.length, 6) for r in runs) == [1.0, 2.0]


def test_a_sharp_turn_is_not_joined():
    pieces = [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0.2, 0.3)])]
    assert len(join_straightest(pieces, max_turn_deg=75)) == 2


def test_a_closed_loop_of_pieces_is_walked_once():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    pieces = [LineString(square[i:i + 2]) for i in range(4)]
    runs = join_straightest(pieces, max_turn_deg=95)
    assert len(runs) == 1 and runs[0].length == pytest.approx(4)


def layer(**kw):
    return LayerPaths(index=0, z=0.05, **kw)


def test_walls_come_first_and_carry_the_solid_fill():
    ring = LineString([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    solid = [LineString([(3.2, y), (4, y)]) for y in (1, 1.1)]
    infill = [LineString([(0, 1.5), (3, 1.5)]), LineString([(1.5, 0), (1.5, 3)])]
    passes = print_passes(layer(perimeters=[ring], solid=solid, infill=infill), W)
    assert passes[0] == ("walls", [ring, *solid])
    assert [role for role, _ in passes] == ["walls", "infill", "infill"]


def test_a_solid_pad_is_left_whole_for_the_slicer_to_fill():
    hatch = [LineString([(0, y), (3, y)]) for y in (0, W, 2 * W)]
    passes = print_passes(layer(infill=hatch, pattern="solid"), W)
    assert passes == [("walls", hatch)]


def test_every_path_lands_in_exactly_one_pass():
    ring = LineString([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    infill = [LineString([(0, y), (3, y)]) for y in (1, 2)]
    infill += [LineString([(x, 0), (x, 3)]) for x in (1, 2)]
    passes = print_passes(layer(perimeters=[ring], infill=infill), W)
    drawn = sum(ln.length for _, group in passes for ln in group)
    assert drawn == pytest.approx(ring.length + sum(ln.length for ln in infill))


def test_a_strip_one_tip_wide_is_one_bead_down_its_middle():
    strip = LineString([(0, 0), (3, 0)]).buffer(W / 2, cap_style="flat")
    fill = predict_fill(strip, W)
    assert fill.single == [True]
    assert fill.length == pytest.approx(3, rel=0.05)
    assert fill.gap_fraction < 0.01


def test_a_wide_area_is_filled_in_loops_a_tip_apart():
    fill = predict_fill(box(0, 0, 1, 1), W)
    assert len(fill.loops) == 5                 # 1 mm across, 0.1 mm a loop each side
    assert fill.gap_fraction < 0.02


def test_a_tip_wider_than_a_strip_cannot_print_it():
    strip = LineString([(0, 0), (3, 0)]).buffer(W / 2, cap_style="flat")
    fill = predict_fill(strip, 1.5 * W)
    assert not fill.loops
    assert fill.gap_fraction == pytest.approx(1.0, abs=0.01)


def test_the_artwork_matches_the_passes_the_gerber_writes():
    ring = LineString([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    infill = [LineString([(0, 1.5), (3, 1.5)]), LineString([(1.5, 0), (1.5, 3)])]
    lp = layer(perimeters=[ring], infill=infill,
               walls_area=box(-0.1, -0.1, 3.1, 3.1).difference(box(0.1, 0.1, 2.9, 2.9)))
    art = pass_artwork(lp, W)
    assert [role for role, _ in art] == [role for role, _ in print_passes(lp, W)]
    assert art[0][1].area == pytest.approx(lp.walls_area.area, rel=0.01)
    assert art[1][1].area == pytest.approx(3 * W, rel=0.05)
