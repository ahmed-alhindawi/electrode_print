"""Layers, walls and density calibration."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from electrode_slicer.patterns import get_pattern
from electrode_slicer.slicer import SliceConfig, Slicer, coverage, layer_heights

PATTERNS = ["lines", "grid", "triangles", "stars", "honeycomb", "gyroid",
            "serpentine", "hilbert"]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [{"perimeters": -1}, {"solid_layers": -1}, {"slice_at": "middle"},
     {"density_metric": "volume"}],
)
def test_nonsense_settings_are_refused(bad):
    with pytest.raises(ValueError):
        SliceConfig(**bad)


# --------------------------------------------------------------------------
# layer heights
# --------------------------------------------------------------------------


def test_layer_heights_tolerate_bounding_box_float_noise():
    """A 0.3 mm part is three 0.1 mm layers, not three plus a sliver."""
    assert len(layer_heights(0.30000001424, 0.1, 0.1)) == 3
    assert len(layer_heights(0.3, 0.1, 0.1)) == 3
    assert len(layer_heights(0.35, 0.1, 0.1)) == 4


def test_first_layer_height_is_honoured():
    zs = layer_heights(0.5, 0.1, 0.2, "bottom")
    assert zs[0][1] == pytest.approx(0.2)
    assert len(zs) == 4


def test_slice_at_positions_sit_inside_their_layer():
    for where in ("mid", "bottom", "top"):
        for z, top in layer_heights(0.5, 0.1, 0.1, where):
            assert 0.0 <= z <= top


def test_slice_at_mid_takes_the_middle_of_each_layer():
    assert [round(z, 4) for z, _ in layer_heights(0.3, 0.1, 0.1, "mid")] == [0.05, 0.15, 0.25]


def test_a_part_thinner_than_one_layer_still_gets_one():
    """Better a single thin layer than an empty plate."""
    plan = layer_heights(0.02, 0.1, 0.1)
    assert len(plan) == 1
    assert plan[0][1] == pytest.approx(0.02), "the layer is clipped to the part"


def test_a_part_of_no_height_gets_no_layers():
    assert layer_heights(0.0, 0.1, 0.1) == []


# --------------------------------------------------------------------------
# walls
# --------------------------------------------------------------------------


def test_walls_leave_room_for_infill(printer):
    slicer = Slicer(printer.with_overrides(line_width=0.4), SliceConfig(perimeters=2))
    loops, region = slicer.walls(box(0, 0, 10, 10))
    assert len(loops) == 2
    # Two 0.4 mm walls eat 0.4 mm off each side of the square.
    assert region.bounds[0] == pytest.approx(0.8, abs=0.02)


def test_walls_on_a_shape_too_thin_for_them(printer):
    slicer = Slicer(printer.with_overrides(line_width=0.4), SliceConfig(perimeters=4))
    loops, region = slicer.walls(box(0, 0, 10, 0.3))
    assert region.is_empty
    assert len(loops) <= 1


def test_zero_perimeters_hands_the_whole_section_to_the_infill(printer):
    slicer = Slicer(printer, SliceConfig(perimeters=0))
    loops, region = slicer.walls(box(0, 0, 10, 10))
    assert loops == []
    assert region.area == pytest.approx(100.0)


# --------------------------------------------------------------------------
# density calibration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("target", [0.15, 0.35, 0.60])
def test_calibration_hits_the_requested_density(name, target, printer, square):
    """Cross-pattern comparison is only fair if the density means one thing."""
    slicer = Slicer(printer.with_overrides(line_width=0.4), SliceConfig())
    scale, achieved = slicer.solve_scale(get_pattern(name), square, target, 0.0, 0.0)
    assert scale > 0
    assert achieved == pytest.approx(target, abs=0.01)


@pytest.mark.parametrize("name", PATTERNS)
def test_every_pattern_saturates_at_the_floor_it_declares(name, printer):
    """The solver refuses to probe below ``min_scale``, so that floor has to
    be a value at which the pattern is genuinely solid.  Set it too high and
    dense infill silently becomes unreachable.
    """
    from shapely.geometry import Point

    slicer = Slicer(printer, SliceConfig())
    pattern = get_pattern(name)
    region = Point(0, 0).buffer(2.0, quad_segs=64)
    floor = pattern.min_scale(printer.line_width)
    covered = slicer.measure(pattern, region, floor, 0.0, 0.05)
    # Anything this dense routes to the solid fill in practice, so
    # "essentially solid" is the bar that matters.
    assert covered > 0.95, f"{name} is only {covered:.1%} covered at its floor of {floor}"


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("target", [0.80, 0.95])
def test_the_densest_useful_infill_is_reachable(name, target, printer, square):
    """The saturation floor must not put a requestable density out of reach."""
    slicer = Slicer(printer, SliceConfig())
    _, achieved = slicer.solve_scale(get_pattern(name), square, target, 0.0, 0.05)
    assert achieved == pytest.approx(target, abs=0.015)


def test_coverage_by_area_and_by_length_agree_when_nothing_crosses(square, printer):
    from electrode_slicer.geometry import hatch
    lines = hatch(square, 1.0, 0.0)
    by_area = coverage(lines, square, 0.25, "area")
    by_length = coverage(lines, square, 0.25, "length")
    assert by_area == pytest.approx(by_length, rel=0.05)


def test_length_metric_over_reads_a_crossing_lattice(square):
    from electrode_slicer.patterns import FillContext
    lines = get_pattern("grid").generate(FillContext(region=square, scale=0.8, line_width=0.4))
    assert coverage(lines, square, 0.4, "length") > coverage(lines, square, 0.4, "area")


def test_coverage_of_nothing_is_zero(square):
    assert coverage([], square, 0.25) == 0.0


# --------------------------------------------------------------------------
# building layers
# --------------------------------------------------------------------------


def test_line_infill_alternates_direction_between_layers(printer):
    slicer = Slicer(printer, SliceConfig(infill_angle=45.0, alternate_angles=True))
    lines = get_pattern("lines")
    assert slicer.infill_angle(lines, 0) == 45.0
    assert slicer.infill_angle(lines, 1) == 135.0


def test_lattices_and_tpms_are_not_rotated_between_layers(printer):
    slicer = Slicer(printer, SliceConfig(infill_angle=45.0, alternate_angles=True))
    for name in ("grid", "triangles", "stars", "gyroid", "honeycomb"):
        pattern = get_pattern(name)
        assert slicer.infill_angle(pattern, 1) == 45.0, name


def test_alternation_can_be_switched_off(printer):
    slicer = Slicer(printer, SliceConfig(alternate_angles=False, infill_angle=45.0))
    assert slicer.infill_angle(get_pattern("lines"), 1) == 45.0


def test_build_layer_produces_walls_and_measured_infill(printer, square):
    slicer = Slicer(printer, SliceConfig(perimeters=2))
    layer = slicer.build_layer(0, 0.05, square, "grid", 0.4)
    assert layer.perimeters
    assert layer.infill
    assert layer.pattern == "grid"
    assert layer.density_actual == pytest.approx(0.4, abs=0.02)
    assert layer.overlap.crossings > 0


def test_full_density_is_routed_through_the_solid_fill(printer, square):
    slicer = Slicer(printer, SliceConfig())
    layer = slicer.build_layer(0, 0.05, square, "grid", 1.0)
    # At 100% every lattice degenerates to beads laid edge to edge.
    assert layer.density_actual > 0.95
    assert layer.overlap.crossings == 0


def test_an_empty_cross_section_produces_an_empty_layer(printer):
    from shapely.geometry import Polygon
    layer = Slicer(printer).build_layer(0, 0.05, Polygon(), "grid", 0.4)
    assert layer.all_paths == []
    assert layer.length == 0.0


def test_zero_density_leaves_the_walls_but_no_infill(printer, square):
    layer = Slicer(printer).build_layer(0, 0.05, square, "grid", 0.0)
    assert layer.perimeters
    assert layer.infill == []


def test_repeated_layers_reuse_a_solved_scale(printer, square, monkeypatch):
    """A thin flat part repeats one cross section; solving it once is enough."""
    slicer = Slicer(printer, SliceConfig())
    calls = []
    original = slicer.solve_scale
    monkeypatch.setattr(
        slicer, "solve_scale",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    for index in range(4):
        slicer.build_layer(index, 0.05, square, "grid", 0.4)
    assert len(calls) == 1


def test_build_walks_every_layer(printer, square):
    slicer = Slicer(printer)
    zs = [0.05, 0.15, 0.25]
    layers = slicer.build([square] * 3, zs, "lines", 0.3)
    assert [lp.index for lp in layers] == [0, 1, 2]
    assert [lp.z for lp in layers] == zs


def test_solid_layers_override_the_skin_only(printer, square):
    slicer = Slicer(printer, SliceConfig(solid_layers=1))
    layers = slicer.build([square] * 4, [0.05, 0.15, 0.25, 0.35], "grid", 0.3)
    assert [lp.pattern for lp in layers] == ["solid", "grid", "grid", "solid"]
    assert layers[0].density_target == 1.0
    assert layers[1].density_target == pytest.approx(0.3)
