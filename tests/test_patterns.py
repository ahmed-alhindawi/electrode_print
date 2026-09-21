"""The abstract infill pattern, the registry, and every implementation.

These are the tests that keep a pattern comparison honest: each pattern must
produce the structure it claims, and must respond monotonically to its scale
knob so the slicer can calibrate it.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from electrode_slicer.metrics import analyse_overlap
from electrode_slicer.patterns import (
    CROSSING,
    JUNCTION,
    NONE,
    FillContext,
    InfillPattern,
    all_patterns,
    get_pattern,
    pattern_names,
    register,
)
from electrode_slicer.patterns.base import _REGISTRY
from electrode_slicer.patterns.hatch import HatchPattern
from electrode_slicer.patterns.tpms import TpmsPattern

FILLING = [n for n in pattern_names() if n != "none"]
# Hilbert is left out: its curve only exists on a 2^n grid, so coverage has a
# sub-tolerance wobble as the order steps.  tests/test_curves.py checks the
# property that actually matters there -- that density stays continuous.
LATTICES = ["lines", "grid", "triangles", "stars", "honeycomb", "gyroid",
            "serpentine"]


def fill(name, region, scale, angle=0.0, z=0.0, line_width=0.25):
    return get_pattern(name).generate(
        FillContext(region=region, scale=scale, angle_deg=angle, z=z, line_width=line_width)
    )


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def test_every_registered_pattern_declares_its_contract():
    for pattern in all_patterns():
        assert pattern.name
        assert pattern.overlap_class in (NONE, JUNCTION, CROSSING)
        assert pattern.description, f"{pattern.name} needs a description for --list-patterns"
        assert isinstance(pattern, InfillPattern)


def test_registry_lookup_and_a_helpful_error():
    assert get_pattern("grid").name == "grid"
    with pytest.raises(KeyError, match="unknown infill pattern"):
        get_pattern("spirograph")


def test_all_patterns_is_ordered_by_how_much_it_crosses_itself():
    classes = [p.overlap_class for p in all_patterns()]
    rank = {NONE: 0, JUNCTION: 1, CROSSING: 2}
    assert [rank[c] for c in classes] == sorted(rank[c] for c in classes)


def test_a_duplicate_name_is_refused():
    class Clone(InfillPattern):
        name = "grid"
        description = "x"

        def generate(self, ctx):
            return []

    with pytest.raises(ValueError, match="duplicate"):
        register(Clone())


def test_a_nameless_or_mislabelled_pattern_is_refused():
    class Nameless(InfillPattern):
        description = "x"

        def generate(self, ctx):
            return []

    class BadClass(InfillPattern):
        name = "bad-class"
        overlap_class = "sometimes"
        description = "x"

        def generate(self, ctx):
            return []

    with pytest.raises(ValueError, match="no name"):
        register(Nameless())
    with pytest.raises(ValueError, match="overlap_class"):
        register(BadClass())
    assert "bad-class" not in _REGISTRY


def test_a_pattern_cannot_be_instantiated_without_generate():
    class Incomplete(InfillPattern):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_a_third_party_pattern_plugs_in_without_touching_the_core():
    class Dots(HatchPattern):
        name = "test-dots"
        overlap_class = CROSSING
        description = "a test pattern registered from outside the package"
        angles = (0.0, 90.0)

    try:
        register(Dots())
        assert get_pattern("test-dots").families == 2
        assert "test-dots" in pattern_names()
    finally:
        _REGISTRY.pop("test-dots", None)


# --------------------------------------------------------------------------
# behaviour shared by every pattern
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", FILLING)
def test_output_is_clipped_to_the_region(name, square):
    lines = fill(name, square, 1.2)
    assert lines, f"{name} produced nothing at a workable scale"
    padded = square.buffer(1e-6)
    assert all(padded.contains(ln) for ln in lines)


@pytest.mark.parametrize("name", LATTICES)
def test_coverage_falls_as_the_scale_knob_grows(name, square):
    """The one rule the slicer's calibration depends on."""
    lengths = [
        sum(ln.length for ln in fill(name, square, scale))
        for scale in (0.8, 1.2, 1.8, 2.7)
    ]
    assert lengths == sorted(lengths, reverse=True), f"{name} is not monotonic: {lengths}"


@pytest.mark.parametrize("name", FILLING)
def test_degenerate_inputs_give_nothing_rather_than_raising(name, square):
    assert fill(name, square, 0.0) == []
    assert fill(name, square, -1.0) == []
    assert fill(name, Polygon(), 1.0) == []


def test_the_none_pattern_fills_nothing(square):
    assert fill("none", square, 1.0) == []


# --------------------------------------------------------------------------
# the crossing axis -- what the electrode work turns on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("lines", NONE), ("serpentine", NONE), ("hilbert", NONE), ("gyroid", NONE),
     ("solid", NONE), ("honeycomb", JUNCTION),
     ("grid", CROSSING), ("triangles", CROSSING), ("stars", CROSSING)],
)
def test_patterns_are_labelled_with_the_right_overlap_class(name, expected):
    assert get_pattern(name).overlap_class == expected


@pytest.mark.parametrize("name", ["lines", "serpentine", "hilbert", "gyroid"])
def test_patterns_labelled_crossing_free_really_do_not_cross(name, square):
    lines = fill(name, square, 1.5)
    assert analyse_overlap(lines, lines, 0.25, 0.1, square.area).crossings == 0


@pytest.mark.parametrize(("name", "least"), [("grid", 10), ("triangles", 10), ("stars", 20)])
def test_patterns_labelled_crossing_really_do_cross(name, least, square):
    lines = fill(name, square, 1.5)
    assert analyse_overlap(lines, lines, 0.25, 0.1, square.area).crossings >= least


def test_more_line_families_means_more_crossings_at_one_spacing(square):
    counts = []
    for name in ("lines", "grid", "triangles", "stars"):
        lines = fill(name, square, 1.5)
        counts.append(analyse_overlap(lines, lines, 0.25, 0.1, square.area).crossings)
    assert counts == sorted(counts)
    assert counts[0] == 0


def test_honeycomb_nodes_are_junctions_not_crossings(square):
    lines = fill("honeycomb", square, 1.2)
    stats = analyse_overlap(lines, lines, 0.2, 0.1, square.area)
    assert stats.junctions > 0
    assert stats.crossings == 0


# --------------------------------------------------------------------------
# individual patterns
# --------------------------------------------------------------------------


def test_grid_lays_twice_the_bead_of_lines(square):
    one = sum(ln.length for ln in fill("lines", square, 1.0))
    two = sum(ln.length for ln in fill("grid", square, 1.0))
    assert two == pytest.approx(2 * one, rel=0.05)


def test_hatch_families_are_the_declared_angles():
    assert get_pattern("lines").families == 1
    assert get_pattern("grid").families == 2
    assert get_pattern("triangles").families == 3
    assert get_pattern("stars").families == 4


def test_honeycomb_dissolves_shared_walls(square):
    """Neighbouring cells share a wall; it must be traced once, not twice."""
    cells = fill("honeycomb", square, 1.5)
    assert cells
    doubled = sum(
        a.intersection(b).length
        for i, a in enumerate(cells)
        for b in cells[i + 1:]
    )
    assert doubled == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ["gyroid"])
def test_tpms_varies_through_z(name, square):
    """Successive layers must get different curves, not a repeated stencil.

    Total curve length is a poor witness: the gyroid field is symmetric in
    x and y, so some pairs of heights differ in shape at identical length.
    Compare the swept footprints instead.
    """
    from electrode_slicer.geometry import toolpath_area

    a = fill(name, square, 4.0, z=0.0)
    b = fill(name, square, 4.0, z=1.0)
    assert a and b
    foot_a, foot_b = toolpath_area(a, 0.2), toolpath_area(b, 0.2)
    assert foot_a.symmetric_difference(foot_b).area > 0.25 * foot_a.area


def test_tpms_patterns_declare_themselves_z_dependent():
    assert get_pattern("gyroid").z_dependent
    assert not get_pattern("grid").z_dependent


def test_a_new_tpms_surface_is_just_a_new_field(square):
    import numpy as np

    class Diamond(TpmsPattern):
        name = "test-diamond"
        description = "Schwarz-D, added in a test"

        def field(self, u, v, w):
            return np.sin(u) * np.sin(v) * np.sin(w) + np.sin(u) * np.cos(v) * np.cos(w)

    try:
        register(Diamond())
        assert fill("test-diamond", square, 3.0, z=0.4)
    finally:
        _REGISTRY.pop("test-diamond", None)


def test_solid_ignores_density_and_lays_beads_edge_to_edge():
    solid = get_pattern("solid")
    assert solid.initial_scale(0.2, 0.25) == pytest.approx(0.25)
    assert solid.initial_scale(0.9, 0.25) == pytest.approx(0.25)


def test_analytic_initial_scale_follows_the_family_formula():
    """k families of width w at spacing s cover k*w/s of the area."""
    assert get_pattern("lines").initial_scale(0.25, 0.5) == pytest.approx(2.0)
    assert get_pattern("grid").initial_scale(0.25, 0.5) == pytest.approx(4.0)
    assert get_pattern("stars").initial_scale(0.25, 0.5) == pytest.approx(8.0)


def test_honeycomb_initial_scale_uses_the_hex_cell_formula():
    import math
    expected = 2.0 * 0.5 / (math.sqrt(3.0) * 0.25)
    assert get_pattern("honeycomb").initial_scale(0.25, 0.5) == pytest.approx(expected)
