"""Single-stroke fills: serpentine and Hilbert.

These replaced concentric, which was crossing-free -- and so looked like a
good control for the hump experiment -- but whose nested rings never touch,
leaving most of its ink electrically isolated.  These two are crossing-free
*and* drawn as one continuous stroke, so nothing can be stranded.  The tests
below pin both halves of that claim.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Point, box

from electrode_slicer import SweepRequest, run_sweep
from electrode_slicer.geometry import robust_union, toolpath_area
from electrode_slicer.patterns import FillContext, get_pattern
from electrode_slicer.patterns.curves import _rounded

DISC = Point(0, 0).buffer(2.0, quad_segs=64)
CURVES = ["serpentine", "hilbert"]
SINGLE_STROKE = [*CURVES, "stitched"]


def fill(name, region=DISC, scale=0.3, angle=0.0, line_width=0.1):
    return get_pattern(name).generate(
        FillContext(region=region, scale=scale, angle_deg=angle, line_width=line_width)
    )


# --------------------------------------------------------------------------
# one continuous stroke
# --------------------------------------------------------------------------


def test_a_serpentine_of_a_convex_region_is_a_single_path():
    """The whole point: rows joined end to end rather than left loose."""
    assert len(fill("serpentine", box(0, 0, 10, 10), scale=1.0)) == 1


def test_a_serpentine_covers_every_row_it_would_have_hatched():
    rows = get_pattern("lines").generate(
        FillContext(region=box(0, 0, 10, 10), scale=1.0, line_width=0.1))
    snake = fill("serpentine", box(0, 0, 10, 10), scale=1.0)
    straight = sum(ln.length for ln in rows)
    snaked = sum(ln.length for ln in snake)
    # The connectors add length; the rows themselves are not lost.
    assert snaked > straight


def test_a_serpentine_breaks_rather_than_printing_across_thin_air():
    """Two lobes joined by nothing: a connector between them would be a
    bead laid over the gap, off the part entirely."""
    dumbbell = Point(-4, 0).buffer(2, quad_segs=32).union(Point(4, 0).buffer(2, quad_segs=32))
    paths = fill("serpentine", dumbbell, scale=0.8)
    assert len(paths) > 1
    padded = dumbbell.buffer(1e-6)
    assert all(padded.contains(p) for p in paths)


@pytest.mark.parametrize("name", CURVES)
def test_neither_curve_ever_crosses_itself(name):
    for scale in (0.15, 0.3, 0.6, 1.2):
        lines = fill(name, scale=scale)
        assert all(ln.is_simple for ln in lines), f"{name} at {scale}"


# --------------------------------------------------------------------------
# rounded turns
# --------------------------------------------------------------------------


def test_rounding_removes_square_corners():
    square_turn = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    rounded = _rounded(square_turn, radius=1.0)
    assert len(rounded) > len(square_turn)
    assert (10.0, 0.0) not in rounded, "the sharp vertex itself is cut away"

    def sharpest(points):
        worst = 180.0
        for a, b, c in zip(points, points[1:], points[2:], strict=False):
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            worst = min(worst, 180.0 - math.degrees(math.acos(cos)))
        return worst

    assert sharpest(square_turn) == pytest.approx(90.0, abs=1)
    assert sharpest(rounded) > 120.0, "no corner sharper than a gentle bend remains"


def test_rounding_never_eats_more_than_half_a_segment():
    """A huge radius on a short segment would fold the path back on itself."""
    tight = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert LineString(_rounded(tight, radius=50.0)).is_simple


def test_rounding_leaves_short_and_straight_paths_alone():
    assert _rounded([(0.0, 0.0), (1.0, 1.0)], 1.0) == [(0.0, 0.0), (1.0, 1.0)]
    assert _rounded([(0.0, 0.0)], 1.0) == [(0.0, 0.0)]
    straight = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert LineString(_rounded(straight, 0.5)).length == pytest.approx(2.0)


# --------------------------------------------------------------------------
# the Hilbert construction
# --------------------------------------------------------------------------


def test_hilbert_density_is_continuous_not_quantised():
    """The curve only exists on a 2^n grid.  Driving it by cell size rather
    than by order is what keeps density from jumping in factors of four."""
    lengths = [
        sum(ln.length for ln in fill("hilbert", scale=cell))
        for cell in (0.30, 0.28, 0.26, 0.24, 0.22, 0.20)
    ]
    for shorter, longer in zip(lengths, lengths[1:], strict=False):
        assert longer > shorter, "a finer cell must lay more bead"
        assert longer / shorter < 1.6, "and must not jump by a whole order"


def test_hilbert_refuses_a_cell_too_fine_to_build():
    """4^order points; a tiny cell on a big part would exhaust memory."""
    assert fill("hilbert", box(0, 0, 400, 400), scale=0.05) == []


@pytest.mark.parametrize("name", CURVES)
def test_both_curves_fill_the_region_they_are_given(name):
    lines = fill(name, scale=0.25)
    covered = toolpath_area(lines, 0.1).intersection(DISC).area / DISC.area
    assert covered > 0.25, f"{name} covered only {covered:.0%}"


# --------------------------------------------------------------------------
# what they were added for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", CURVES)
@pytest.mark.parametrize("density", [0.2, 0.4, 0.6, 0.8])
def test_neither_curve_strands_any_ink_on_the_real_electrode(
    name, density, electrode, printer
):
    result = run_sweep(electrode, printer,
                       SweepRequest(patterns=[name], densities=[density], reference=0))
    for layer in result.layers_by_variant[0]:
        assert layer.overlap.islands == 1, f"{name} at {density:.0%}"
        assert layer.overlap.connected_fraction == pytest.approx(1.0)
    assert not any("blobs of ink" in w for w in result.warnings)


def test_no_registered_pattern_strands_ink(electrode, printer):
    """The bar concentric failed, applied to everything that is left.

    A pattern that isolates part of its own ink is not usable as an
    electrode at all, so this is a gate on the whole set rather than a
    property of these two.
    """
    from electrode_slicer.patterns import all_patterns

    names = [p.name for p in all_patterns() if p.name != "none"]
    result = run_sweep(electrode, printer,
                       SweepRequest(patterns=names, densities=[0.2, 0.8], reference=0))
    for var in result.variants:
        dead = 1.0 - min(lp.overlap.connected_fraction
                         for lp in result.layers_by_variant[var.index])
        assert dead == pytest.approx(0.0), f"{var.ident} strands {dead:.0%} of its ink"
    assert not any("blobs of ink" in w for w in result.warnings)


@pytest.mark.parametrize("name", CURVES)
def test_both_stay_crossing_free_controls(name, electrode, printer):
    """A handful of grazes from the rounded turns is fine; a lattice's worth
    of deliberate crossings is not."""
    result = run_sweep(electrode, printer,
                       SweepRequest(patterns=[name, "grid"], densities=[0.6], reference=0))
    counts = {p: sum(x.overlap.crossings for x in layers)
              for (p, _), layers in result.unique().items()}
    assert counts[name] < 0.1 * counts["grid"]
    assert get_pattern(name).overlap_class == "none"


# --------------------------------------------------------------------------
# stitched: the compromise between humps and bending
# --------------------------------------------------------------------------


def test_stitched_is_a_serpentine_that_crosses_itself():
    """Its whole reason to exist: the base stays a single soft stroke, the
    stitches supply the humps."""
    plain = fill("serpentine", scale=0.3)
    stitched = fill("stitched", scale=0.3)
    from electrode_slicer.metrics import analyse_overlap

    assert analyse_overlap(plain, plain, 0.1, 0.1, DISC.area).crossings == 0
    assert analyse_overlap(stitched, stitched, 0.1, 0.1, DISC.area).crossings > 0
    assert sum(ln.length for ln in stitched) > sum(ln.length for ln in plain)


def test_the_stitch_ratio_trades_humps_against_everything_else():
    from electrode_slicer.metrics import analyse_overlap

    def humps(ratio):
        lines = get_pattern("stitched").generate(
            FillContext(region=DISC, scale=0.3, line_width=0.1, stitch_ratio=ratio))
        return analyse_overlap(lines, lines, 0.1, 0.1, DISC.area).crossings

    tight, loose = humps(2.0), humps(16.0)
    assert tight > loose, "closer stitches mean more crossings"
    assert loose > 0, "even the sparsest setting still stitches"


def test_the_ratio_falls_back_to_the_pattern_default():
    from electrode_slicer.metrics import analyse_overlap

    def humps(**kw):
        lines = get_pattern("stitched").generate(
            FillContext(region=DISC, scale=0.3, line_width=0.1, **kw))
        return analyse_overlap(lines, lines, 0.1, 0.1, DISC.area).crossings

    assert humps() == humps(stitch_ratio=None)
    assert humps() == humps(stitch_ratio=get_pattern("stitched").ratio)


def test_a_nonsense_ratio_cannot_put_stitches_closer_than_the_rows():
    lines = get_pattern("stitched").generate(
        FillContext(region=DISC, scale=0.3, line_width=0.1, stitch_ratio=0.0))
    assert lines


def test_stitched_sits_between_the_soft_and_the_braced(electrode, printer):
    """The compromise, asserted end to end: more humps than a plain
    serpentine, more give than a full lattice."""
    from electrode_slicer.slicer import SliceConfig

    def measure(pattern, ratio=None):
        result = run_sweep(
            electrode, printer,
            SweepRequest(patterns=[pattern], densities=[0.6], reference=0),
            SliceConfig(flexibility=True, stitch_ratio=ratio),
        )
        top = result.layers_by_variant[0][-1]
        return top.overlap.crossings_per_mm2, top.flex.bend_compliance

    soft_x, soft_bend = measure("serpentine")
    mid_x, mid_bend = measure("stitched", 8.0)
    hard_x, hard_bend = measure("stars")

    assert soft_x < mid_x < hard_x, "humps rise from serpentine to stitched to stars"
    assert hard_bend < mid_bend < soft_bend, "and compliance falls the other way"


def test_stitched_strands_no_ink_either(electrode, printer):
    result = run_sweep(electrode, printer,
                       SweepRequest(patterns=["stitched"], densities=[0.2, 0.6], reference=0))
    for var in result.variants:
        for layer in result.layers_by_variant[var.index]:
            assert layer.overlap.connected_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the overlay robustness they exposed
# --------------------------------------------------------------------------


def test_robust_union_survives_a_geos_topology_failure():
    """A dense Hilbert fill defeated GEOS' exact predicates outright, and it
    can happen to any pattern whose beads run very nearly tangent."""
    region = Point(0, 0).buffer(2.0, quad_segs=64)
    lines = get_pattern("hilbert").generate(
        FillContext(region=region, scale=0.1235, line_width=0.1))
    merged = toolpath_area(lines, 0.1)
    assert not merged.is_empty
    assert merged.is_valid


def test_robust_union_handles_nothing_and_empties():
    assert robust_union([]).is_empty
    assert robust_union([None, Point(0, 0).buffer(0)]).is_empty
    assert robust_union([box(0, 0, 1, 1), box(2, 2, 3, 3)]).area == pytest.approx(2.0)
