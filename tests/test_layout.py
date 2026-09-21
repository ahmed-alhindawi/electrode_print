"""Arranging a sweep on the build plate."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from electrode_slicer.layout import (
    LayoutConfig,
    Variant,
    assign_cells,
    build_variants,
    layout,
)

FOOTPRINT = (6.5, 5.0)


def boxes_for(variants, footprint=FOOTPRINT):
    return [box(v.dx, v.dy, v.dx + footprint[0], v.dy + footprint[1]) for v in variants]


# --------------------------------------------------------------------------
# building the sweep
# --------------------------------------------------------------------------


def test_the_sweep_is_the_product_of_patterns_and_densities():
    variants = build_variants(["lines", "grid"], [0.2, 0.4, 0.6], reference=0)
    assert len(variants) == 6
    assert {(v.pattern, v.density) for v in variants} == {
        ("lines", 0.2), ("lines", 0.4), ("lines", 0.6),
        ("grid", 0.2), ("grid", 0.4), ("grid", 0.6),
    }


def test_a_fully_dense_reference_is_added_by_default():
    variants = build_variants(["lines"], [0.4])
    refs = [v for v in variants if v.is_reference]
    assert len(refs) == 1
    assert refs[0].pattern == "solid"
    assert refs[0].density == 1.0


def test_the_reference_can_be_dropped_or_replicated():
    assert not any(v.is_reference for v in build_variants(["lines"], [0.4], reference=0))
    assert sum(v.is_reference for v in build_variants(["lines"], [0.4], reference=3)) == 3


def test_replicates_multiply_every_combination():
    variants = build_variants(["lines", "grid"], [0.2, 0.4], replicates=3, reference=0)
    assert len(variants) == 12
    assert sorted(v.replicate for v in variants)[:3] == [0, 0, 0]


def test_a_single_replicate_is_not_numbered():
    """Identifiers stay clean when there is nothing to disambiguate."""
    one = build_variants(["lines"], [0.4], replicates=1, reference=0)[0]
    many = build_variants(["lines"], [0.4], replicates=2, reference=0)
    assert one.ident == "lines_40pct"
    assert [v.ident for v in many] == ["lines_40pct_r0", "lines_40pct_r1"]


def test_reference_identifiers_are_distinguishable():
    ref = build_variants(["lines"], [0.4], reference=1)[-1]
    assert ref.ident.startswith("reference_")
    assert ref.label == "sol 100%"


def test_variants_expose_the_overlap_class_of_their_pattern():
    variants = build_variants(["lines", "grid"], [0.4], reference=0)
    assert variants[0].overlap_class == "none"
    assert variants[1].overlap_class == "crossing"


@pytest.mark.parametrize("bad", [{"replicates": 0}, {"reference": -1}])
def test_impossible_sweeps_are_refused(bad):
    with pytest.raises(ValueError):
        build_variants(["lines"], [0.4], **bad)


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


def test_matrix_gives_one_row_per_pattern_and_one_column_per_density():
    variants = build_variants(["lines", "grid", "stars"], [0.2, 0.4, 0.6, 0.8], reference=0)
    cols, rows = assign_cells(variants)
    assert (cols, rows) == (4, 3)
    by_pattern = {}
    for v in variants:
        by_pattern.setdefault(v.pattern, set()).add(v.row)
    assert all(len(r) == 1 for r in by_pattern.values()), "each pattern owns one row"
    assert {v.col for v in variants if v.density == 0.2} == {0}


def test_rows_follow_the_order_the_patterns_were_given():
    variants = build_variants(["stars", "lines", "grid"], [0.4], reference=0)
    assign_cells(variants)
    assert [v.pattern for v in sorted(variants, key=lambda v: v.row)] == \
        ["stars", "lines", "grid"]


def test_columns_are_sorted_by_density_whatever_order_they_arrive_in():
    variants = build_variants(["lines"], [0.8, 0.2, 0.6], reference=0)
    assign_cells(variants)
    ordered = sorted(variants, key=lambda v: v.col)
    assert [v.density for v in ordered] == [0.2, 0.6, 0.8]


def test_replicates_sit_together_inside_their_density_block():
    variants = build_variants(["lines"], [0.2, 0.4], replicates=3, reference=0)
    assign_cells(variants, replicates=3)
    first = sorted(v.col for v in variants if v.density == 0.2)
    second = sorted(v.col for v in variants if v.density == 0.4)
    assert first == [0, 1, 2]
    assert second == [3, 4, 5]


def test_the_reference_gets_a_row_of_its_own():
    variants = build_variants(["lines", "grid"], [0.4], reference=2)
    cols, rows = assign_cells(variants)
    ref_rows = {v.row for v in variants if v.is_reference}
    sweep_rows = {v.row for v in variants if not v.is_reference}
    assert len(ref_rows) == 1
    assert ref_rows.isdisjoint(sweep_rows)
    assert rows == len(sweep_rows) + 1


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_placed_parts_never_overlap_and_stay_on_the_plate(printer, layout_config):
    variants = build_variants(["lines", "grid", "stars"], [0.2, 0.4, 0.6, 0.8])
    layout(variants, FOOTPRINT, printer, layout_config)
    placed = boxes_for(variants)
    for i, a in enumerate(placed):
        minx, miny, maxx, maxy = a.bounds
        assert minx >= 0 and maxx <= printer.plate_x
        assert miny >= 0 and maxy <= printer.plate_y
        for b in placed[i + 1:]:
            assert a.intersection(b).area == 0


def test_the_first_pattern_reads_at_the_top_of_the_plate(printer, layout_config):
    """Row 0 is the first pattern given, and +Y is the back of the plate."""
    variants = build_variants(["lines", "grid", "stars"], [0.4], reference=0)
    layout(variants, FOOTPRINT, printer, layout_config)
    ordered = sorted(variants, key=lambda v: v.row)
    assert sorted((v.dy for v in ordered), reverse=True) == [v.dy for v in ordered]


def test_density_rises_left_to_right(printer, layout_config):
    variants = build_variants(["lines"], [0.2, 0.4, 0.8], reference=0)
    layout(variants, FOOTPRINT, printer, layout_config)
    ordered = sorted(variants, key=lambda v: v.density)
    assert sorted(v.dx for v in ordered) == [v.dx for v in ordered]


def test_the_block_is_centred_on_the_plate(printer, layout_config):
    variants = build_variants(["lines", "grid"], [0.2, 0.8], reference=0)
    layout(variants, FOOTPRINT, printer, layout_config)
    placed = boxes_for(variants)
    right = max(b.bounds[2] for b in placed)
    bottom = min(b.bounds[1] for b in placed)
    top = max(b.bounds[3] for b in placed)
    # The left gutter carries the row labels, so only Y is exactly centred.
    assert pytest.approx(bottom, rel=0.05) == (bottom + printer.plate_y - top) / 2
    assert right < printer.plate_x


def test_axis_labels_describe_the_grid(printer, layout_config):
    variants = build_variants(["lines", "grid"], [0.2, 0.8], reference=1)
    grid = layout(variants, FOOTPRINT, printer, layout_config)
    assert [text for _, text in grid.col_labels] == ["20%", "80%"]
    assert [text for _, text in grid.row_labels] == ["lines", "grid", "100% ref"]


def test_one_heading_per_density_block_not_per_replicate(printer, layout_config):
    variants = build_variants(["lines"], [0.2, 0.8], replicates=3, reference=0)
    grid = layout(variants, FOOTPRINT, printer, layout_config, replicates=3)
    assert [text for _, text in grid.col_labels] == ["20%", "80%"]


def test_labels_can_be_switched_off(printer):
    variants = build_variants(["lines"], [0.4], reference=0)
    grid = layout(variants, FOOTPRINT, printer, LayoutConfig(labels=False))
    assert grid.col_labels == []
    assert grid.row_labels == []


def test_pack_layout_fills_rows_in_the_order_given(printer):
    variants = build_variants(["lines", "grid"], [0.2, 0.4], reference=0)
    grid = layout(variants, FOOTPRINT, printer, LayoutConfig(mode="pack", columns=2))
    assert grid.cols == 2
    assert [(v.col, v.row) for v in variants] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_pack_layout_fits_as_many_as_it_can_when_not_told(printer):
    variants = build_variants(["lines"], [0.1 * i for i in range(1, 10)], reference=0)
    grid = layout(variants, FOOTPRINT, printer, LayoutConfig(mode="pack"))
    assert grid.rows == 1, "nine small parts fit across a 220 mm plate"


def test_a_sweep_too_wide_for_the_plate_is_refused(printer, layout_config):
    variants = build_variants(["lines"], [0.01 * i for i in range(1, 60)], reference=0)
    with pytest.raises(ValueError, match="columns"):
        layout(variants, FOOTPRINT, printer, layout_config)


def test_a_sweep_too_tall_for_the_plate_is_refused(printer):
    tall = [Variant(index=i, pattern="lines", density=0.4) for i in range(60)]
    with pytest.raises(ValueError, match="rows"):
        layout(tall, (60.0, 60.0), printer, LayoutConfig(mode="pack", columns=1))


def test_a_part_bigger_than_the_plate_is_refused(printer, layout_config):
    variants = build_variants(["lines"], [0.4], reference=0)
    with pytest.raises(ValueError, match="does not fit"):
        layout(variants, (500.0, 5.0), printer, layout_config)


@pytest.mark.parametrize("bad", [{"mode": "spiral"}, {"gap": -1}])
def test_nonsense_layout_settings_are_refused(bad):
    with pytest.raises(ValueError):
        LayoutConfig(**bad)
