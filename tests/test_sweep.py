"""The pipeline that the CLI and the web front end share."""

from __future__ import annotations

import pytest

from electrode_slicer import SweepRequest, run_sweep
from electrode_slicer.io import SvgStyle, load_mesh
from electrode_slicer.layout import LayoutConfig
from electrode_slicer.slicer import SliceConfig

pytestmark = pytest.mark.usefixtures("electrode")


def sweep(electrode, printer, patterns=("lines", "grid"), densities=(0.2, 0.6), **kw):
    return run_sweep(
        electrode, printer,
        SweepRequest(patterns=list(patterns), densities=list(densities),
                     replicates=kw.pop("replicates", 1), reference=kw.pop("reference", 1)),
        kw.pop("config", None), kw.pop("layout_config", None), kw.pop("style", None),
        **kw,
    )


# --------------------------------------------------------------------------
# the request
# --------------------------------------------------------------------------


def test_an_unknown_pattern_is_caught_before_any_work_happens():
    with pytest.raises(KeyError, match="unknown infill pattern"):
        SweepRequest(patterns=["spirograph"])


@pytest.mark.parametrize("bad", [{"patterns": []}, {"densities": []}, {"densities": [1.5]}])
def test_an_empty_or_impossible_request_is_refused(bad):
    with pytest.raises(ValueError):
        SweepRequest(**bad)


# --------------------------------------------------------------------------
# running it
# --------------------------------------------------------------------------


def test_a_sweep_covers_every_combination_plus_the_reference(electrode, printer):
    result = sweep(electrode, printer)
    assert len(result.variants) == 2 * 2 + 1
    assert result.n_layers == 3
    assert all(len(per) == 3 for per in result.layers_by_variant)


def test_replicates_share_their_toolpaths(electrode, printer):
    """Only the placement differs, so a copy must not cost a second slice."""
    result = sweep(electrode, printer, patterns=("lines",), densities=(0.4,),
                   replicates=3, reference=0)
    assert len(result.variants) == 3
    first = result.layers_by_variant[0]
    assert all(per is first for per in result.layers_by_variant)
    assert len({v.dx for v in result.variants}) == 3, "but they sit in different places"


def test_every_electrode_is_placed_on_the_plate(electrode, printer):
    result = sweep(electrode, printer)
    for var in result.variants:
        assert 0 <= var.dx <= printer.plate_x
        assert 0 <= var.dy <= printer.plate_y


def test_the_reference_is_fully_dense(electrode, printer):
    result = sweep(electrode, printer)
    ref = next(v for v in result.variants if v.is_reference)
    assert result.layers_by_variant[ref.index][-1].density_actual > 0.95


def test_crossings_separate_the_patterns_at_a_matched_density(electrode, printer):
    """The whole point of the sweep, asserted end to end."""
    result = sweep(electrode, printer, patterns=("lines", "grid", "stars"),
                   densities=(0.6,), reference=0)
    rows = {p: layers for (p, _), layers in result.unique().items()}
    crossings = {p: sum(lp.overlap.crossings for lp in layers) for p, layers in rows.items()}
    assert crossings["lines"] == 0
    assert 0 < crossings["grid"] < crossings["stars"]


def test_crossing_patterns_lay_more_ink_at_the_same_measured_density(electrode, printer):
    """A confound the experiment has to control for.

    Overlapping beads deposit material without adding coverage, so at matched
    density a crossing pattern walks further than a flat one.  The gap is
    small at moderate density and severe at high density, which is why the
    manifest reports path length and deposited volume per electrode.
    """
    def spread(density):
        result = sweep(electrode, printer, patterns=("lines", "grid", "stars"),
                       densities=(density,), reference=0)
        lengths = [sum(lp.length for lp in layers) for layers in result.unique().values()]
        return max(lengths) / min(lengths)

    assert spread(0.2) < 1.2, "at 20% the patterns are near enough matched on ink"
    assert spread(0.8) > spread(0.2), "and the gap opens up as the fill gets dense"


def test_every_variant_prints_as_one_connected_conductor(electrode, printer):
    result = sweep(electrode, printer, patterns=("lines", "grid", "stars", "honeycomb"),
                   densities=(0.2, 0.4, 0.6, 0.8))
    for var in result.variants:
        per = result.layers_by_variant[var.index]
        assert max(lp.overlap.islands for lp in per) == 1, var.ident


def test_connected_patterns_report_no_dead_ink(electrode, printer):
    result = sweep(electrode, printer, patterns=("lines", "grid", "gyroid", "stars"),
                   densities=(0.2, 0.4, 0.6, 0.8), reference=0)
    for var in result.variants:
        for layer in result.layers_by_variant[var.index]:
            assert layer.overlap.connected_fraction == pytest.approx(1.0), var.ident
    assert not any("blobs of ink" in w for w in result.warnings)


def test_solid_is_not_reported_as_a_failed_calibration(electrode, printer):
    """solid ignores density by design; complaining it 'cannot resolve 20%'
    is the tool misreading its own deliberate override."""
    result = sweep(electrode, printer, patterns=("solid",),
                   densities=(0.2, 0.4, 0.6, 0.8), reference=0)
    assert not any("cannot resolve" in w for w in result.warnings)
    notes = [w for w in result.warnings if "ignores the density sweep" in w]
    assert len(notes) == 1, "one note for the pattern, not one per density"
    assert "--reference" in notes[0]


def test_a_solid_layer_records_the_target_it_actually_used(electrode, printer):
    result = sweep(electrode, printer, patterns=("solid",), densities=(0.2,), reference=0)
    for layer in result.layers_by_variant[0]:
        assert layer.density_target == 1.0
        assert layer.pattern == "solid"


def test_duplicate_warnings_are_collapsed(electrode, printer):
    result = sweep(electrode, printer, patterns=("solid",),
                   densities=(0.2, 0.3, 0.4, 0.5, 0.6), reference=0)
    assert len(result.warnings) == len(set(result.warnings))


def test_an_unreachable_density_is_reported_rather_than_silently_missed(electrode, printer):
    """At a 0.25 mm bead, 10% on a 5 mm disc is one or two serpentine rows,
    and the connectors joining them push it past the target however wide the
    rows are spaced.  The tool has to say so rather than quietly overshoot."""
    result = sweep(electrode, printer, patterns=("serpentine",), densities=(0.10,),
                   reference=0)
    assert any("cannot resolve" in w for w in result.warnings)
    # Which way it misses depends on the layer -- the one carrying the tab
    # differs from the plain disc -- so only the size of the miss is pinned.
    worst = max(abs(lp.density_actual - 0.10)
                for lp in result.layers_by_variant[0] if lp.region_area > 0)
    assert worst > 0.01, "the manifest records what was achieved, not what was asked"


def test_a_part_taller_than_the_machine_is_refused(electrode, printer):
    with pytest.raises(ValueError, match="Z travel"):
        run_sweep(electrode, printer.with_overrides(z_travel=0.1), SweepRequest())


def test_a_part_too_small_for_the_bead_is_refused(stl_path, printer):
    tiny = load_mesh(stl_path, units="mm")       # metres read as millimetres
    with pytest.raises(ValueError, match="too small|thinner"):
        run_sweep(tiny, printer, SweepRequest(patterns=["lines"], densities=[0.4]))


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


def test_the_result_renders_itself_to_svg_and_json(electrode, printer):
    result = sweep(electrode, printer)
    svg = result.svg()
    assert svg.startswith("<?xml") and svg.rstrip().endswith("</svg>")
    manifest = result.manifest("model/electrode.stl")
    assert manifest["source"] == "model/electrode.stl"
    assert len(manifest["variants"]) == len(result.variants)


def test_unique_collapses_replicates_for_reporting(electrode, printer):
    result = sweep(electrode, printer, patterns=("lines",), densities=(0.4,),
                   replicates=4, reference=0)
    assert list(result.unique()) == [("lines", 0.4)]


def test_solid_layers_override_the_skin_across_the_whole_sweep(electrode, printer):
    result = sweep(electrode, printer, patterns=("grid",), densities=(0.3,),
                   reference=0, config=SliceConfig(solid_layers=1))
    per = result.layers_by_variant[0]
    assert [lp.pattern for lp in per] == ["solid", "grid", "solid"]
    assert not any("cannot resolve" in w for w in result.warnings), \
        "a deliberate solid skin is not a calibration failure"


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------


def test_caches_are_reusable_and_change_nothing(electrode, printer):
    """The interactive front end keeps them between requests."""
    scales, layers = {}, {}
    first = sweep(electrode, printer, scale_cache=scales, layer_cache=layers)
    assert scales and layers
    second = sweep(electrode, printer, scale_cache=scales, layer_cache=layers)
    assert first.svg() == second.svg()


def test_a_changed_bead_width_is_not_served_from_the_cache(electrode, printer):
    """A stale cache here would quietly print the wrong geometry."""
    scales, layers = {}, {}
    coarse = sweep(electrode, printer.with_overrides(line_width=0.25),
                   scale_cache=scales, layer_cache=layers)
    fine = sweep(electrode, printer.with_overrides(line_width=0.1),
                 scale_cache=scales, layer_cache=layers)
    assert coarse.svg() != fine.svg()
    assert coarse.total_path_mm < fine.total_path_mm, "a finer bead walks further"


def test_a_changed_setting_is_not_served_from_the_cache(electrode, printer):
    scales, layers = {}, {}
    two = sweep(electrode, printer, config=SliceConfig(perimeters=2),
                scale_cache=scales, layer_cache=layers)
    four = sweep(electrode, printer, config=SliceConfig(perimeters=4),
                 scale_cache=scales, layer_cache=layers)
    assert two.svg() != four.svg()


def test_presentation_changes_reuse_the_built_geometry(electrode, printer):
    scales, layers = {}, {}
    sweep(electrode, printer, scale_cache=scales, layer_cache=layers)
    before = len(layers)
    sweep(electrode, printer, style=SvgStyle(mode="outline"),
          layout_config=LayoutConfig(mode="pack"),
          scale_cache=scales, layer_cache=layers)
    assert len(layers) == before, "nothing geometric changed, so nothing was rebuilt"
