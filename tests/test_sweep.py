"""The pipeline that the CLI and the web front end share."""

from __future__ import annotations

import pytest

from electrode_slicer import SweepRequest, run_sweep
from electrode_slicer.geometry import as_polygons
from electrode_slicer.io import SvgStyle, load_mesh
from electrode_slicer.layout import LayoutConfig
from electrode_slicer.slicer import Modifier, SliceConfig

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


def test_asking_for_flexibility_is_not_served_from_a_cache_without_it(electrode, printer):
    """The analysis is attached to the layer, so a layer built without it
    cannot stand in for one built with it."""
    scales, layers = {}, {}
    plain = sweep(electrode, printer, patterns=("grid",), densities=(0.4,), reference=0,
                  scale_cache=scales, layer_cache=layers)
    assert not plain.layers_by_variant[0][-1].flex.solved
    analysed = sweep(electrode, printer, patterns=("grid",), densities=(0.4,), reference=0,
                     config=SliceConfig(flexibility=True),
                     scale_cache=scales, layer_cache=layers)
    assert analysed.layers_by_variant[0][-1].flex.solved


def test_the_stitch_ratio_is_part_of_the_cache_key(electrode, printer):
    scales, layers = {}, {}
    tight = sweep(electrode, printer, patterns=("stitched",), densities=(0.6,), reference=0,
                  config=SliceConfig(stitch_ratio=2.0),
                  scale_cache=scales, layer_cache=layers)
    loose = sweep(electrode, printer, patterns=("stitched",), densities=(0.6,), reference=0,
                  config=SliceConfig(stitch_ratio=16.0),
                  scale_cache=scales, layer_cache=layers)
    assert tight.svg() != loose.svg()
    tight_x = sum(lp.overlap.crossings for lp in tight.layers_by_variant[0])
    loose_x = sum(lp.overlap.crossings for lp in loose.layers_by_variant[0])
    assert tight_x > loose_x


def test_presentation_changes_reuse_the_built_geometry(electrode, printer):
    scales, layers = {}, {}
    sweep(electrode, printer, scale_cache=scales, layer_cache=layers)
    before = len(layers)
    sweep(electrode, printer, style=SvgStyle(mode="outline"),
          layout_config=LayoutConfig(mode="pack"),
          scale_cache=scales, layer_cache=layers)
    assert len(layers) == before, "nothing geometric changed, so nothing was rebuilt"


# --------------------------------------------------------------------------
# Gerber export
# --------------------------------------------------------------------------


def test_every_layer_becomes_one_gerber_file(electrode, printer):
    result = sweep(electrode, printer)
    files = result.gerbers()
    assert len(files) == result.n_layers
    for text in files:
        assert "%FSLAX46Y46*%" in text and "%MOMM*%" in text
        assert "%TF.FileFunction,Copper,L1,Top*%" in text
        assert text.rstrip().endswith("M02*")


def test_the_gerber_aperture_is_the_bead(electrode, printer):
    text = sweep(electrode, printer).gerbers()[0]
    assert f"%ADD10C,{printer.line_width:g}*%" in text


def test_gerber_coordinates_stay_on_the_plate(electrode, printer):
    import re
    text = sweep(electrode, printer).gerbers()[0]
    points = [(int(x) / 1e6, int(y) / 1e6)
              for x, y in re.findall(r"X(-?\d+)Y(-?\d+)D0[12]\*", text)]
    assert points
    assert all(0 <= x <= printer.plate_x and 0 <= y <= printer.plate_y for x, y in points)


def test_a_filename_cannot_break_the_gerber_syntax(electrode, printer):
    text = sweep(electrode, printer).gerbers(title="bad*name%")[0]
    assert text.splitlines()[0].count("*") == 1
    assert "%" not in text.splitlines()[0]


def _drawn(text):
    """Strokes and total drawn length of a Gerber layer, arcs included."""
    import math
    import re
    strokes, drawn, here, mode = 0, 0.0, None, "G01"
    for line in text.splitlines():
        if line in ("G01*", "G02*", "G03*"):
            mode = line[:3]
            continue
        m = re.fullmatch(r"X(-?\d+)Y(-?\d+)(?:I(-?\d+)J(-?\d+))?D0([12])\*", line)
        if not m:
            continue
        point = (int(m[1]) / 1e6, int(m[2]) / 1e6)
        if m[5] == "2":
            strokes += 1
        elif mode == "G01":
            drawn += math.dist(here, point)
        else:
            centre = (here[0] + int(m[3]) / 1e6, here[1] + int(m[4]) / 1e6)
            a0 = math.atan2(here[1] - centre[1], here[0] - centre[0])
            a1 = math.atan2(point[1] - centre[1], point[0] - centre[0])
            sweep = (a1 - a0) % (2 * math.pi)
            if mode == "G02":
                sweep = 2 * math.pi - sweep
            drawn += sweep * math.dist(here, centre)
        here = point
    return strokes, drawn


def test_the_gerber_joins_fragments_but_draws_the_same_length(electrode, printer):
    result = sweep(electrode, printer, patterns=("honeycomb",), densities=(0.4,), reference=0,
                   layout_config=LayoutConfig(fiducials=False))
    layer = result.layers_by_variant[0][0]
    strokes, drawn = _drawn(result.gerbers()[0])
    assert strokes < len(layer.all_paths) / 2
    # Arcs follow the true curve the chords cut across, so a touch longer.
    assert drawn == pytest.approx(layer.length, rel=2e-3)


def test_a_perimeter_is_written_as_a_few_arcs(electrode, printer):
    result = sweep(electrode, printer, patterns=("lines",), densities=(0.4,), reference=0,
                   layout_config=LayoutConfig(fiducials=False))
    text = result.gerbers()[0]
    ring = result.layers_by_variant[0][0].perimeters[0]
    arcs = sum(1 for line in text.splitlines() if "I" in line and line.endswith("D01*"))
    assert "G75*" in text
    assert 2 <= arcs < len(ring.coords) / 5
    assert _drawn(text)[1] == pytest.approx(result.layers_by_variant[0][0].length, rel=2e-3)


def test_the_walls_pass_is_the_area_to_fill_not_the_strokes(electrode, printer):
    # A modifier smaller than the part leaves a solid band whose hatch ends,
    # drawn as strokes, gave the NOVA a sawtooth edge and pinholes to fill
    # round.  The walls pass is the area itself, edge clean.
    w, d, h = electrode.extents
    cut = Modifier("box", (w / 2, d / 2, h / 2), (w / 2, d / 2, 2 * h))
    result = sweep(electrode, printer, patterns=("lines",), densities=(0.4,), reference=0,
                   config=SliceConfig(modifiers=(cut,)),
                   layout_config=LayoutConfig(fiducials=False))
    layer = result.layers_by_variant[0][0]
    files = result.gerber_passes()
    walls = files[0][3].splitlines()
    assert files[0][2] == "walls"
    assert walls.count("G36*") == walls.count("G37*") == len(as_polygons(layer.walls_area))
    # Nothing is drawn outside a region: no stroke starts after the last G37.
    last = max(i for i, line in enumerate(walls) if line == "G37*")
    assert not any(line.endswith("D02*") for line in walls[last:])
    # The band covers the slice less what the pattern fills.
    assert layer.walls_area.area == pytest.approx(layer.slice_area - layer.region_area, rel=1e-6)
    # Written as plainly as a copper pour: arcs where the outline is a circle,
    # and not every facet of the mesh anywhere else.
    start = walls.index("G36*")
    assert any("I" in line for line in walls[start:])
    drawn = sum(line.endswith("D01*") for line in walls[start:])
    facets = sum(len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
                 for p in as_polygons(layer.walls_area))
    assert drawn < facets / 2
    # The infill passes still draw strokes, and no regions.
    assert all("G36*" not in text for _, _, role, text in files if role == "infill")


def test_a_solid_pad_hands_over_its_whole_outline(electrode, printer):
    result = sweep(electrode, printer, patterns=("lines",), densities=(0.4,), reference=1)
    pad = next(per[0] for var, per in zip(result.variants, result.layers_by_variant, strict=True)
               if var.is_reference)
    assert pad.walls_area.area == pytest.approx(pad.slice_area)
