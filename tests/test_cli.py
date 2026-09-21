"""The command line front end, driven end to end."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from electrode_slicer.cli import build_parser, main, parse_density, parse_plate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slice_electrode.py"
SVG_NS = "{http://www.w3.org/2000/svg}"


def run(*args, expect=0):
    """Run the script the way a user would, in its own process."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == expect, proc.stderr
    return proc


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def test_plate_is_given_as_width_by_depth_by_height():
    assert parse_plate("220x300x40") == (220.0, 300.0, 40.0)
    assert parse_plate("220X300X40") == (220.0, 300.0, 40.0)


@pytest.mark.parametrize("bad", ["220x300", "a x b x c", "220x300x0", "-1x1x1"])
def test_a_malformed_plate_is_rejected(bad):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_plate(bad)


def test_densities_are_percentages():
    assert parse_density("40") == pytest.approx(0.4)
    assert parse_density("100") == pytest.approx(1.0)
    assert parse_density("1") == pytest.approx(0.01), "1 means one percent, not 100%"


@pytest.mark.parametrize("bad", ["101", "-5", "half"])
def test_an_out_of_range_density_is_rejected(bad):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_density(bad)


def test_the_defaults_are_the_documented_sweep():
    args = build_parser().parse_args([])
    assert args.infill_density == [0.20, 0.40, 0.60, 0.80]
    assert args.layout == "matrix"
    assert args.reference == 1
    assert args.svg_mode == "stroke"


# --------------------------------------------------------------------------
# running it
# --------------------------------------------------------------------------


def test_list_patterns_explains_the_crossing_axis(capsys):
    assert main(["--list-patterns"]) == 0
    out = capsys.readouterr().out
    assert "crossing" in out and "junction" in out
    for name in ("lines", "grid", "stars", "gyroid", "honeycomb"):
        assert name in out


def test_a_sweep_produces_a_valid_svg_and_manifest(stl_path, tmp_path):
    out = tmp_path / "sweep.svg"
    run(stl_path, "-o", out, "--infill-pattern", "lines", "grid",
        "--infill-density", "20", "40", "--quiet")

    root = ET.fromstring(out.read_text())
    assert root.get("width") == "220mm"
    assert len([g for g in root.findall(f"{SVG_NS}g") if g.get("id", "").startswith("layer_")]) == 3

    manifest = json.loads(out.with_suffix(".json").read_text())
    assert len(manifest["variants"]) == 5          # 2 patterns x 2 densities + reference
    assert manifest["printer"]["line_width_mm"] == 0.1
    assert manifest["model"]["extents_mm"][0] == pytest.approx(6.5, abs=0.01)

    by_id = {v["id"]: v for v in manifest["variants"]}
    assert by_id["lines_40pct"]["crossings_total"] == 0
    assert by_id["grid_40pct"]["crossings_total"] > 0
    for variant in manifest["variants"]:
        assert variant["max_islands"] == 1, "every electrode must print as one conductor"
        for layer in variant["layers"]:
            assert layer["density_actual_pct"] == pytest.approx(
                layer["density_target_pct"], abs=1.5
            )


def test_the_svg_stroke_is_the_100_micron_bead(stl_path, tmp_path):
    out = tmp_path / "bead.svg"
    run(stl_path, "-o", out, "--infill-pattern", "grid", "--infill-density", "40", "--quiet")
    root = ET.fromstring(out.read_text())
    widths = {p.get("stroke-width") for p in root.iter(f"{SVG_NS}path") if p.get("stroke")}
    assert widths == {"0.1"}


def test_the_bead_width_can_be_overridden(stl_path, tmp_path):
    out = tmp_path / "wide.svg"
    run(stl_path, "-o", out, "--line-width", "0.3", "--infill-pattern", "grid",
        "--infill-density", "40", "--quiet")
    root = ET.fromstring(out.read_text())
    widths = {p.get("stroke-width") for p in root.iter(f"{SVG_NS}path") if p.get("stroke")}
    assert widths == {"0.3"}


def test_the_summary_reports_the_crossing_axis(stl_path, tmp_path):
    proc = run(stl_path, "-o", tmp_path / "s.svg", "--infill-pattern", "lines", "stars",
               "--infill-density", "40")
    assert "X/mm2" in proc.stderr and "isl" in proc.stderr
    assert "crossing" in proc.stderr and "none" in proc.stderr


def test_a_dense_sweep_warns_about_uneven_ink(stl_path, tmp_path):
    proc = run(stl_path, "-o", tmp_path / "s.svg", "--infill-pattern", "lines", "stars",
               "--infill-density", "80")
    assert "ink used varies" in proc.stderr


def test_outline_mode_warns_that_it_hides_the_crossings(stl_path, tmp_path):
    proc = run(stl_path, "-o", tmp_path / "o.svg", "--svg-mode", "outline",
               "--infill-pattern", "grid", "--infill-density", "40")
    assert "discards" in proc.stderr


def test_split_layers_writes_one_file_per_layer(stl_path, tmp_path):
    out = tmp_path / "split.svg"
    run(stl_path, "-o", out, "--split-layers", "--infill-pattern", "lines",
        "--infill-density", "40", "--quiet")
    parts = sorted(p.name for p in tmp_path.glob("split_layer*.svg"))
    assert parts == ["split_layer000.svg", "split_layer001.svg", "split_layer002.svg"]


def test_the_matrix_puts_patterns_down_and_densities_across(stl_path, tmp_path):
    out = tmp_path / "m.svg"
    run(stl_path, "-o", out, "--infill-pattern", "lines", "grid", "stars",
        "--infill-density", "20", "40", "60", "80", "--quiet")
    manifest = json.loads(out.with_suffix(".json").read_text())
    cells = {v["id"]: tuple(v["cell"]) for v in manifest["variants"]}
    # One row per pattern...
    for pattern in ("lines", "grid", "stars"):
        assert len({cells[k][1] for k in cells if k.startswith(pattern + "_")}) == 1
    # ...one column per density, in ascending order.
    for pattern in ("lines", "grid", "stars"):
        cols = [cells[f"{pattern}_{d}pct"][0] for d in (20, 40, 60, 80)]
        assert cols == sorted(cols) == [0, 1, 2, 3]
    # ...and the reference on a row of its own.
    ref_row = cells["reference_solid_100pct"][1]
    assert ref_row not in {cells[k][1] for k in cells if not k.startswith("reference")}


def test_solid_layers_are_judged_against_their_own_target(stl_path, tmp_path):
    """--solid-layers overrides the skin; that is not a calibration failure."""
    out = tmp_path / "skin.svg"
    proc = run(stl_path, "-o", out, "--infill-pattern", "grid",
               "--infill-density", "30", "--solid-layers", "1", "--reference", "0")
    assert "cannot resolve" not in proc.stderr

    layers = json.loads(out.with_suffix(".json").read_text())["variants"][0]["layers"]
    assert layers[0]["density_actual_pct"] > 90, "first layer should be solid"
    assert layers[-1]["density_actual_pct"] > 90, "last layer should be solid"
    assert layers[1]["density_actual_pct"] == pytest.approx(30, abs=1.5)
    assert layers[1]["crossings"] > 0
    assert layers[-1]["crossings"] == 0, "a solid skin leaves no humps"


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_a_missing_model_fails_cleanly(tmp_path):
    proc = run(tmp_path / "nope.stl", "-o", tmp_path / "x.svg", expect=1)
    assert "no such file" in proc.stderr


def test_a_part_scaled_below_the_bead_width_is_refused(stl_path, tmp_path):
    proc = run(stl_path, "-o", tmp_path / "x.svg", "--input-units", "mm", expect=1)
    assert "too small" in proc.stderr or "thinner" in proc.stderr


def test_a_sweep_too_big_for_the_plate_is_refused(stl_path, tmp_path):
    proc = run(stl_path, "-o", tmp_path / "x.svg", "--scale", "12",
               "--infill-pattern", "lines", "grid", "triangles", "stars",
               "--infill-density", "10", "20", "30", "40", "50", "60", expect=1)
    assert "plate" in proc.stderr


def test_an_unknown_printer_is_refused(stl_path, tmp_path):
    proc = run(stl_path, "-o", tmp_path / "x.svg", "--printer", "makerbot", expect=2)
    assert "invalid choice" in proc.stderr
