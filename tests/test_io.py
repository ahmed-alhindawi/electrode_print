"""Reading STLs in, writing SVGs and manifests out."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import LineString, box

from electrode_slicer.io import (
    SvgStyle,
    build_manifest,
    build_svg,
    flipper,
    fmt,
    load_mesh,
    load_mesh_bytes,
    path_d,
    polygon_d,
    write_manifest,
    write_svg,
)
from electrode_slicer.layout import LayoutConfig, build_variants, layout
from electrode_slicer.slicer import SliceConfig, Slicer

SVG_NS = "{http://www.w3.org/2000/svg}"


# --------------------------------------------------------------------------
# reading models
# --------------------------------------------------------------------------


def test_units_are_guessed_from_the_model_size(stl_path):
    """The supplied STL is authored in metres; nothing in the file says so."""
    loaded = load_mesh(stl_path)
    assert loaded.units == "m"
    assert loaded.guessed
    assert math.isclose(loaded.width, 6.5, abs_tol=0.01)


def test_a_declared_unit_is_taken_at_its_word(stl_path):
    loaded = load_mesh(stl_path, units="mm")
    assert loaded.units == "mm"
    assert not loaded.guessed
    assert loaded.width < 0.01


def test_the_model_is_moved_to_the_origin_and_onto_the_plate(stl_path):
    loaded = load_mesh(stl_path)
    assert loaded.mesh.bounds[0] == pytest.approx([0, 0, 0], abs=1e-9)


def test_scale_is_applied_on_top_of_the_units(stl_path):
    assert load_mesh(stl_path, scale=2.0).width == pytest.approx(13.0, abs=0.02)


def test_rotation_swaps_the_footprint(stl_path):
    upright = load_mesh(stl_path)
    turned = load_mesh(stl_path, rotate_deg=90.0)
    assert turned.width == pytest.approx(upright.depth, abs=0.01)
    assert turned.depth == pytest.approx(upright.width, abs=0.01)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mesh(tmp_path / "nothing.stl")


def test_a_negative_scale_is_refused(stl_path):
    with pytest.raises(ValueError, match="scale"):
        load_mesh(stl_path, scale=-1.0)


def test_an_upload_loads_the_same_as_a_file(stl_path):
    from_disk = load_mesh(stl_path)
    from_bytes = load_mesh_bytes(stl_path.read_bytes())
    assert from_bytes.extents == pytest.approx(from_disk.extents)
    assert from_bytes.units == from_disk.units


def test_an_empty_upload_is_refused():
    with pytest.raises(ValueError, match="empty"):
        load_mesh_bytes(b"")


# --------------------------------------------------------------------------
# number and path formatting
# --------------------------------------------------------------------------


def test_number_formatting_is_compact_and_has_no_negative_zero():
    assert fmt(1.50000) == "1.5"
    assert fmt(-0.00001) == "0"
    assert fmt(12.0) == "12"
    assert fmt(0.0) == "0"


def test_path_d_closes_rings_without_repeating_the_first_point():
    d = path_d([(0, 0), (1, 0), (1, 1), (0, 0)], close=True)
    assert d.endswith("Z")
    assert d.count(",") == 3


def test_path_d_of_a_single_point_is_nothing():
    assert path_d([(0, 0)], close=False) == ""


def test_path_d_of_a_zero_length_ring_is_nothing():
    # Written as "M 0,0 L Z" it would stop a browser drawing the whole path.
    assert path_d([(1, 1), (1, 1)], close=True) == ""


def test_polygon_d_emits_holes_as_extra_subpaths():
    donut = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    assert polygon_d(donut).count("M ") == 2


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------


def test_plate_coordinates_are_flipped_into_svg_space():
    to_svg = flipper(300.0, flip=True)
    assert to_svg(LineString([(10, 0), (10, 300)])).coords[0] == (10.0, 300.0)


def test_the_flip_can_be_switched_off():
    assert flipper(300.0, flip=False)(LineString([(10, 0), (10, 5)])).coords[0] == (10.0, 0.0)


# --------------------------------------------------------------------------
# the SVG document
# --------------------------------------------------------------------------


@pytest.fixture
def plate(printer):
    """A tiny finished sweep, ready to render."""
    slicer = Slicer(printer, SliceConfig())
    square = box(0, 0, 4, 4)
    zs = [0.05, 0.15]
    variants = build_variants(["lines", "grid"], [0.4], reference=1)
    grid = layout(variants, (4.0, 4.0), printer, LayoutConfig())
    built = {}
    layers = []
    for v in variants:
        key = (v.pattern, v.density)
        built.setdefault(key, slicer.build([square, square], zs, v.pattern, v.density))
        layers.append(built[key])
    return printer, variants, layers, zs, grid


def test_the_document_is_millimetres_at_plate_size(plate):
    printer, variants, layers, zs, grid = plate
    svg = build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid)
    root = ET.fromstring(svg)
    assert root.get("width") == "220mm"
    assert root.get("height") == "300mm"
    assert root.get("viewBox") == "0 0 220 300"
    assert root.get("data-units") == "mm"


def test_each_print_layer_is_its_own_group(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid))
    ids = [g.get("id") for g in root.findall(f"{SVG_NS}g")]
    assert "layer_000" in ids and "layer_001" in ids
    assert "build-plate" in ids and "annotations" in ids


def test_a_single_layer_can_be_exported_alone(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(
        build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid, only_layer=1)
    )
    ids = [g.get("id") for g in root.findall(f"{SVG_NS}g")]
    assert "layer_001" in ids and "layer_000" not in ids


def test_every_electrode_carries_its_parameters_as_data_attributes(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid))
    groups = {g.get("id"): g for g in root.iter(f"{SVG_NS}g") if g.get("data-pattern")}
    assert "grid_40pct" in groups
    assert groups["grid_40pct"].get("data-overlap") == "crossing"
    assert int(groups["grid_40pct"].get("data-crossings")) > 0
    assert groups["lines_40pct"].get("data-overlap") == "none"


def test_stroke_width_is_the_bead_width(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid))
    widths = {p.get("stroke-width") for p in root.iter(f"{SVG_NS}path") if p.get("stroke")}
    assert widths == {fmt(printer.line_width)}


def test_a_100_micron_bead_is_written_as_a_0_1_mm_stroke(plate):
    printer, variants, layers, zs, grid = plate
    fine = printer.with_overrides(line_width=0.1)
    root = ET.fromstring(build_svg(fine, variants, layers, zs, SvgStyle(), 4.0, grid))
    widths = {p.get("stroke-width") for p in root.iter(f"{SVG_NS}path") if p.get("stroke")}
    assert widths == {"0.1"}


def test_outline_mode_fills_areas_instead_of_stroking_paths(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(
        build_svg(printer, variants, layers, zs, SvgStyle(mode="outline"), 4.0, grid)
    )
    paths = [p for p in root.iter(f"{SVG_NS}path") if p.get("class", "").endswith("outline")]
    assert paths
    assert all(p.get("fill") != "none" and p.get("stroke") == "none" for p in paths)


def test_both_mode_emits_each_role_twice(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(
        build_svg(printer, variants, layers, zs, SvgStyle(mode="both"), 4.0, grid)
    )
    classes = [p.get("class") for p in root.iter(f"{SVG_NS}path")]
    assert "infill" in classes and "infill-outline" in classes


def test_annotations_are_separable_from_the_toolpaths(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid))
    notes = next(g for g in root.findall(f"{SVG_NS}g") if g.get("id") == "annotations")
    labels = [t.text for t in notes.iter(f"{SVG_NS}text")]
    assert "lines" in labels and "40%" in labels
    # The headings name each electrode, so none carries a caption of its own.
    assert not any(t.startswith("grd") for t in labels)


def test_a_packed_plate_captions_each_electrode(printer, plate):
    _, variants, layers, zs, _ = plate
    grid = layout(variants, (4.0, 4.0), printer, LayoutConfig(mode="pack"))
    root = ET.fromstring(build_svg(printer, variants, layers, zs, SvgStyle(), 4.0, grid))
    labels = [t.text for t in root.iter(f"{SVG_NS}text")]
    assert sorted(labels) == sorted(v.label for v in variants)


def test_labels_can_be_left_out_entirely(plate):
    printer, variants, layers, zs, grid = plate
    root = ET.fromstring(
        build_svg(printer, variants, layers, zs, SvgStyle(labels=False), 4.0, grid)
    )
    assert not list(root.iter(f"{SVG_NS}text"))


def test_an_invalid_svg_mode_is_refused():
    with pytest.raises(ValueError, match="svg mode"):
        SvgStyle(mode="wireframe")


def test_writing_creates_parent_directories(tmp_path):
    target = write_svg(tmp_path / "a" / "b" / "plate.svg", "<svg/>")
    assert target.read_text() == "<svg/>"


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


def test_the_manifest_records_the_machine_the_model_and_every_electrode(plate, stl_path):
    printer, variants, layers, zs, grid = plate
    loaded = load_mesh(stl_path)
    manifest = build_manifest(
        stl_path, loaded, printer, SliceConfig(), LayoutConfig(),
        variants, layers, [(z, z + 0.05) for z in zs], SvgStyle(),
    )
    assert manifest["printer"]["line_width_mm"] == printer.line_width
    assert manifest["model"]["input_units"] == "m"
    assert len(manifest["variants"]) == len(variants)

    by_id = {v["id"]: v for v in manifest["variants"]}
    assert by_id["lines_40pct"]["crossings_total"] == 0
    assert by_id["grid_40pct"]["crossings_total"] > 0
    assert by_id["reference_solid_100pct"]["is_reference"] is True
    for entry in manifest["variants"]:
        assert len(entry["layers"]) == len(zs)
        assert entry["max_islands"] >= 1


def test_the_manifest_is_json_serialisable(plate, stl_path, tmp_path):
    printer, variants, layers, zs, grid = plate
    manifest = build_manifest(
        stl_path, load_mesh(stl_path), printer, SliceConfig(), LayoutConfig(),
        variants, layers, [(z, z + 0.05) for z in zs], SvgStyle(),
    )
    target = write_manifest(tmp_path / "m.json", manifest)
    assert json.loads(target.read_text())["variants"][0]["pattern"] == "lines"
