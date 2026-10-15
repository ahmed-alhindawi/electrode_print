"""Alignment marks: the same in every file, clear of everything, and asymmetric."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import Point, box

from electrode_slicer import SweepRequest, run_sweep
from electrode_slicer.layout import LayoutConfig

pytestmark = pytest.mark.usefixtures("electrode")

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def plate(electrode, printer):
    return run_sweep(electrode, printer,
                     SweepRequest(patterns=["lines", "grid"], densities=[0.2, 0.6]))


def test_three_marks_sit_in_an_l_round_the_electrodes(plate):
    bl, br, tr = plate.fiducials
    assert bl.y == pytest.approx(br.y) and br.x == pytest.approx(tr.x)
    assert bl.x < br.x and br.y < tr.y


def test_no_turn_or_flip_maps_the_marks_onto_themselves(plate):
    pts = [(f.x, f.y) for f in plate.fiducials]
    cx = sum(x for x, _ in pts) / 3
    cy = sum(y for _, y in pts) / 3
    norm = sorted((round(x - cx, 6), round(y - cy, 6)) for x, y in pts)
    for fx, fy, swap in [(-1, -1, False), (-1, 1, False), (1, -1, False),
                         (1, 1, True), (-1, -1, True), (1, -1, True), (-1, 1, True)]:
        moved = sorted((round(fx * (y if swap else x), 6), round(fy * (x if swap else y), 6))
                       for x, y in norm)
        assert moved != norm


def test_marks_stand_clear_of_every_electrode(plate):
    w, d = plate.loaded.width, plate.loaded.extents[1]
    parts = [box(v.dx, v.dy, v.dx + w, v.dy + d) for v in plate.variants]
    for f in plate.fiducials:
        disc = Point(f.x, f.y).buffer(f.diameter / 2)
        assert all(disc.distance(p) >= 0.9 for p in parts)


def test_the_dot_stands_clear_of_the_ring(plate, printer):
    # Touching, a region slicer would merge them into one blob.
    for f in plate.fiducials:
        assert f.gap >= printer.line_width
        assert f.diameter == pytest.approx(10 * printer.line_width)


def fiducial_block(text):
    start = text.index("G04 alignment marks")
    return text[start:text.index("D10*", start) + 4]


def test_every_gerber_file_carries_the_same_marks(plate):
    files = [text for *_, text in plate.gerber_passes()] + plate.gerbers()
    blocks = {fiducial_block(t) for t in files}
    assert len(blocks) == 1
    block = blocks.pop()
    assert block.count("D03*") == 3            # three centre dots, flashed
    assert all("%ADD11C," in t for t in files)


def test_the_svg_and_the_manifest_record_the_marks(plate):
    root = ET.fromstring(plate.svg())
    group = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "fiducials")
    assert len(group.findall(f"{SVG_NS}path")) == 3
    assert len(group.findall(f"{SVG_NS}circle")) == 3
    marks = json.loads(json.dumps(plate.manifest()))["fiducials"]
    assert [(m["x_mm"], m["y_mm"]) for m in marks] == [
        (round(f.x, 4), round(f.y, 4)) for f in plate.fiducials]


def test_marks_can_be_left_off(electrode, printer):
    result = run_sweep(electrode, printer, SweepRequest(patterns=["lines"], densities=[0.2]),
                       layout_config=LayoutConfig(fiducials=False))
    assert result.fiducials == []
    assert "alignment marks" not in result.gerbers()[0]
    assert 'id="fiducials"' not in result.svg()


def test_marks_crowded_into_the_margin_are_flagged(electrode, printer):
    # Room for the electrodes, but not for marks beside them.
    tight = printer.with_overrides(plate_x=34, plate_y=40, margin=5)
    result = run_sweep(electrode, tight, SweepRequest(patterns=["lines", "grid"],
                                                      densities=[0.2, 0.4, 0.6], reference=0),
                       layout_config=LayoutConfig(gap=1, labels=False))
    assert any("alignment marks" in w for w in result.warnings)


def test_marks_follow_the_plate_layout(electrode, printer):
    matrix = run_sweep(electrode, printer, SweepRequest(patterns=["lines"], densities=[0.2]))
    packed = run_sweep(electrode, printer, SweepRequest(patterns=["lines"], densities=[0.2]),
                       layout_config=LayoutConfig(mode="pack"))
    for result in (matrix, packed):
        assert re.search(r"G04 alignment marks", result.gerbers()[0])
        assert len(result.fiducials) == 3
