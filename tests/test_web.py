"""The local web front end.

The server is started on an ephemeral port for the HTTP tests, so they never
collide with a copy the user happens to be running.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request

import pytest

from electrode_slicer.web import SliceService, serve


@pytest.fixture(scope="module")
def stl_b64():
    from tests.conftest import STL
    if not STL.exists():
        pytest.skip("model/electrode.stl not present")
    return base64.b64encode(STL.read_bytes()).decode()


@pytest.fixture
def service():
    return SliceService()


@pytest.fixture(scope="module")
def server():
    httpd = serve("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def call(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------


def test_the_catalogue_lists_patterns_printers_and_defaults(service):
    meta = service.patterns()
    names = {p["name"] for p in meta["patterns"]}
    assert {"lines", "grid", "stars", "gyroid", "honeycomb"} <= names
    assert all(p["overlap"] in ("none", "junction", "crossing") for p in meta["patterns"])
    assert meta["defaults"]["line_width"] == 0.1
    assert any(p["name"] == "voltera-nova" for p in meta["printers"])


def test_slicing_returns_an_svg_a_manifest_and_a_summary(service, stl_b64):
    out = service.slice({"stl": stl_b64, "patterns": ["lines", "grid"],
                         "densities": [20, 60], "reference": 1})
    assert out["svg"].startswith("<?xml")
    assert out["electrodes"] == 5
    assert out["layers"] == 3
    assert len(out["manifest"]["variants"]) == 5
    rows = {(r["pattern"], r["target_pct"]): r for r in out["summary"]}
    assert rows[("lines", 60.0)]["crossings"] == 0
    assert rows[("grid", 60.0)]["crossings"] > 0
    assert all(r["islands"] == 1 for r in out["summary"])


def test_the_preview_matches_what_the_cli_would_write(service, stl_b64, printer):
    """One pipeline, so the browser cannot drift from the machine."""
    from electrode_slicer import SweepRequest, load_mesh, run_sweep
    from electrode_slicer.io import SvgStyle
    from electrode_slicer.layout import LayoutConfig
    from electrode_slicer.printer import VOLTERA_NOVA
    from electrode_slicer.slicer import SliceConfig

    web = service.slice({"stl": stl_b64, "patterns": ["stars"], "densities": [40],
                         "reference": 0})
    direct = run_sweep(
        load_mesh(__import__("tests.conftest", fromlist=["STL"]).STL),
        VOLTERA_NOVA,
        SweepRequest(patterns=["stars"], densities=[0.4], reference=0),
        SliceConfig(), LayoutConfig(), SvgStyle(),
    )
    assert web["svg"] == direct.svg(title="electrodes")


def test_the_mesh_is_parsed_once_and_reused(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}
    service.slice(payload)
    first = service.mesh_for(base64.b64decode(stl_b64), "auto", 1.0, 0.0)
    second = service.mesh_for(base64.b64decode(stl_b64), "auto", 1.0, 0.0)
    assert first is second


def test_a_different_scale_is_not_served_from_the_mesh_cache(service, stl_b64):
    raw = base64.b64decode(stl_b64)
    assert service.mesh_for(raw, "auto", 1.0, 0.0) is not service.mesh_for(raw, "auto", 2.0, 0.0)


def test_repeating_a_request_reuses_the_built_geometry(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["lines", "grid"], "densities": [40],
               "reference": 0}
    first = service.slice(payload)
    built = len(service._layers)
    second = service.slice(payload)
    assert first["svg"] == second["svg"]
    assert len(service._layers) == built, "nothing changed, so nothing was rebuilt"


def test_changing_the_bead_width_rebuilds(service, stl_b64):
    base = {"stl": stl_b64, "patterns": ["grid"], "densities": [40], "reference": 0}
    coarse = service.slice({**base, "line_width": 0.3})
    fine = service.slice({**base, "line_width": 0.1})
    assert coarse["svg"] != fine["svg"]
    assert 'stroke-width="0.3"' in coarse["svg"]
    assert 'stroke-width="0.1"' in fine["svg"]


# --------------------------------------------------------------------------
# changing the machine from the front end
# --------------------------------------------------------------------------


def test_the_catalogue_carries_enough_to_fill_the_printer_form(service):
    meta = service.patterns()
    machine = next(m for m in meta["printers"] if m["name"] == "voltera-nova")
    assert machine["plate"] == [220.0, 300.0, 40.0]
    assert machine["line_width"] == 0.1
    for key in ("layer_height", "margin", "resolution", "first_layer_height"):
        assert key in machine
    for key in ("plate_x", "plate_y", "z_travel", "margin", "resolution", "gap"):
        assert key in meta["defaults"], key


def test_the_build_plate_can_be_changed(service, stl_b64):
    out = service.slice({"stl": stl_b64, "patterns": ["lines"], "densities": [40],
                         "reference": 0, "plate_x": 100, "plate_y": 80, "z_travel": 15})
    assert out["plate"] == [100.0, 80.0, 15.0]
    assert 'width="100mm"' in out["svg"]
    assert 'height="80mm"' in out["svg"]
    assert 'data-plate="100x80x15"' in out["svg"]


@pytest.mark.parametrize("width", [0.05, 0.2, 0.4])
def test_the_extrusion_width_reaches_the_toolpaths_and_the_stroke(width, service, stl_b64):
    base = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}
    out = service.slice({**base, "line_width": width})
    assert out["printer"]["line_width"] == width
    assert f'stroke-width="{width}"' in out["svg"]
    # A wider bead covers the same area in fewer, shorter passes.
    finer = service.slice({**base, "line_width": width / 2})
    assert finer["summary"][0]["path_mm"] > out["summary"][0]["path_mm"]


def test_the_layer_height_changes_the_layer_count(service, stl_b64):
    base = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}
    assert service.slice({**base, "layer_height": 0.1})["layers"] == 3
    assert service.slice({**base, "layer_height": 0.05})["layers"] == 6


def test_a_blank_first_layer_follows_the_layer_height(service, stl_b64):
    """The field is left empty on purpose; the machine decides."""
    base = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0,
            "layer_height": 0.06}
    assert service.slice({**base, "first_layer_height": None})["printer"]["first_layer"] == 0.06
    assert service.slice({**base, "first_layer_height": 0.2})["printer"]["first_layer"] == 0.2


def test_the_margin_and_gap_change_how_much_fits(service, stl_b64):
    base = {"stl": stl_b64, "patterns": ["lines", "grid"], "densities": [20, 40, 60, 80],
            "reference": 0, "replicates": 4}
    roomy = service.slice({**base, "margin": 2, "gap": 1})
    assert roomy["electrodes"] == 32
    with pytest.raises(ValueError, match="plate|fit"):
        service.slice({**base, "margin": 95, "gap": 20})


def test_the_resolution_trades_file_size_for_fidelity(service, stl_b64):
    base = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}
    fine = service.slice({**base, "resolution": 0})
    coarse = service.slice({**base, "resolution": 0.05})
    assert len(coarse["svg"]) < len(fine["svg"])
    assert coarse["printer"]["resolution"] == 0.05


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"plate_x": -10}, "positive"),
        ({"plate_y": 0}, "positive"),
        ({"z_travel": 0}, "positive"),
        ({"line_width": 0}, "positive"),
        ({"layer_height": -1}, "positive"),
        ({"margin": -5}, "negative"),
        ({"margin": 200}, "usable plate"),
        ({"plate_x": 5, "plate_y": 5}, "usable plate"),
    ],
)
def test_an_impossible_machine_is_refused_with_a_reason(override, message, service, stl_b64):
    with pytest.raises(ValueError, match=message):
        service.slice({"stl": stl_b64, "patterns": ["lines"], "densities": [40],
                       "reference": 0, **override})


def test_a_part_too_tall_for_a_shortened_z_travel_is_refused(service, stl_b64):
    with pytest.raises(ValueError, match="Z travel"):
        service.slice({"stl": stl_b64, "patterns": ["lines"], "densities": [40],
                       "reference": 0, "z_travel": 0.1})


def test_an_unknown_preset_falls_back_rather_than_failing(service, stl_b64):
    """The dropdown cannot offer one, but a stale request should still work."""
    out = service.slice({"stl": stl_b64, "patterns": ["lines"], "densities": [40],
                         "reference": 0, "printer": "some-other-machine"})
    assert out["printer"]["name"] == "voltera-nova"


# --------------------------------------------------------------------------
# electrode detail, for the 3D view
# --------------------------------------------------------------------------


def test_detail_returns_one_electrode_as_stacked_polylines(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["stars"], "densities": [40], "reference": 0}
    detail = service.detail({**payload, "variant": "stars_40pct"})
    assert detail["variant"] == "stars_40pct"
    assert detail["overlap"] == "crossing"
    assert detail["extents"] == pytest.approx([6.5, 5.0, 0.3], abs=0.01)
    assert detail["line_width"] == 0.1
    assert len(detail["layers"]) == 3
    for layer in detail["layers"]:
        assert layer["perimeters"] and layer["infill"]
        assert all(len(point) == 2 for path in layer["infill"] for point in path)
        assert layer["top"] > layer["z"], "the bead sits below the top of its layer"


def test_detail_carries_the_hump_positions(service, stl_b64):
    """The SVG has no way to say where a bead was laid over another."""
    payload = {"stl": stl_b64, "patterns": ["stars"], "densities": [40], "reference": 0}
    detail = service.detail({**payload, "variant": "stars_40pct"})
    humps = [h for layer in detail["layers"] for h in layer["crossings"]]
    assert humps
    for x, y, meeting in humps:
        assert 0 <= x <= detail["extents"][0] + 0.5
        assert 0 <= y <= detail["extents"][1] + 0.5
        assert meeting >= 2, "a crossing is at least two beads"


def test_detail_hump_counts_agree_with_the_summary(service, stl_b64):
    """The 3D markers and the measurements table must not disagree."""
    payload = {"stl": stl_b64, "patterns": ["grid"], "densities": [60], "reference": 0}
    summary = next(r for r in service.slice(payload)["summary"] if r["pattern"] == "grid")
    detail = service.detail({**payload, "variant": "grid_60pct"})
    assert sum(len(layer["crossings"]) for layer in detail["layers"]) == summary["crossings"]


def test_a_crossing_free_pattern_reports_no_humps(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}
    detail = service.detail({**payload, "variant": "lines_40pct"})
    assert all(layer["crossings"] == [] for layer in detail["layers"])


def test_detail_is_small_enough_to_send_on_every_click(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["stars"], "densities": [80], "reference": 0}
    detail = service.detail({**payload, "variant": "stars_80pct"})
    assert len(json.dumps(detail)) < 2_000_000


def test_asking_for_an_electrode_that_is_not_on_the_plate(service, stl_b64):
    with pytest.raises(ValueError, match="no electrode called"):
        service.detail({"stl": stl_b64, "patterns": ["lines"], "densities": [40],
                        "variant": "gyroid_90pct"})


def test_the_slice_response_lists_what_can_be_inspected(service, stl_b64):
    out = service.slice({"stl": stl_b64, "patterns": ["lines", "grid"],
                         "densities": [20, 60], "reference": 1})
    ids = [v["id"] for v in out["variants"]]
    assert "grid_60pct" in ids
    assert any(v["reference"] for v in out["variants"])
    # Every listed electrode must actually be fetchable.
    for entry in out["variants"]:
        assert entry["id"] in ids


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "no model"),
        ({"stl": "not base64!!"}, "base64"),
        ({"stl": "", "patterns": ["lines"]}, "no model"),
    ],
)
def test_bad_uploads_are_refused_with_a_reason(service, payload, message):
    with pytest.raises(ValueError, match=message):
        service.slice(payload)


def test_an_unreasonably_large_sweep_is_refused(service, stl_b64):
    with pytest.raises(ValueError, match="keep it under"):
        service.slice({"stl": stl_b64, "patterns": ["lines"],
                       "densities": list(range(1, 90)), "replicates": 9})


def test_an_unknown_pattern_is_refused(service, stl_b64):
    with pytest.raises(KeyError, match="unknown infill pattern"):
        service.slice({"stl": stl_b64, "patterns": ["spirograph"], "densities": [40]})


# --------------------------------------------------------------------------
# over HTTP
# --------------------------------------------------------------------------


def test_the_page_is_served_at_the_root(server):
    status, body, kind = call(server, "/")
    assert status == 200
    assert kind.startswith("text/html")
    assert b"Electrode Slicer" in body
    assert b'id="drop"' in body and b'id="stats"' in body


def test_the_catalogue_endpoint_answers(server):
    status, body, kind = call(server, "/api/patterns")
    assert status == 200 and kind == "application/json"
    assert len(json.loads(body)["patterns"]) >= 9


def test_health_check(server):
    status, body, _ = call(server, "/api/health")
    assert status == 200 and json.loads(body) == {"ok": True}


def test_unknown_routes_are_404(server):
    assert call(server, "/nope")[0] == 404
    assert call(server, "/api/nope")[0] == 404


def test_a_path_traversal_cannot_escape_the_static_directory(server):
    status, body, _ = call(server, "/static/../../../../etc/passwd")
    assert status == 404
    assert b"root:" not in body


def test_slicing_over_http(server, stl_b64):
    status, body, _ = call(server, "/api/slice", {
        "stl": stl_b64, "patterns": ["lines", "stars"], "densities": [20, 60],
        "reference": 1,
    })
    assert status == 200
    data = json.loads(body)
    assert data["electrodes"] == 5
    assert data["svg"].startswith("<?xml")
    assert {r["pattern"] for r in data["summary"]} == {"lines", "stars", "solid"}


def test_a_rejected_request_comes_back_as_400_with_a_reason(server, stl_b64):
    status, body, _ = call(server, "/api/slice", {"stl": stl_b64, "patterns": ["nope"]})
    assert status == 400
    assert "unknown infill pattern" in json.loads(body)["error"]


def test_a_body_that_is_not_an_object_is_refused(server):
    status, body, _ = call(server, "/api/slice", "just a string")
    assert status == 400
    assert "JSON object" in json.loads(body)["error"]


def test_an_empty_post_is_refused(server):
    request = urllib.request.Request(server + "/api/slice", data=b"",
                                     headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError("expected a rejection")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


def test_detail_over_http(server, stl_b64):
    status, body, _ = call(server, "/api/detail", {
        "stl": stl_b64, "patterns": ["grid"], "densities": [40],
        "reference": 0, "variant": "grid_40pct",
    })
    assert status == 200
    detail = json.loads(body)
    assert detail["pattern"] == "grid"
    assert sum(len(layer["crossings"]) for layer in detail["layers"]) > 0


def test_detail_rejects_an_unknown_electrode_over_http(server, stl_b64):
    status, body, _ = call(server, "/api/detail", {
        "stl": stl_b64, "patterns": ["grid"], "densities": [40], "variant": "nope",
    })
    assert status == 400
    assert "no electrode called" in json.loads(body)["error"]


def test_a_part_that_will_not_fit_is_explained(server, stl_b64):
    status, body, _ = call(server, "/api/slice", {
        "stl": stl_b64, "patterns": ["lines"], "densities": [40], "scale": 900,
    })
    assert status == 400
    assert "Z travel" in json.loads(body)["error"]
