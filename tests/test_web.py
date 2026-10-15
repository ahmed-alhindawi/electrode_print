"""The local web front end.

The server is started on an ephemeral port for the HTTP tests, so they never
collide with a copy the user happens to be running.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import socket
import threading
import urllib.error
import urllib.request
import zipfile

import pytest

from electrode_slicer.web import SliceService, serve


@pytest.fixture(scope="module")
def stl_b64():
    from tests.conftest import STL
    if not STL.exists():
        pytest.skip("model/electrode_6mm.stl not present")
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


def connect(base):
    host, port = base.removeprefix("http://").split(":")
    return http.client.HTTPConnection(host, int(port), timeout=60)


def test_the_connection_stays_open_after_a_response(server):
    # VS Code forwards a dev container's port through a relay that exits as
    # soon as the server hangs up, dropping whatever it has not passed on; a
    # page served on a connection the server closes can arrive cut short.
    conn = connect(server)
    try:
        conn.request("GET", "/")
        page = conn.getresponse()
        body = page.read()
        assert page.status == 200 and not page.will_close
        assert len(body) == int(page.getheader("Content-Length"))
        sock = conn.sock

        # A refusal after the body has been read leaves the connection usable.
        conn.request("POST", "/api/slice", body=b'"just a string"',
                     headers={"Content-Type": "application/json"})
        refused = conn.getresponse()
        refused.read()
        assert refused.status == 400 and not refused.will_close

        conn.request("GET", "/api/health")
        health = conn.getresponse()
        assert json.loads(health.read()) == {"ok": True}
        assert conn.sock is sock
    finally:
        conn.close()


def test_a_post_refused_unread_ends_the_connection(server):
    # Kept alive, the unread body would be parsed as the next request.
    conn = connect(server)
    try:
        conn.request("POST", "/api/nope", body=b'{"stl": "x"}',
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        assert json.loads(response.read())["error"] == "not found"
        assert response.status == 404 and response.will_close
    finally:
        conn.close()


def test_an_oversized_upload_is_refused_without_waiting_for_it(server):
    host, port = server.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=30) as sock:
        sock.sendall(b"POST /api/slice HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Length: 999999999999\r\n\r\n")
        reply = b""
        while chunk := sock.recv(65536):     # the server hangs up after replying
            reply += chunk
    assert reply.startswith(b"HTTP/1.1 413")
    assert b"Connection: close" in reply


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


def test_gerber_export_zips_one_gtl_per_layer(server, stl_b64):
    status, body, kind = call(server, "/api/gerber", {
        "stl": stl_b64, "patterns": ["grid"], "densities": [40], "reference": 0,
        "filename": "electrode.stl", "gtl_mode": "layers",
    })
    assert status == 200
    assert kind == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert names == [f"electrode-layer_{i:03d}.GTL" for i in range(3)]


def test_a_single_layer_part_exports_a_bare_gtl(service, stl_b64):
    name, body, kind = service.gerber({
        "stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0,
        "layer_height": 1.0, "filename": "pad.stl", "gtl_mode": "layers",
    })
    assert name == "pad-plate.GTL"
    assert kind == "application/vnd.gerber"
    assert body.decode("ascii").rstrip().endswith("M02*")


def test_every_electrode_reaches_the_gerber_outside_any_extended_command(service, stl_b64):
    # A stray "%" opens an extended command, and a viewer then skips every
    # electrode up to the next one.
    _, body, _ = service.gerber({
        "stl": stl_b64, "patterns": ["lines", "grid"], "densities": [20, 60],
        "reference": 1, "layer_height": 1.0, "gtl_mode": "layers", "fiducials": False,
    })
    lines = body.decode("ascii").splitlines()
    assert all(ln.startswith("%") and ln.endswith("*%") for ln in lines if "%" in ln)
    assert sum(ln.startswith("G04 ") for ln in lines) == 1 + 5


def test_the_mesh_comes_back_in_the_frame_modifiers_are_placed_in(service, stl_b64):
    import numpy as np
    data = service.mesh({"stl": stl_b64})
    vertices = np.frombuffer(base64.b64decode(data["vertices"]), "<f4").reshape(-1, 3)
    faces = np.frombuffer(base64.b64decode(data["faces"]), "<u4").reshape(-1, 3)
    assert vertices.min(axis=0) == pytest.approx([0, 0, 0], abs=1e-4)
    assert vertices.max(axis=0) == pytest.approx(data["extents"], abs=1e-3)
    assert faces.max() < len(vertices)


def test_a_modifier_confines_the_sweep_to_the_electrode_face(service, stl_b64):
    # The electrode is a 5 mm disc at the origin with its probe tab running
    # off to +x on the first layer only.
    base = {"stl": stl_b64, "patterns": ["lines"], "densities": [20], "reference": 0}
    disc = {"shape": "cylinder", "center": [2.5, 2.5, 0.15], "size": [5, 5, 0.4]}
    whole = service.detail({**base, "variant": "lines_20pct"})
    split = service.detail({**base, "variant": "lines_20pct", "modifiers": [disc]})
    assert not any(layer["solid"] for layer in whole["layers"])
    tab = split["layers"][0]
    assert tab["solid"]
    assert min(x for path in tab["solid"] for x, _ in path) > 4.5
    assert max(x for path in tab["infill"] for x, _ in path) < 5.0 + split["line_width"]


@pytest.mark.parametrize("bad", [
    "nope", [{"shape": "cone", "center": [0, 0, 0], "size": [1, 1, 1]}],
    [{"shape": "box", "center": [0, 0], "size": [1, 1, 1]}],
    [{"shape": "box", "center": [0, 0, 0], "size": [1, -1, 1]}],
])
def test_a_bad_modifier_is_explained(server, stl_b64, bad):
    status, body, _ = call(server, "/api/slice", {"stl": stl_b64, "modifiers": bad})
    assert status == 400
    assert "modifier" in json.loads(body)["error"]


def test_gerber_passes_put_the_walls_first_then_crossing_free_infill(service, stl_b64):
    _, body, kind = service.gerber({
        "stl": stl_b64, "patterns": ["grid"], "densities": [40], "reference": 0,
        "filename": "electrode.stl",
    })
    names = sorted(zipfile.ZipFile(io.BytesIO(body)).namelist())
    assert kind == "application/zip"
    assert names[0] == "electrode-layer_000-pass_1-walls.GTL"
    # A grid's two families cross, so they cannot share a pass.
    assert "electrode-layer_000-pass_3-infill.GTL" in names


def test_gerber_can_still_be_written_one_file_per_layer(service, stl_b64):
    _, body, _ = service.gerber({
        "stl": stl_b64, "patterns": ["grid"], "densities": [40], "reference": 0,
        "filename": "electrode.stl", "gtl_mode": "layers",
    })
    names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert names == [f"electrode-layer_{i:03d}.GTL" for i in range(3)]


def test_an_unknown_gtl_mode_is_refused(service, stl_b64):
    with pytest.raises(ValueError, match="gtl_mode"):
        service.gerber({"stl": stl_b64, "gtl_mode": "both"})


def test_a_rejected_gerber_export_explains_itself(server, stl_b64):
    status, body, _ = call(server, "/api/gerber", {"stl": stl_b64, "patterns": ["nope"]})
    assert status == 400
    assert "unknown infill pattern" in json.loads(body)["error"]


def test_toolpaths_predict_each_pass_filled_by_the_tip(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["grid"], "densities": [40], "reference": 0}
    out = service.toolpaths({**payload, "variant": "grid_40pct", "layer": 1})
    assert out["layer"] == 1 and out["tip_um"] == 100
    roles = [p["role"] for p in out["passes"]]
    assert roles[0] == "walls" and roles.count("infill") >= 2
    for p in out["passes"]:
        assert p["artwork"] and p["loops"] and len(p["single"]) == len(p["loops"])
        assert p["gap_pct"] < 5, "a 100 um tip prints 100 um beads"


def test_a_tip_too_wide_for_the_beads_shows_as_gaps(service, stl_b64):
    payload = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}
    out = service.toolpaths({**payload, "variant": "lines_40pct", "tip_um": 150})
    assert out["passes"][1]["gap_pct"] > 90


@pytest.mark.parametrize("bad", [{"layer": 99}, {"tip_um": 0}, {"variant": "nope"}])
def test_a_bad_toolpaths_request_is_explained(service, stl_b64, bad):
    payload = {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0,
               "variant": "lines_40pct", **bad}
    with pytest.raises(ValueError):
        service.toolpaths(payload)


def test_the_toolpaths_route_is_served(server, stl_b64):
    status, body, _ = call(server, "/api/toolpaths", {
        "stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0,
        "variant": "lines_40pct"})
    assert status == 200
    assert json.loads(body)["passes"]
