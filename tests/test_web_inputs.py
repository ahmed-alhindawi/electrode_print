"""Malformed requests must be refused cleanly, never crash the server.

The browser sends whatever is in the form.  An empty or half-typed number box
arrives as JSON null, so every field has to survive that and say which one was
wrong -- a 500 and "the slicer failed" tells the user nothing and leaves a
traceback in a terminal they may not be watching.
"""

from __future__ import annotations

import base64
import itertools
import json
import threading
import urllib.error
import urllib.request

import pytest

from electrode_slicer.web import SliceService, serve

NUMERIC = ["line_width", "layer_height", "perimeters", "solid_layers",
           "replicates", "reference", "scale", "rotate", "gap", "margin",
           "infill_angle"]
TEXTUAL = ["layout", "svg_mode", "units", "printer"]
NASTY = [None, "", "   ", "abc", [], {}, True, -1, 0, 1.5, 1e9, -1e9, "NaN"]


@pytest.fixture(scope="module")
def stl_b64():
    from tests.conftest import STL
    if not STL.exists():
        pytest.skip("model/electrode.stl not present")
    return base64.b64encode(STL.read_bytes()).decode()


@pytest.fixture(scope="module")
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


def post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, {}
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except ValueError:
            return exc.code, {}


def base_payload(stl_b64):
    return {"stl": stl_b64, "patterns": ["lines"], "densities": [40], "reference": 0}


# --------------------------------------------------------------------------
# the field sweep
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", NUMERIC + TEXTUAL + ["patterns", "densities"])
def test_no_field_can_crash_the_server(field, server, stl_b64):
    """Every value must come back 200 or 400 -- never a 500."""
    for value in NASTY:
        code, body = post(server, "/api/slice", {**base_payload(stl_b64), field: value})
        assert code in (200, 400), f"{field}={value!r} gave {code}"
        if code == 400:
            assert body.get("error"), f"{field}={value!r} was refused without a reason"


@pytest.mark.parametrize("field", NUMERIC)
def test_a_blank_number_box_falls_back_to_the_default(field, service, stl_b64):
    """Clearing a field in the browser sends null; that must not be fatal."""
    for blank in (None, "", "   "):
        out = service.slice({**base_payload(stl_b64), field: blank})
        # Blanking "reference" restores its default of one, so the count is
        # one or two depending on the field -- the point is that it sliced.
        assert out["electrodes"] >= 1
        assert out["svg"].startswith("<?xml")


@pytest.mark.parametrize("field", NUMERIC)
def test_an_unparseable_number_names_the_field_it_came_from(field, service, stl_b64):
    with pytest.raises(ValueError, match=field):
        service.slice({**base_payload(stl_b64), field: "abc"})


@pytest.mark.parametrize("field", NUMERIC)
def test_a_number_sent_as_a_list_names_the_field(field, service, stl_b64):
    with pytest.raises(ValueError, match=field):
        service.slice({**base_payload(stl_b64), field: [1]})


@pytest.mark.parametrize("field", ["perimeters", "replicates", "solid_layers"])
def test_an_absurd_count_is_refused_rather_than_attempted(field, service, stl_b64):
    with pytest.raises(ValueError, match="unreasonably large"):
        service.slice({**base_payload(stl_b64), field: 10**9})


def test_infinity_and_nan_are_refused(service, stl_b64):
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            service.slice({**base_payload(stl_b64), "line_width": value})


# --------------------------------------------------------------------------
# uploads
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (None, "no model"),
        ("", "no model"),
        (12345, "base64 text"),
        (["a"], "base64 text"),
        ("not base64!!", "valid base64"),
    ],
)
def test_a_bad_model_field_is_explained(model, message, service, stl_b64):
    with pytest.raises(ValueError, match=message):
        service.slice({**base_payload(stl_b64), "stl": model})


def test_a_corrupt_upload_does_not_surface_a_loader_traceback(service):
    """trimesh raises all sorts from inside its loaders; none of it is useful.

    Which of the two messages comes back depends on how far the loader gets
    before giving up; either is a plain sentence about the upload.
    """
    with pytest.raises(ValueError, match="could not read that upload|no geometry found"):
        service.slice({"stl": base64.b64encode(b"hello world" * 50).decode(),
                       "patterns": ["lines"], "densities": [40]})


def test_a_truncated_stl_is_explained(service, stl_b64):
    with pytest.raises(ValueError, match="could not read|no geometry"):
        service.slice({**base_payload(stl_b64), "stl": stl_b64[:200]})


def test_an_ascii_stl_loads(service):
    """Plenty of CAD tools export ASCII rather than binary."""
    ascii_stl = b"""solid t
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 5 0 0
  vertex 0 5 0
 endloop
endfacet
endsolid t
"""
    from electrode_slicer.io import load_mesh_bytes
    loaded = load_mesh_bytes(ascii_stl, units="mm")
    assert loaded.width == pytest.approx(5.0)


def test_a_non_utf8_ascii_stl_loads(service):
    """A degree sign in the header should not take the loader down."""
    from electrode_slicer.io import load_mesh_bytes
    ascii_stl = (
        b"solid \xb0t\nfacet normal 0 0 1\n outer loop\n"
        b"  vertex 0 0 0\n  vertex 5 0 0\n  vertex 0 5 0\n endloop\n"
        b"endfacet\nendsolid t\n"
    )
    assert load_mesh_bytes(ascii_stl, units="mm").width == pytest.approx(5.0)


# --------------------------------------------------------------------------
# the detail endpoint
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", [None, "", 7, ["x"], {}, "nope"])
def test_a_bad_variant_is_refused_cleanly(variant, server, stl_b64):
    code, body = post(server, "/api/detail",
                      {**base_payload(stl_b64), "variant": variant})
    assert code == 400
    assert body.get("error")


# --------------------------------------------------------------------------
# request framing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["null", "[]", '"text"', "12", "not json at all", ""])
def test_a_body_that_is_not_an_object_is_refused(body, server):
    request = urllib.request.Request(
        server + "/api/slice", data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError(f"{body!r} should have been refused")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


@pytest.mark.parametrize("field", ["patterns", "densities"])
def test_unticking_everything_is_explained_not_silently_defaulted(field, service, stl_b64):
    """Coming back with a plate of 'lines' would look like the UI ignored you."""
    with pytest.raises(ValueError, match="select at least one"):
        service.slice({**base_payload(stl_b64), field: []})


def test_an_omitted_list_still_falls_back_to_a_default(service, stl_b64):
    payload = base_payload(stl_b64)
    payload.pop("patterns")
    assert service.slice(payload)["electrodes"] >= 1


def test_the_first_load_defaults_span_the_crossing_axis(service):
    """A new user should see the effect without touching anything."""
    defaults = service.patterns()["defaults"]
    classes = {p["name"]: p["overlap"] for p in service.patterns()["patterns"]}
    spanned = {classes[name] for name in defaults["patterns"]}
    assert "none" in spanned and "crossing" in spanned
    assert defaults["densities"] == [20, 40, 60, 80]


def test_a_combinatorial_explosion_is_refused_before_it_starts(service, stl_b64):
    with pytest.raises(ValueError, match="keep it under"):
        service.slice({**base_payload(stl_b64),
                       "patterns": ["lines", "grid"],
                       "densities": list(range(1, 100)), "replicates": 9})


def test_the_whole_nasty_matrix_stays_on_its_feet(server, stl_b64):
    """The broad sweep: nothing in here may return a 500."""
    fields = NUMERIC + TEXTUAL
    failures = []
    for field, value in itertools.product(fields, [None, "abc", [], {}, True]):
        code, _ = post(server, "/api/slice", {**base_payload(stl_b64), field: value})
        if code not in (200, 400):
            failures.append((field, value, code))
    assert not failures, failures
