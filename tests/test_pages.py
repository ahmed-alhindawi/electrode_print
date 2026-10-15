"""The static (GitHub Pages) build: the same service, answered without HTTP."""

from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest

from electrode_slicer.web.pages import handle
from electrode_slicer.web.pages.build import PYODIDE, build


@pytest.fixture(scope="module")
def stl_b64():
    from tests.conftest import STL
    if not STL.exists():
        pytest.skip("model/electrode_6mm.stl not present")
    return base64.b64encode(STL.read_bytes()).decode()


def test_the_page_metadata_comes_back_as_json():
    status, body, kind, _ = handle("GET", "/api/patterns", "")
    assert status == 200 and kind == "application/json"
    assert any(p["name"] == "hilbert" for p in json.loads(body)["patterns"])


def test_a_slice_is_answered_just_as_the_server_would(stl_b64):
    request = {"stl": stl_b64, "patterns": ["lines"], "densities": [20], "reference": 0}
    status, body, _, _ = handle("POST", "/api/slice", json.dumps(request))
    assert status == 200
    assert json.loads(body)["electrodes"] == 1


def test_a_gerber_export_carries_its_filename(stl_b64):
    request = {"stl": stl_b64, "patterns": ["lines"], "densities": [20], "reference": 0,
               "filename": "probe.stl"}
    status, body, kind, headers = handle("POST", "/api/gerber", json.dumps(request))
    assert status == 200 and kind == "application/zip"
    assert 'filename="probe-plate-gerber.zip"' in headers["Content-Disposition"]
    assert zipfile.ZipFile(io.BytesIO(body)).namelist()


@pytest.mark.parametrize("method, route, body, status", [
    ("GET", "/api/nope", "", 404),
    ("POST", "/api/nope", "{}", 404),
    ("POST", "/api/slice", "not json", 400),
    ("POST", "/api/slice", "[1, 2]", 400),
    ("POST", "/api/slice", json.dumps({"patterns": ["nope"]}), 400),
])
def test_bad_requests_are_refused_with_a_reason(method, route, body, status):
    got, reply, _, _ = handle(method, route, body)
    assert got == status
    assert json.loads(reply)["error"]


def test_the_site_is_the_page_the_worker_and_the_package(tmp_path):
    out = build(tmp_path / "site")
    page = (out / "index.html").read_text()
    # The shim has to be in place before the page's first request.
    assert page.index('<script src="shim.js"></script>') < page.index("<script>")
    worker = (out / "worker.js").read_text()
    assert f"pyodide/v{PYODIDE}/" in worker and "{{" not in worker
    names = zipfile.ZipFile(out / "electrode_slicer.zip").namelist()
    assert "electrode_slicer/web/pages/__init__.py" in names
    assert "electrode_slicer/slicer.py" in names
    assert not any("__pycache__" in n or n.endswith((".pyc", ".html", ".js")) for n in names)
    assert (out / ".nojekyll").exists()
