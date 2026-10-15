"""Assemble the static site: the page, the worker, and this package zipped."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]                    # src/electrode_slicer
STATIC = HERE.parent / "static"

#: The Pyodide release the worker loads.  It ships numpy, scipy, shapely,
#: networkx and contourpy built for WebAssembly; trimesh is pure Python and
#: comes from PyPI.
PYODIDE = "0.28.3"


def build(out: str | Path) -> Path:
    """Write the site into `out`, replacing whatever was there."""
    import trimesh

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    page = (STATIC / "index.html").read_text(encoding="utf-8")
    # The shim must run before the page's own script makes its first request.
    marker = "<script>"
    assert marker in page, "index.html has no inline script to put the shim before"
    page = page.replace(marker, '<script src="shim.js"></script>\n' + marker, 1)
    (out / "index.html").write_text(page, encoding="utf-8")

    (out / "shim.js").write_text((HERE / "shim.js").read_text(encoding="utf-8"),
                                 encoding="utf-8")
    worker = (HERE / "worker.js").read_text(encoding="utf-8")
    worker = worker.replace("{{PYODIDE}}", PYODIDE).replace("{{TRIMESH}}", trimesh.__version__)
    (out / "worker.js").write_text(worker, encoding="utf-8")

    # The package as Pyodide will import it, without the parts it never runs.
    with zipfile.ZipFile(out / "electrode_slicer.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(PACKAGE.rglob("*")):
            rel = path.relative_to(PACKAGE.parent)
            if path.is_dir() or "__pycache__" in rel.parts or path.suffix == ".pyc":
                continue
            if path.suffix in (".js", ".html"):
                continue                     # the page itself is served alongside
            bundle.write(path, rel.as_posix())

    # GitHub Pages runs Jekyll by default, which hides files starting with "_".
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out
