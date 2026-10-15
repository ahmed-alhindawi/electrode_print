"""A local web front end for the slicer.

Drop an STL on the page, choose infill patterns and densities, and get the
plate back as an SVG, or as Gerber top-copper (.GTL) files, to download.

It runs the same :func:`~electrode_slicer.sweep.run_sweep` pipeline the command
line uses, rather than reimplementing the geometry in JavaScript.  That costs a
round trip per change, and buys the guarantee that what the browser previews is
exactly what the machine would be sent -- which for a tool whose whole job is
measuring sub-millimetre overlap is worth more than the latency.

Only the standard library is used here, so there is nothing extra to install::

    uv run python -m electrode_slicer.web
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import socketserver
import sys
import threading
import traceback
import webbrowser
import zipfile
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..geometry import as_polygons
from ..io import SvgStyle, load_mesh_bytes
from ..layout import LayoutConfig
from ..metrics import crossing_points
from ..passes import pass_artwork, predict_fill
from ..patterns import all_patterns
from ..printer import PRINTERS, VOLTERA_NOVA
from ..slicer import Modifier, SliceConfig
from ..sweep import SweepRequest, run_sweep

__all__ = ["serve", "main", "SliceService"]

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 64 * 1024 * 1024        # a generous STL, and a hard ceiling
MAX_ELECTRODES = 400                 # refuse a request that would take minutes
MAX_MODIFIERS = 32


class SliceService:
    """The work behind the API, kept separate from the HTTP plumbing.

    Parsing an STL costs more than slicing it, and the browser re-sends the
    same model on every slider nudge, so meshes are cached by content hash.
    """

    def __init__(self, cache_size: int = 4, scale_cache_size: int = 4096) -> None:
        self._cache: OrderedDict[tuple, object] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        # Solved infill scales survive between requests, so changing the SVG
        # mode or the layout redraws immediately instead of recalibrating
        # every pattern and density from scratch.
        self._scales: dict = {}
        self._layers: dict = {}
        self._scale_cache_size = scale_cache_size

    def mesh_for(self, data: bytes, units: str, scale: float, rotate: float):
        key = (hashlib.sha256(data).hexdigest(), units, round(scale, 6), round(rotate, 4))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        try:
            loaded = load_mesh_bytes(data, units, scale, rotate)
        except ValueError:
            raise
        except Exception as exc:
            # trimesh raises all sorts from inside its loaders, and a bad
            # upload is the user's most likely mistake, not a server fault.
            raise ValueError(
                "could not read that upload as an STL; is it a complete, "
                "uncorrupted .stl file?"
            ) from exc
        with self._lock:
            self._cache[key] = loaded
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return loaded

    def patterns(self) -> dict:
        return {
            "patterns": [
                {"name": p.name, "overlap": p.overlap_class,
                 "description": p.description, "families": p.families}
                for p in all_patterns()
            ],
            "printers": [
                {"name": p.name, "plate": list(p.plate), "line_width": p.line_width,
                 "layer_height": p.layer_height,
                 "first_layer_height": p.first_layer_height,
                 "margin": p.margin, "resolution": p.resolution,
                 "description": p.description}
                for p in PRINTERS.values()
            ],
            "defaults": {
                "printer": VOLTERA_NOVA.name,
                "plate_x": VOLTERA_NOVA.plate_x,
                "plate_y": VOLTERA_NOVA.plate_y,
                "z_travel": VOLTERA_NOVA.z_travel,
                "line_width": VOLTERA_NOVA.line_width,
                "layer_height": VOLTERA_NOVA.layer_height,
                "margin": VOLTERA_NOVA.margin,
                "resolution": VOLTERA_NOVA.resolution,
                "gap": 4.0,
                # Enough to span the crossing axis end to end without making
                # the first slice a long wait; the rest are one click away.
                "patterns": ["lines", "grid", "stars"],
                # Bracketing 20%, which printed best on the 8 mm electrode with a
                # 0.1 mm nozzle.
                "densities": [10, 15, 20, 25],
            },
        }

    def mesh(self, payload: dict) -> dict:
        """The model as loaded -- units, scale and rotation applied -- for placing modifiers.

        It is the same frame the modifiers are read in, so what lines up on
        screen lines up in the slicer.  Sent as base64 typed arrays: a mesh
        as JSON numbers is several times the size and slow to parse.
        """
        loaded = self._loaded(payload)
        mesh = loaded.mesh
        return {
            "vertices": base64.b64encode(
                np.ascontiguousarray(mesh.vertices, dtype="<f4").tobytes()).decode("ascii"),
            "faces": base64.b64encode(
                np.ascontiguousarray(mesh.faces, dtype="<u4").tobytes()).decode("ascii"),
            "extents": [round(float(v), 4) for v in loaded.extents],
            "units": loaded.units, "guessed": loaded.guessed,
        }

    def slice(self, payload: dict) -> dict:
        result = self._run(payload)
        return {
            "svg": result.svg(title="electrodes"),
            "manifest": result.manifest(payload.get("filename", "upload.stl")),
            "warnings": result.warnings,
            "model": {
                "units": result.loaded.units, "guessed": result.loaded.guessed,
                "extents": [round(v, 4) for v in result.loaded.extents],
            },
            "summary": _summary(result),
            "plate": list(result.printer.plate),
            "printer": {
                "name": result.printer.name,
                "line_width": result.printer.line_width,
                "layer_height": result.printer.layer_height,
                "first_layer": result.printer.first_layer,
                "margin": result.printer.margin,
                "resolution": result.printer.resolution,
            },
            "layers": result.n_layers,
            "electrodes": len(result.variants),
            "variants": [
                {"id": v.ident, "label": v.label, "pattern": v.pattern,
                 "density_pct": round(v.density * 100, 1),
                 "overlap": v.overlap_class, "reference": v.is_reference}
                for v in result.variants
            ],
        }

    def gerber(self, payload: dict) -> tuple[str, bytes, str]:
        """The plate as Gerber: a lone .GTL, or a zip of them.

        By default each layer is split into print passes (see
        :mod:`~electrode_slicer.passes`), because the NOVA fills Gerber
        artwork with Arachne rather than following its strokes; ``gtl_mode:
        "layers"`` writes one file per layer as drawn instead.

        Returned as (filename, body, content type).  Kept off the slice
        response because it is only wanted on export, not on every nudge.
        """
        result = self._run(payload)
        stem = Path(_text(payload, "filename", "electrodes")).stem or "electrodes"
        # The name goes into a header; keep it to characters that cannot break one.
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
        mode = _text(payload, "gtl_mode", "passes")
        if mode == "passes":
            # Named so a sorted listing is the print order.
            files = [(f"{stem}-layer_{idx:03d}-pass_{k}-{role}.GTL", text)
                     for idx, k, role, text in result.gerber_passes(title=stem)]
        elif mode == "layers":
            files = [(f"{stem}-layer_{idx:03d}.GTL", text)
                     for idx, text in enumerate(result.gerbers(title=stem))]
        else:
            raise ValueError("gtl_mode must be 'passes' or 'layers'")
        if len(files) == 1:
            return f"{stem}-plate.GTL", files[0][1].encode("ascii"), "application/vnd.gerber"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, text in files:
                bundle.writestr(name, text)
        return f"{stem}-plate-gerber.zip", buffer.getvalue(), "application/zip"

    def _loaded(self, payload: dict):
        """The uploaded model, decoded, checked and prepared (cached)."""
        model = payload.get("stl")
        if not model:
            raise ValueError("no model uploaded")
        if not isinstance(model, str):
            raise ValueError(
                f"the model must be sent as base64 text, not a {type(model).__name__}"
            )
        try:
            data = base64.b64decode(model, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise ValueError("the uploaded model is not valid base64") from exc
        if len(data) > MAX_UPLOAD:
            raise ValueError(f"model is larger than {MAX_UPLOAD // (1024 * 1024)} MB")
        return self.mesh_for(
            data, _text(payload, "units", "auto"),
            _num(payload, "scale", 1.0), _num(payload, "rotate", 0.0),
        )

    def _run(self, payload: dict):
        patterns = [
            _pattern_name(p) for p in _sequence(payload, "patterns", ["lines"])
        ]
        densities = [
            _number(d, "density") / 100.0
            for d in _sequence(payload, "densities", [40])
        ]
        replicates = _num(payload, "replicates", 1, int)
        reference = _num(payload, "reference", 1, int)
        planned = len(patterns) * len(densities) * max(replicates, 1) + max(reference, 0)
        if planned > MAX_ELECTRODES:
            raise ValueError(
                f"that is {planned} electrodes; keep it under {MAX_ELECTRODES} "
                f"or run the sweep from the command line"
            )

        printer = PRINTERS.get(_text(payload, "printer", ""), VOLTERA_NOVA)
        # Each request rebuilds the machine from its preset, so a field the
        # browser leaves blank falls back to the preset rather than sticking
        # at whatever it was last set to.
        printer = printer.with_overrides(
            plate_x=_opt_float(payload, "plate_x"),
            plate_y=_opt_float(payload, "plate_y"),
            z_travel=_opt_float(payload, "z_travel"),
            line_width=_opt_float(payload, "line_width"),
            layer_height=_opt_float(payload, "layer_height"),
            first_layer_height=_opt_float(payload, "first_layer_height"),
            margin=_opt_float(payload, "margin"),
            resolution=_opt_float(payload, "resolution"),
        )
        config = SliceConfig(
            perimeters=_num(payload, "perimeters", 2, int),
            infill_angle=_num(payload, "infill_angle", 45.0),
            solid_layers=_num(payload, "solid_layers", 0, int),
            calibrate=bool(payload.get("calibrate", True)),
            modifiers=_modifiers(payload),
            outside=_text(payload, "outside", "solid"),
        )
        layout_config = LayoutConfig(
            mode=_text(payload, "layout", "matrix"),
            gap=_num(payload, "gap", 4.0),
            fiducials=bool(payload.get("fiducials", True)),
        )
        style = SvgStyle(
            mode=_text(payload, "svg_mode", "stroke"),
            labels=bool(payload.get("labels", True)),
        )

        loaded = self._loaded(payload)
        request = SweepRequest(
            patterns=patterns, densities=densities,
            replicates=replicates, reference=reference,
        )
        if len(self._scales) > self._scale_cache_size:
            self._scales.clear()
        if len(self._layers) > self._scale_cache_size:
            self._layers.clear()
        return run_sweep(loaded, printer, request, config, layout_config, style,
                         scale_cache=self._scales, layer_cache=self._layers)


    def detail(self, payload: dict) -> dict:
        """One electrode's toolpaths, for the 3D view.

        The plate SVG carries everything the flat view needs, but stacking
        layers in 3D wants them as polylines with their heights, and the humps
        want their positions -- which the SVG has no way to express.
        """
        wanted = _text(payload, "variant", "")
        result = self._run(payload)
        variants = {v.ident: v for v in result.variants}
        if wanted not in variants:
            raise ValueError(f"no electrode called {wanted!r} on this plate")
        var = variants[wanted]
        printer = result.printer

        layers = []
        for layer, (_, top) in zip(result.layers_by_variant[var.index],
                                   result.layer_plan, strict=True):
            layers.append({
                "index": layer.index,
                "z": round(layer.z, 5),
                "top": round(top, 5),
                "pattern": layer.pattern,
                "density_pct": round(layer.density_actual * 100, 2),
                "perimeters": [_coords(ln) for ln in layer.perimeters],
                "infill": [_coords(ln) for ln in layer.infill],
                "solid": [_coords(ln) for ln in layer.solid],
                "crossings": [
                    [round(x, 4), round(y, 4), n]
                    for x, y, n in crossing_points(layer.infill)
                ],
            })
        return {
            "variant": var.ident,
            "label": var.label,
            "pattern": var.pattern,
            "overlap": var.overlap_class,
            "density_pct": round(var.density * 100, 1),
            "extents": [round(v, 4) for v in result.loaded.extents],
            "line_width": printer.line_width,
            "layer_height": printer.layer_height,
            "layers": layers,
        }

    def toolpaths(self, payload: dict) -> dict:
        """How the NOVA is likely to fill one electrode's GTL passes on one layer.

        The NOVA does not follow the strokes in a Gerber file: it fills the
        copper each file draws, with a tip of its own.  This hands back each
        pass's copper alongside concentric loops a `tip_um` wide filling it
        (see :func:`~electrode_slicer.passes.predict_fill`) and whatever they
        miss, which is where its slicer will have to improvise.
        """
        wanted = _text(payload, "variant", "")
        result = self._run(payload)
        variants = {v.ident: v for v in result.variants}
        if wanted not in variants:
            raise ValueError(f"no electrode called {wanted!r} on this plate")
        var = variants[wanted]
        per_layer = result.layers_by_variant[var.index]
        index = _num(payload, "layer", 0, int)
        if not 0 <= index < len(per_layer):
            raise ValueError(f"layer must be between 0 and {len(per_layer) - 1}")
        tip_um = _num(payload, "tip_um", 100.0)
        if not 10.0 <= tip_um <= 2000.0:
            raise ValueError("tip_um must be between 10 and 2000 microns")
        tip = tip_um / 1000.0
        width = result.printer.line_width

        passes = []
        for k, (role, area) in enumerate(pass_artwork(per_layer[index], width)):
            fill = predict_fill(area, tip)
            passes.append({
                "pass": k + 1,
                "role": role,
                "artwork": _rings(area),
                "loops": [_coords(ln) for ln in fill.loops],
                "single": fill.single,
                "gaps": _rings(fill.gaps),
                "area_mm2": round(fill.area, 4),
                "path_mm": round(fill.length, 2),
                "gap_pct": round(fill.gap_fraction * 100, 2),
            })
        return {
            "variant": var.ident,
            "label": var.label,
            "layer": index,
            "layers": len(per_layer),
            "tip_um": tip_um,
            "line_width": width,
            "extents": [round(v, 4) for v in result.loaded.extents],
            "passes": passes,
        }


def _rings(area) -> list[list[list[list[float]]]]:
    """An area as polygons, each a list of rings: the outline, then its holes."""
    return [[_coords(p.exterior), *(_coords(r) for r in p.interiors)]
            for p in as_polygons(area)]


def _coords(line) -> list[list[float]]:
    return [[round(x, 4), round(y, 4)] for x, y in line.coords]


# --------------------------------------------------------------------------
# reading the request
#
# The browser sends whatever is in the form.  An empty or unparseable number
# field arrives as null, so every one of these has to survive it and say which
# field was wrong -- "the slicer failed" helps nobody.
# --------------------------------------------------------------------------


def _pattern_name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"infill pattern must be a non-empty name, got {value!r}"
        )
    return value.strip()


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value, field: str, kind=float):
    if isinstance(value, bool | list | dict | tuple):
        raise ValueError(f"{field} must be a number, not a {type(value).__name__}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    if kind is int:
        if abs(out) > 1e6:
            raise ValueError(f"{field} is unreasonably large ({out:g})")
        return int(out)
    return out


def _num(payload: dict, key: str, default, kind=float):
    """A number with a fallback: a blank field means 'leave it alone'."""
    value = payload.get(key, default)
    return default if _blank(value) else _number(value, key, kind)


def _opt_float(payload: dict, key: str) -> float | None:
    """An override that may be absent, in which case the printer's own wins."""
    value = payload.get(key)
    return None if _blank(value) else _number(value, key)


def _text(payload: dict, key: str, default: str) -> str:
    value = payload.get(key, default)
    if _blank(value):
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text, not a {type(value).__name__}")
    return value


def _sequence(payload: dict, key: str, default: list) -> list:
    """A list that must not be empty once it has been asked for.

    Falling back to a default here would quietly slice something the user did
    not choose -- unticking every pattern would come back with a plate full of
    lines rather than an explanation.
    """
    if key not in payload or _blank(payload[key]):
        return list(default)
    value = payload[key]
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list, not a {type(value).__name__}")
    if not value:
        raise ValueError(f"select at least one {key.rstrip('s')}")
    return list(value)


def _triple(value, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field} must be three numbers, x y z")
    return tuple(_number(v, field) for v in value)


def _modifiers(payload: dict) -> tuple[Modifier, ...]:
    """The volumes the sweep is confined to; none means the whole part."""
    value = payload.get("modifiers")
    if _blank(value):
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"modifiers must be a list, not a {type(value).__name__}")
    if len(value) > MAX_MODIFIERS:
        raise ValueError(f"at most {MAX_MODIFIERS} modifiers")
    out = []
    for i, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise ValueError(f"modifier {i} must be an object")
        out.append(Modifier(
            shape=_text(item, "shape", "box"),
            center=_triple(item.get("center"), f"modifier {i} centre"),
            size=_triple(item.get("size"), f"modifier {i} size"),
        ))
    return tuple(out)


def _summary(result) -> list[dict]:
    """One row per distinct (pattern, density), matching the CLI's table."""
    rows = []
    for (pattern, density), per_layer in result.unique().items():
        top = per_layer[-1]
        swept = [lp for lp in per_layer if lp.density_target == density] or per_layer
        rows.append({
            "pattern": pattern,
            "overlap": next(p.overlap_class for p in all_patterns() if p.name == pattern),
            "target_pct": round(density * 100, 1),
            "actual_pct": round(sum(lp.density_actual for lp in swept) / len(swept) * 100, 1),
            "path_mm": round(sum(lp.length for lp in per_layer), 1),
            "crossings_per_mm2": round(top.overlap.crossings_per_mm2, 2),
            "crossings": sum(lp.overlap.crossings for lp in per_layer),
            "junctions": sum(lp.overlap.junctions for lp in per_layer),
            "overlap_pct": round(
                sum(lp.overlap.overlap_fraction for lp in per_layer) / len(per_layer) * 100, 1),
            "crossing_volume_nl": round(
                sum(lp.overlap.crossing_excess_volume_mm3 for lp in per_layer) * 1e3, 2),
            "hump_mm": round(top.overlap.hump_height_mm, 3),
            "islands": max(lp.overlap.islands for lp in per_layer),
            "dead_pct": round(
                (1 - min(lp.overlap.connected_fraction for lp in per_layer)) * 100, 1),
        })
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "electrode-slicer"
    # Keep the connection open after a response instead of hanging up, as
    # HTTP/1.0 does.  VS Code forwards a dev container's port through a relay
    # that exits the moment the server closes, dropping whatever it has not
    # yet passed on; when its pipe is small the page arrives cut short and its
    # script never runs.  With keep-alive the browser closes the connection,
    # and only once it has the whole response.
    protocol_version = "HTTP/1.1"
    service: SliceService

    def log_message(self, fmt, *args):  # quieter than the default
        if self.server.verbose:                                  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def log_refusal(self, code: int, reason: str) -> None:
        """Always report a rejected request.

        Successful requests are noise, but a refusal is the one thing the user
        needs to see, and hiding it behind --verbose means the browser shows a
        message the terminal cannot explain.
        """
        sys.stderr.write(
            f"{self.log_date_time_string()}  {code} {self.command} {self.path}  {reason}\n"
        )
        sys.stderr.flush()

    # -- helpers ---------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str,
              headers: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _static(self, name: str) -> None:
        # Resolve inside STATIC so a crafted path cannot escape the directory.
        target = (STATIC / name).resolve()
        if not target.is_file() or STATIC.resolve() not in target.parents:
            self.log_refusal(404, f"no static file {name!r}")
            self._json(404, {"error": f"{name} not found"})
            return
        kind = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), kind)

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._static("index.html")
        elif route == "/api/patterns":
            self._json(200, self.service.patterns())
        elif route == "/api/health":
            self._json(200, {"ok": True})
        elif route.startswith("/static/"):
            self._static(route[len("/static/"):])
        else:
            self.log_refusal(404, "unknown route")
            self._json(404, {"error": "not found"})

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        # A request refused before its body is read has to end the connection:
        # kept alive, the unread body would be parsed as the next request.
        route = self.path.split("?", 1)[0]
        if route not in ("/api/slice", "/api/detail", "/api/gerber", "/api/mesh",
                         "/api/toolpaths"):
            self.log_refusal(404, "unknown route")
            self.close_connection = True
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self.log_refusal(400, "empty request body")
            self.close_connection = True
            self._json(400, {"error": "empty request"})
            return
        if length > MAX_UPLOAD * 2:      # base64 inflates by about a third
            self.log_refusal(413, f"{length} byte body")
            self.close_connection = True
            self._json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self.log_refusal(400, f"malformed JSON: {exc}")
            self._json(400, {"error": "request body is not valid JSON"})
            return
        if not isinstance(payload, dict):
            self.log_refusal(400, f"body was a {type(payload).__name__}, not an object")
            self._json(400, {"error": "request body must be a JSON object"})
            return
        try:
            if route == "/api/gerber":
                name, body, kind = self.service.gerber(payload)
                self._send(200, body, kind,
                           {"Content-Disposition": f'attachment; filename="{name}"'})
                return
            work = {"/api/slice": self.service.slice, "/api/detail": self.service.detail,
                    "/api/mesh": self.service.mesh,
                    "/api/toolpaths": self.service.toolpaths}[route]
            self._json(200, work(payload))
        except (ValueError, KeyError, TypeError) as exc:
            # Everything the slicer rejects on purpose: a part that will not
            # fit, an unknown pattern, an impossible machine, a field the
            # browser sent as null because its input box was empty.
            reason = str(exc).strip("'")
            cause = exc.__cause__
            self.log_refusal(
                400, reason + (f"  [{type(cause).__name__}: {cause}]" if cause else "")
            )
            self._json(400, {"error": reason})
        except Exception:                                    # pragma: no cover
            self.log_refusal(500, "unhandled error -- traceback follows")
            traceback.print_exc()
            self._json(500, {"error": "the slicer failed; see the server log"})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
          verbose: bool = False):
    """Start the server and return it; the caller owns the lifetime."""
    handler = type("BoundHandler", (Handler,), {"service": SliceService()})
    socketserver.TCPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.verbose = verbose                                  # type: ignore[attr-defined]
    if open_browser:
        threading.Timer(0.4, webbrowser.open, (f"http://{host}:{httpd.server_address[1]}/",)).start()
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="electrode-slicer-web",
        description="Local web front end for the electrode slicer.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open a browser window")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    httpd = serve(args.host, args.port, args.open, args.verbose)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"electrode slicer running at {url}   (ctrl-c to stop)")
    print("rejected requests are logged here; --verbose logs every request")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
