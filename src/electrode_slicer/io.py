"""Reading models in and writing plates out.

Input is an STL, from a path or from raw bytes so the web front end can hand
over an upload.  Output is an SVG in real millimetres plus a JSON manifest
recording what every electrode on the plate is and what it measured.
"""

from __future__ import annotations

import io as _stdlib_io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import trimesh
from shapely import affinity
from shapely.geometry import Polygon

from .geometry import as_polygons, toolpath_area

__all__ = [
    "UNIT_SCALE",
    "LoadedMesh",
    "load_mesh",
    "load_mesh_bytes",
    "SvgStyle",
    "build_svg",
    "write_svg",
    "build_manifest",
    "write_manifest",
]

UNIT_SCALE = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "mil": 0.0254}


# --------------------------------------------------------------------------
# STL input
# --------------------------------------------------------------------------


@dataclass
class LoadedMesh:
    """A mesh in millimetres, sitting on Z=0 with its bbox corner at origin."""

    mesh: object
    units: str          # the unit it was interpreted as
    scale: float        # the extra scale factor applied on top
    guessed: bool       # True when the unit was inferred rather than given

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.mesh.extents)

    @property
    def width(self) -> float:
        return self.extents[0]

    @property
    def depth(self) -> float:
        return self.extents[1]

    @property
    def height(self) -> float:
        return self.extents[2]


def guess_units(mesh) -> str:
    """Nothing in an STL records its unit, so infer one from the size.

    A part whose longest side is under 1 unit is almost certainly metres,
    exported by a CAD tool that works in SI; anything else is taken at face
    value as millimetres.
    """
    return "m" if float(mesh.extents.max()) < 1.0 else "mm"


def _prepare(mesh, units: str, extra_scale: float, rotate_deg: float) -> LoadedMesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("no geometry found in the model")
    if extra_scale <= 0:
        raise ValueError("scale must be positive")

    guessed = units == "auto"
    chosen = guess_units(mesh) if guessed else units
    if chosen not in UNIT_SCALE:
        raise ValueError(f"unknown unit {chosen!r}; known: {', '.join(UNIT_SCALE)}")
    mesh.apply_scale(UNIT_SCALE[chosen] * extra_scale)

    if rotate_deg:
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(math.radians(rotate_deg), [0, 0, 1])
        )

    lower = mesh.bounds[0]
    mesh.apply_translation([-lower[0], -lower[1], -lower[2]])
    return LoadedMesh(mesh=mesh, units=chosen, scale=extra_scale, guessed=guessed)


def load_mesh(path: str | Path, units: str = "auto", scale: float = 1.0,
              rotate_deg: float = 0.0) -> LoadedMesh:
    """Load an STL from disk, in millimetres and sitting on the plate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file")
    return _prepare(trimesh.load_mesh(path, process=True), units, scale, rotate_deg)


def load_mesh_bytes(data: bytes, units: str = "auto", scale: float = 1.0,
                    rotate_deg: float = 0.0, file_type: str = "stl") -> LoadedMesh:
    """Same, from raw bytes -- what an upload from the web front end gives."""
    if not data:
        raise ValueError("empty model upload")
    mesh = trimesh.load_mesh(_stdlib_io.BytesIO(data), file_type=file_type, process=True)
    return _prepare(mesh, units, scale, rotate_deg)


# --------------------------------------------------------------------------
# SVG output
# --------------------------------------------------------------------------


@dataclass
class SvgStyle:
    """How the plate is drawn.  Not machine settings -- presentation only."""

    mode: str = "stroke"          # "stroke", "outline" or "both"
    perimeter_colour: str = "#1f4e79"
    infill_colour: str = "#c0392b"
    plate_colour: str = "#9aa0a6"
    label_colour: str = "#3c4043"
    labels: bool = True
    label_size: float = 1.6
    plate_outline: bool = True
    flip_y: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("stroke", "outline", "both"):
            raise ValueError("svg mode must be 'stroke', 'outline' or 'both'")


def fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def path_d(coords, close: bool) -> str:
    pts = list(coords)
    if len(pts) < 2:
        return ""
    if close and pts[0] == pts[-1]:
        pts = pts[:-1]
    head = f"M {fmt(pts[0][0])},{fmt(pts[0][1])}"
    tail = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts[1:])
    return f"{head} L {tail}" + (" Z" if close else "")


def polygon_d(poly: Polygon) -> str:
    parts = [path_d(poly.exterior.coords, True)]
    parts.extend(path_d(ring.coords, True) for ring in poly.interiors)
    return " ".join(p for p in parts if p)


def flipper(plate_h: float, flip: bool):
    """Map plate coordinates to SVG coordinates.

    The flip is baked into the coordinates rather than applied as a group
    transform, because CAM importers routinely ignore transforms.
    """
    if not flip:
        return lambda geom: geom
    return lambda geom: affinity.scale(
        affinity.translate(geom, 0, -plate_h), xfact=1.0, yfact=-1.0, origin=(0, 0)
    )


def _variant_group(var, layer, printer, style: SvgStyle, to_svg, indent: str) -> list[str]:
    if not layer.all_paths:
        return []
    out = [
        f'{indent}<g id="{var.ident}" data-pattern="{var.pattern}" '
        f'data-density="{var.density * 100:.1f}" '
        f'data-overlap="{var.overlap_class}" '
        f'data-crossings="{layer.overlap.crossings}">'
    ]
    roles = {
        "perimeter": [affinity.translate(ln, var.dx, var.dy) for ln in layer.perimeters],
        "infill": [affinity.translate(ln, var.dx, var.dy) for ln in layer.infill],
    }
    for role, lines in roles.items():
        if not lines:
            continue
        colour = style.perimeter_colour if role == "perimeter" else style.infill_colour
        if style.mode in ("stroke", "both"):
            ds = [d for d in (path_d(to_svg(ln).coords, ln.is_ring) for ln in lines) if d]
            if ds:
                out.append(
                    f'{indent}  <path class="{role}" fill="none" stroke="{colour}" '
                    f'stroke-width="{fmt(printer.line_width)}" stroke-linecap="round" '
                    f'stroke-linejoin="round" d="{" ".join(ds)}"/>'
                )
        if style.mode in ("outline", "both"):
            solid = toolpath_area(lines, printer.line_width)
            ds = [d for d in (polygon_d(to_svg(p)) for p in as_polygons(solid)) if d]
            if ds:
                opacity = ' fill-opacity="0.35"' if style.mode == "both" else ""
                out.append(
                    f'{indent}  <path class="{role}-outline" fill="{colour}"'
                    f'{opacity} fill-rule="evenodd" stroke="none" d="{" ".join(ds)}"/>'
                )
    out.append(f"{indent}</g>")
    return out


def build_svg(printer, variants, layers_by_variant, layer_zs, style: SvgStyle,
              part_w: float, grid=None, only_layer: int | None = None,
              title: str = "electrodes") -> str:
    """Render the whole plate as an SVG document, in millimetres."""
    plate_x, plate_y, plate_z = printer.plate
    to_svg = flipper(plate_y, style.flip_y)
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{fmt(plate_x)}mm" height="{fmt(plate_y)}mm" '
        f'viewBox="0 0 {fmt(plate_x)} {fmt(plate_y)}" '
        f'data-units="mm" data-printer="{printer.name}" '
        f'data-plate="{fmt(plate_x)}x{fmt(plate_y)}x{fmt(plate_z)}">',
        f"  <title>{title}</title>",
        f"  <desc>{printer.name} toolpaths, 1 user unit = 1 mm, line width "
        f"{fmt(printer.line_width)} mm, layer height {fmt(printer.layer_height)} mm.</desc>",
    ]

    if style.plate_outline:
        lines.append('  <g id="build-plate" inkscape:groupmode="layer" '
                     'inkscape:label="build-plate (do not print)">')
        lines.append(
            f'    <rect x="0" y="0" width="{fmt(plate_x)}" height="{fmt(plate_y)}" '
            f'fill="none" stroke="{style.plate_colour}" stroke-width="0.3" '
            f'stroke-dasharray="2 2"/>'
        )
        lines.append("  </g>")

    for idx, z in enumerate(layer_zs):
        if only_layer is not None and idx != only_layer:
            continue
        body: list[str] = []
        for var in variants:
            body.extend(_variant_group(
                var, layers_by_variant[var.index][idx], printer, style, to_svg, "    "
            ))
        if not body:
            continue
        lines.append(
            f'  <g id="layer_{idx:03d}" inkscape:groupmode="layer" '
            f'inkscape:label="layer {idx} (z={z:.4f} mm)" data-z="{z:.6f}">'
        )
        lines.extend(body)
        lines.append("  </g>")

    if style.labels:
        lines.append('  <g id="annotations" inkscape:groupmode="layer" '
                     'inkscape:label="annotations (do not print)" '
                     f'fill="{style.label_colour}" font-family="monospace" '
                     f'font-size="{fmt(style.label_size)}">')

        def place(px: float, py: float, text: str, anchor: str = "middle") -> str:
            sx, sy = (px, plate_y - py) if style.flip_y else (px, py)
            return (f'    <text x="{fmt(sx)}" y="{fmt(sy)}" text-anchor="{anchor}">'
                    f"{text}</text>")

        for var in variants:
            lines.append(place(
                var.dx + part_w / 2.0, var.dy - style.label_size * 0.6, var.label
            ))
        if grid is not None:
            # Axis headings, so the plate can be read without the manifest.
            for x, text in grid.col_labels:
                lines.append(place(x, grid.col_label_y, text))
            for y, text in grid.row_labels:
                lines.append(place(grid.x0 - style.label_size * 0.6, y, text, anchor="end"))
        lines.append("  </g>")

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_svg(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def build_manifest(source, loaded: LoadedMesh, printer, config, layout_cfg,
                   variants, layers_by_variant, layer_plan, style) -> dict:
    """A record of what is on the plate and what each electrode measured."""
    manifest = {
        "source": str(source),
        "printer": {
            "name": printer.name,
            "plate_mm": list(printer.plate),
            "line_width_mm": printer.line_width,
            "layer_height_mm": printer.layer_height,
            "first_layer_height_mm": printer.first_layer,
            "margin_mm": printer.margin,
            "resolution_mm": printer.resolution,
        },
        "model": {
            "input_units": loaded.units,
            "units_guessed": loaded.guessed,
            "scale": loaded.scale,
            "extents_mm": list(loaded.extents),
        },
        "slicing": {
            "perimeters": config.perimeters,
            "infill_angle_deg": config.infill_angle,
            "infill_overlap": config.infill_overlap,
            "alternate_angles": config.alternate_angles,
            "solid_layers": config.solid_layers,
            "slice_at": config.slice_at,
            "calibrated": config.calibrate,
            "density_metric": config.density_metric,
            "tolerance": config.tolerance,
        },
        "layout": {"mode": layout_cfg.mode, "gap_mm": layout_cfg.gap},
        "svg_mode": style.mode,
        "layers": [{"index": i, "slice_z_mm": z, "top_z_mm": t}
                   for i, (z, t) in enumerate(layer_plan)],
        "variants": [],
    }
    for var in variants:
        per_layer = layers_by_variant[var.index]
        total_len = sum(lp.length for lp in per_layer)
        manifest["variants"].append({
            "id": var.ident,
            "label": var.label,
            "pattern": var.pattern,
            "overlap_class": var.overlap_class,
            "density_target_pct": round(var.density * 100, 4),
            "replicate": var.replicate,
            "is_reference": var.is_reference,
            "cell": [var.col, var.row],
            "origin_mm": [round(var.dx, 4), round(var.dy, 4)],
            "path_length_mm": round(total_len, 3),
            "deposited_volume_mm3": round(
                total_len * printer.line_width * printer.layer_height, 5
            ),
            # Rolled up for the impedance regression: total humps, how thickly
            # they are fed, and what the skin-facing top layer looks like.
            "max_islands": max(lp.overlap.islands for lp in per_layer),
            "dead_ink_fraction": round(
                1.0 - min(lp.overlap.connected_fraction for lp in per_layer), 5),
            "crossings_total": sum(lp.overlap.crossings for lp in per_layer),
            "crossings_top_layer": per_layer[-1].overlap.crossings,
            "crossings_per_mm2_top_layer": round(
                per_layer[-1].overlap.crossings_per_mm2, 4),
            "hump_height_top_layer_mm": round(per_layer[-1].overlap.hump_height_mm, 5),
            "excess_volume_mm3": round(
                sum(lp.overlap.excess_volume_mm3 for lp in per_layer), 6),
            "crossing_excess_volume_mm3": round(
                sum(lp.overlap.crossing_excess_volume_mm3 for lp in per_layer), 6),
            "layers": [{
                "index": lp.index,
                "z_mm": round(lp.z, 6),
                "pattern": lp.pattern,
                "slice_area_mm2": round(lp.slice_area, 5),
                "infill_region_area_mm2": round(lp.region_area, 5),
                "density_target_pct": round(lp.density_target * 100, 4),
                "density_actual_pct": round(lp.density_actual * 100, 3),
                "infill_scale_mm": round(lp.scale, 5),
                "path_length_mm": round(lp.length, 3),
                "islands": lp.overlap.islands,
                "connected_fraction": round(lp.overlap.connected_fraction, 5),
                "crossings": lp.overlap.crossings,
                "crossings_per_mm2": round(lp.overlap.crossings_per_mm2, 4),
                "junctions": lp.overlap.junctions,
                "max_multiplicity": lp.overlap.max_multiplicity,
                "mean_multiplicity": round(lp.overlap.mean_multiplicity, 4),
                "overlap_area_mm2": round(lp.overlap.overlap_area_mm2, 5),
                "overlap_fraction": round(lp.overlap.overlap_fraction, 5),
                "excess_volume_mm3": round(lp.overlap.excess_volume_mm3, 6),
                "crossing_excess_volume_mm3": round(
                    lp.overlap.crossing_excess_volume_mm3, 6),
                "hump_height_mm": round(lp.overlap.hump_height_mm, 5),
                **({"flexibility": {
                    "solved": lp.flex.solved,
                    "bend_compliance": lp.flex.bend_compliance,
                    "bend_compliance_x": lp.flex.bend_compliance_x,
                    "bend_compliance_y": lp.flex.bend_compliance_y,
                    "bend_anisotropy": round(lp.flex.bend_anisotropy, 4),
                    "stretch_compliance": lp.flex.stretch_compliance,
                    "braced_fraction": round(lp.flex.braced_fraction, 5),
                    "orientation_anisotropy": round(lp.flex.anisotropy, 5),
                    "nodes": lp.flex.nodes,
                    "elements": lp.flex.elements,
                }} if lp.flex.solved else {}),
            } for lp in per_layer],
        })
    return manifest


def write_manifest(path: str | Path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
