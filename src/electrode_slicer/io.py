"""Reading models in and writing plates out.

Input is an STL, from a path or from raw bytes so the web front end can hand
over an upload.  Output is an SVG in real millimetres plus a JSON manifest
recording what every electrode on the plate is and what it measured, or one
Gerber top-copper (.GTL) file per layer for tools that take PCB artwork.
"""

from __future__ import annotations

import io as _stdlib_io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import trimesh
from shapely import affinity
from shapely.geometry import Point, Polygon

from .geometry import (
    as_polygons,
    chain_paths,
    fit_arcs,
    order_paths,
    region_contours,
    tidy_area,
    toolpath_area,
)

__all__ = [
    "UNIT_SCALE",
    "LoadedMesh",
    "load_mesh",
    "load_mesh_bytes",
    "SvgStyle",
    "build_svg",
    "write_svg",
    "build_gerber",
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
    # A zero-length sliver clipped off a lattice reads as a closed ring of one
    # point.  Written out it is "M x,y L Z", which is invalid, and a browser
    # stops drawing the whole path -- every other bead with it -- at the error.
    if len(pts) < 2:
        return ""
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
        "solid": [affinity.translate(ln, var.dx, var.dy) for ln in layer.solid],
        "infill": [affinity.translate(ln, var.dx, var.dy) for ln in layer.infill],
    }
    for role, lines in roles.items():
        if not lines:
            continue
        # Solid fill outside the modifiers is plain conductor, not the pattern
        # under test, so it takes the wall colour.
        colour = style.infill_colour if role == "infill" else style.perimeter_colour
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
              title: str = "electrodes", fiducials=()) -> str:
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

    if fiducials:
        # Ink, unlike the annotations: print these with every layer and pass.
        lines.append('  <g id="fiducials" inkscape:groupmode="layer" '
                     'inkscape:label="alignment marks (print with every layer)">')
        for mark in fiducials:
            d = path_d(to_svg(mark.ring()).coords, True)
            cx, cy = to_svg(Point(mark.x, mark.y)).coords[0]
            lines.append(
                f'    <path class="fiducial" fill="none" stroke="{style.perimeter_colour}" '
                f'stroke-width="{fmt(printer.line_width)}" d="{d}"/>'
            )
            lines.append(
                f'    <circle class="fiducial" cx="{fmt(cx)}" cy="{fmt(cy)}" '
                f'r="{fmt(mark.dot / 2.0)}" fill="{style.perimeter_colour}"/>'
            )
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

        # The axis headings name every electrode in a matrix, so a caption
        # under each is only wanted on a packed plate, which has none.
        if grid is None or not (grid.col_labels or grid.row_labels):
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
# Gerber output
# --------------------------------------------------------------------------


def _gerber_xy(value: float) -> str:
    """A coordinate in the 4.6 millimetre format declared in the header."""
    return str(round(value * 1_000_000))


def build_gerber(printer, variants, layers_by_variant, layer_index: int,
                 z: float, title: str = "electrodes", strokes_of=None,
                 label: str = "", fiducials=(), regions_of=None) -> str:
    """One print layer as an RS-274X top-copper (.GTL) file.

    Gerber is flat, so a part more than one layer tall needs a file per layer.
    Every toolpath is drawn as a stroke with a round aperture the width of the
    bead -- the same centreline-plus-width the SVG's stroke mode writes, so
    the crossings survive rather than being unioned away.  Paths that meet end
    to end are joined into continuous strokes and put in nearest-next order,
    so a tool that plans its dispense from the artwork is not left starting
    and stopping at every fragment; the ink laid down is unchanged.  Curved
    runs -- a perimeter above all -- are written as circular arcs (G02/G03)
    rather than dozens of chords, to within the printer's resolution.  Gerber's Y already
    points to the back of the plate, so no flip is applied, and nothing that
    is not ink (plate outline, labels) is written.

    `strokes_of` picks which of a layer's toolpaths to draw -- one print pass
    of it, say (see :mod:`~electrode_slicer.passes`) -- and `label` names that
    pass in the header.  By default every toolpath is drawn.

    `regions_of` picks an area of a layer to write as a filled region
    (G36/G37) rather than as strokes -- the walls pass, which the NOVA fills
    with its own slicer whatever is drawn, and fills cleanly only when handed
    the area itself: a union of strokes has a scalloped edge wherever a fill
    line ends, and pinholes wherever beads meet, and every one of those turns
    into a kink or a stub in its toolpaths.  Holes are cut in (see
    :func:`~electrode_slicer.geometry.region_contours`).

    A region is written as plainly as a PCB tool writes a copper pour, with
    nothing finer than the bead can print.  Notches and slivers under half a
    bead are tidied away.  Where the outline truly is a circle -- the outer
    wall, a modifier's edge -- it is written as arcs, as a stroke is; the
    rest is straight segments simplified to a quarter of a bead, so the
    mesh's tessellation is not carried through facet by facet.

    `fiducials` -- the plate's alignment marks -- go into every file, the same
    each time, so each print can be registered on the marks the first one
    laid down (see :mod:`~electrode_slicer.fiducials`).  The ring is a stroke
    like any toolpath; the centre dot is a flash of a second, wider aperture.
    """
    if strokes_of is None:
        strokes_of = lambda layer: layer.all_paths  # noqa: E731
    # A comment ends at "*" and "%" opens an extended command, so neither may
    # leak in from a filename.
    title = title.replace("*", "").replace("%", "")
    out = [
        f"G04 {title}: {printer.name} layer {layer_index} z={z:.4f} mm"
        f"{' ' + label if label else ''}*",
        "%TF.GenerationSoftware,electrode_slicer*%",
        "%TF.FileFunction,Copper,L1,Top*%",
        "%TF.FilePolarity,Positive*%",
        "%FSLAX46Y46*%",
        "%MOMM*%",
        "%LPD*%",
        f"%ADD10C,{fmt(printer.line_width)}*%",
    ]
    if fiducials:
        out.append(f"%ADD11C,{fmt(fiducials[0].dot)}*%")
    out += ["D10*", "G75*", "G01*"]
    mode = "G01*"

    def stroke(pts, start: bool = True, straighten: float = 0.0) -> None:
        nonlocal mode
        if start:
            x0, y0 = pts[0]
            out.append(f"X{_gerber_xy(x0)}Y{_gerber_xy(y0)}D02*")
        here = pts[0]
        # A cut-in's bridge must retrace itself exactly, so never fit it.
        moves = (fit_arcs(pts, printer.resolution, straighten=straighten) if len(pts) > 2
                 else [("line", p) for p in pts[1:]])
        for move in moves:
            (x, y) = move[1]
            if move[0] == "line":
                if mode != "G01*":
                    out.append(mode := "G01*")
                out.append(f"X{_gerber_xy(x)}Y{_gerber_xy(y)}D01*")
            else:
                (cx, cy), ccw = move[2], move[3]
                if mode != ("G03*" if ccw else "G02*"):
                    out.append(mode := "G03*" if ccw else "G02*")
                out.append(f"X{_gerber_xy(x)}Y{_gerber_xy(y)}"
                           f"I{_gerber_xy(cx - here[0])}J{_gerber_xy(cy - here[1])}D01*")
            here = (x, y)

    if fiducials:
        out.append("G04 alignment marks: the same in every file*")
        for mark in fiducials:
            stroke(list(mark.ring().coords))
        out.append("D11*")
        for mark in fiducials:
            out.append(f"X{_gerber_xy(mark.x)}Y{_gerber_xy(mark.y)}D03*")
        out.append("D10*")

    strokes: dict[int, list] = {}      # replicates share their layers
    regions: dict[int, list] = {}
    for var in variants:
        layer = layers_by_variant[var.index][layer_index]
        if id(layer) not in strokes:
            strokes[id(layer)] = order_paths(chain_paths(strokes_of(layer)))
            if regions_of:
                bead = printer.line_width
                area = tidy_area(regions_of(layer), bead / 2.0)
                regions[id(layer)] = region_contours(area)
            else:
                regions[id(layer)] = []
        if not strokes[id(layer)] and not regions[id(layer)]:
            continue
        # Not "%": a viewer reads it as opening an extended command and skips
        # everything up to the next one, losing every other electrode.
        out.append(f"G04 {var.ident} {var.pattern} {var.density * 100:.1f} pct*")
        for contour in regions[id(layer)]:
            out.append("G36*")
            for n, (_, piece) in enumerate(contour):
                stroke([(x + var.dx, y + var.dy) for x, y in piece],
                       start=n == 0, straighten=printer.line_width / 4.0)
            out.append("G37*")
        for line in strokes[id(layer)]:
            pts = [(x + var.dx, y + var.dy) for x, y in line.coords]
            if len(pts) >= 2:
                stroke(pts)
    out.append("M02*")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def build_manifest(source, loaded: LoadedMesh, printer, config, layout_cfg,
                   variants, layers_by_variant, layer_plan, style, fiducials=()) -> dict:
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
            # The volumes the sweep was confined to, in the model's own frame.
            "modifiers": [
                {"shape": m.shape, "center_mm": list(m.center), "size_mm": list(m.size)}
                for m in config.modifiers
            ],
            "outside_modifiers": config.outside if config.modifiers else None,
        },
        "layout": {"mode": layout_cfg.mode, "gap_mm": layout_cfg.gap},
        # In plate coordinates, Gerber's frame (Y towards the back).
        "fiducials": [
            {"x_mm": round(f.x, 4), "y_mm": round(f.y, 4), "diameter_mm": f.diameter,
             "dot_mm": f.dot}
            for f in fiducials
        ],
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
