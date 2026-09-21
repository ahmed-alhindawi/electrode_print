"""Running a whole plate: slice once, fill many ways, lay it out, write it.

This is the single pipeline behind both the command line and the web front
end, so what the browser previews is exactly what the CLI would print.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shapely import affinity

from .io import LoadedMesh, SvgStyle, build_manifest, build_svg
from .layout import Grid, LayoutConfig, Variant, build_variants, layout
from .patterns import get_pattern
from .printer import Printer
from .slicer import LayerPaths, SliceConfig, Slicer

__all__ = ["SweepRequest", "SweepResult", "run_sweep"]


@dataclass
class SweepRequest:
    """What to put on the plate."""

    patterns: list[str] = field(default_factory=lambda: ["lines"])
    densities: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    replicates: int = 1
    reference: int = 1

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("at least one infill pattern is required")
        for name in self.patterns:
            get_pattern(name)          # raises with a helpful message if unknown
        if not self.densities:
            raise ValueError("at least one infill density is required")
        for density in self.densities:
            if not 0.0 <= density <= 1.0:
                raise ValueError(f"density {density!r} must be a fraction between 0 and 1")


@dataclass
class SweepResult:
    """Everything a completed plate produced."""

    variants: list[Variant]
    layers_by_variant: list[list[LayerPaths]]
    layer_plan: list[tuple[float, float]]
    grid: Grid
    loaded: LoadedMesh
    printer: Printer
    config: SliceConfig
    layout_config: LayoutConfig
    style: SvgStyle
    warnings: list[str] = field(default_factory=list)

    @property
    def layer_zs(self) -> list[float]:
        return [z for z, _ in self.layer_plan]

    @property
    def n_layers(self) -> int:
        return len(self.layer_plan)

    @property
    def total_path_mm(self) -> float:
        return sum(lp.length for per in self.layers_by_variant for lp in per)

    def unique(self) -> dict[tuple[str, float], list[LayerPaths]]:
        """One entry per distinct (pattern, density), for reporting."""
        seen: dict[tuple[str, float], list[LayerPaths]] = {}
        for var in self.variants:
            seen.setdefault((var.pattern, var.density), self.layers_by_variant[var.index])
        return seen

    def svg(self, only_layer: int | None = None, title: str = "electrodes") -> str:
        return build_svg(
            self.printer, self.variants, self.layers_by_variant, self.layer_zs,
            self.style, self.loaded.width, self.grid, only_layer, title,
        )

    def manifest(self, source: str | Path = "") -> dict:
        return build_manifest(
            source, self.loaded, self.printer, self.config, self.layout_config,
            self.variants, self.layers_by_variant, self.layer_plan, self.style,
        )


def run_sweep(
    loaded: LoadedMesh,
    printer: Printer,
    request: SweepRequest,
    config: SliceConfig | None = None,
    layout_config: LayoutConfig | None = None,
    style: SvgStyle | None = None,
    scale_cache: dict | None = None,
    layer_cache: dict | None = None,
) -> SweepResult:
    """Slice the model once and fill it every way the request asks for.

    The two caches may be carried between calls -- the interactive front end
    does.  ``scale_cache`` saves re-solving a density that has been calibrated
    before; ``layer_cache`` saves rebuilding toolpaths that have not changed,
    so adjusting something presentational like the SVG mode redraws without
    touching the geometry.  Both keys cover every input, so neither can go
    stale when a setting changes.
    """
    config = config or SliceConfig()
    layout_config = layout_config or LayoutConfig()
    style = style or SvgStyle()
    warnings: list[str] = []

    part_w, part_d, part_h = loaded.extents
    printer.check_part(part_w, part_d, part_h)

    slicer = Slicer(printer, config, scale_cache)
    plan = slicer.layer_plan(part_h)
    if not plan:
        raise ValueError(
            f"the part is {part_h:.4f} mm tall, thinner than one "
            f"{printer.first_layer:g} mm layer"
        )
    zs = [z for z, _ in plan]
    slices = slicer.sections(loaded.mesh, zs)

    # Shift every cross section so the part's own bounding box starts at
    # (0, 0); placement then just adds the cell offset.
    minx, miny = loaded.mesh.bounds[0][0], loaded.mesh.bounds[0][1]
    slices = [affinity.translate(s, -minx, -miny) for s in slices]

    variants = build_variants(
        request.patterns, request.densities, request.replicates, request.reference
    )
    grid = layout(variants, (part_w, part_d), printer, layout_config, request.replicates)

    # Everything that decides what a layer's toolpaths look like.  Built
    # layers are immutable once returned, so they are safe to hand out to
    # several variants and to keep between runs.
    fingerprint = (
        printer.line_width, printer.layer_height, printer.resolution,
        config.perimeters, config.infill_angle, config.infill_overlap,
        config.alternate_angles, config.solid_layers, config.calibrate,
        config.density_metric, config.tolerance, config.tpms_resolution,
        config.stitch_ratio,
        tuple(round(z, 9) for z in zs),
        tuple(hash(s.wkb) for s in slices),
    )
    if layer_cache is None:
        layer_cache = {}

    # Identical (pattern, density) combinations share their toolpaths; only
    # the placement differs, so replicates cost nothing extra to generate.
    built: dict[tuple[str, float], list[LayerPaths]] = {}
    layers_by_variant: list[list[LayerPaths]] = []
    for var in variants:
        key = (var.pattern, var.density)
        if key not in built:
            cache_key = (fingerprint, *key)
            if cache_key not in layer_cache:
                layer_cache[cache_key] = slicer.build(slices, zs, var.pattern, var.density)
            built[key] = layer_cache[cache_key]
            for note in _check(var, built[key], config):
                if note not in warnings:      # one line per problem, not per cell
                    warnings.append(note)
        layers_by_variant.append(built[key])

    if all(lp.length == 0 for per in built.values() for lp in per):
        raise ValueError(
            f"no toolpaths were produced: the part is {part_w:.3f} x {part_d:.3f} x "
            f"{part_h:.3f} mm, too small for a {printer.line_width:g} mm bead. "
            f"Check the input units (read as {loaded.units!r}) and the scale."
        )

    return SweepResult(
        variants=variants, layers_by_variant=layers_by_variant, layer_plan=plan,
        grid=grid, loaded=loaded, printer=printer, config=config,
        layout_config=layout_config, style=style, warnings=warnings,
    )


def _check(var: Variant, per_layer: list[LayerPaths], config: SliceConfig) -> list[str]:
    """Problems worth telling the user about before they print the plate."""
    out: list[str] = []

    # A pattern that ignores the density is not a pattern that failed to hit
    # it.  Say so once, rather than once per density with alarming numbers.
    if var.pattern == "solid" and not var.is_reference and var.density < 0.999:
        # Carries no density, so the same note from every density in the sweep
        # collapses to a single line.
        return [
            "solid ignores the density sweep -- every solid electrode is the same "
            "100% pad, so sweeping it produces duplicates. Drop it from the pattern "
            "list; --reference already puts a 100% pad on the plate."
        ]

    stranded = max(lp.overlap.islands for lp in per_layer)
    if stranded > 1:
        dead = 1.0 - min(lp.overlap.connected_fraction for lp in per_layer)
        out.append(
            f"{var.pattern} at {var.density * 100:.0f}% breaks into {stranded} separate "
            f"blobs of ink, leaving {dead * 100:.0f}% of it not connected to a perimeter. "
            f"That share carries no current, so the electrode will behave as though it "
            f"were much sparser than {var.density * 100:.0f}%."
        )
    # Compare each layer against its own target: solid_layers deliberately
    # overrides the sweep value on the skin layers.
    worst = max(
        (abs(lp.density_actual - lp.density_target)
         for lp in per_layer if lp.region_area > 0),
        default=0.0,
    )
    if config.calibrate and worst > max(config.tolerance, 0.01):
        out.append(
            f"{var.pattern} cannot resolve {var.density * 100:.0f}% on this cross "
            f"section; closest is {worst * 100:+.1f}% off. Compare against the achieved "
            f"density in the manifest, not the requested one."
        )
    return out
