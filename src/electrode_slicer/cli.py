"""Command line front end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import SvgStyle, load_mesh, write_manifest, write_svg
from .layout import LayoutConfig
from .patterns import all_patterns, get_pattern, pattern_names
from .printer import PRINTERS, VOLTERA_NOVA, get_printer
from .slicer import SliceConfig
from .sweep import SweepRequest, run_sweep

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------
# argument types
# --------------------------------------------------------------------------


def parse_plate(text: str) -> tuple[float, float, float]:
    parts = text.lower().replace("*", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("plate must be given as WxDxH, e.g. 220x300x40")
    try:
        x, y, z = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad plate size {text!r}") from exc
    if min(x, y, z) <= 0:
        raise argparse.ArgumentTypeError("plate dimensions must be positive")
    return x, y, z


def parse_density(text: str) -> float:
    """Densities are percentages: 40 means 40%."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad density {text!r}") from exc
    if not 0.0 <= value <= 100.0:
        raise argparse.ArgumentTypeError("density must be a percentage between 0 and 100")
    return value / 100.0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from .io import UNIT_SCALE

    p = argparse.ArgumentParser(
        prog="slice-electrode",
        description="Slice an STL into SVG toolpaths for a dispensing printer, "
                    "sweeping infill pattern and density across the build plate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("stl", nargs="?", default="model/electrode.stl", type=Path,
                   help="input STL")
    p.add_argument("-o", "--output", type=Path, default=Path("build/electrodes.svg"),
                   help="output SVG")
    p.add_argument("--list-patterns", action="store_true",
                   help="print the infill patterns with their overlap behaviour and exit")

    g = p.add_argument_group("model")
    g.add_argument("--input-units", choices=("auto", *UNIT_SCALE), default="auto",
                   help="units the STL is authored in ('auto' guesses metres for "
                        "sub-1-unit parts)")
    g.add_argument("--scale", type=float, default=1.0,
                   help="extra scale factor applied after units")
    g.add_argument("--rotate-deg", type=float, default=0.0, help="rotate the part about Z")

    g = p.add_argument_group("printer")
    g.add_argument("--printer", choices=sorted(PRINTERS), default=VOLTERA_NOVA.name)
    g.add_argument("--plate", type=parse_plate, default=None,
                   help="override the build volume as WxDxH in mm")
    g.add_argument("--line-width", type=float, default=None,
                   help="dispensed bead width; also the SVG stroke width")
    g.add_argument("--layer-height", type=float, default=None)
    g.add_argument("--first-layer-height", type=float, default=None)
    g.add_argument("--margin", type=float, default=None, help="keep-out border on the plate")
    g.add_argument("--simplify", type=float, default=None, dest="resolution",
                   help="drop toolpath vertices finer than this, in mm (0 disables)")

    g = p.add_argument_group("slicing")
    g.add_argument("--slice-at", choices=("mid", "bottom", "top"), default="mid",
                   help="where in each layer to take the cross section")
    g.add_argument("--perimeters", type=int, default=2, help="wall loops per layer")
    g.add_argument("--solid-layers", type=int, default=0,
                   help="fully solid top and bottom layers (0 keeps every layer on the sweep)")

    g = p.add_argument_group("infill")
    g.add_argument("--infill-pattern", nargs="+", default=["lines"], choices=pattern_names(),
                   metavar="PATTERN", help=f"one or more of: {', '.join(pattern_names())}")
    g.add_argument("--infill-density", nargs="+", type=parse_density,
                   default=[0.20, 0.40, 0.60, 0.80],
                   metavar="PERCENT", help="one or more densities, in percent")
    g.add_argument("--infill-angle", type=float, default=45.0, help="base infill angle")
    g.add_argument("--infill-overlap", type=float, default=0.15,
                   help="how far open bead ends run into the walls, in line widths")
    g.add_argument("--alternate-angles", action=argparse.BooleanOptionalAction, default=True,
                   help="rotate line infill 90 degrees on alternate layers")
    g.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                   help="numerically hit the requested density instead of trusting "
                        "the formula")
    g.add_argument("--density-metric", choices=("area", "length"), default="area",
                   help="'area' measures real coverage; 'length' is faster but "
                        "over-reads lattices")
    g.add_argument("--tolerance", type=float, default=0.005,
                   help="calibration tolerance as a coverage fraction")
    g.add_argument("--stitch-ratio", type=float, default=None, metavar="N",
                   help="for the 'stitched' pattern: row spacings between cross-"
                        "stitches. Low packs in humps, high keeps the base compliant")
    g.add_argument("--flexibility", action="store_true",
                   help="also model each layer as a frame and report how readily "
                        "it bends (structure only; no material properties)")
    g.add_argument("--tpms-resolution", type=float, default=None,
                   help="sampling step for gyroid/schwarz-p (mm, default period/24)")

    g = p.add_argument_group("plate layout")
    g.add_argument("--layout", choices=("matrix", "pack"), default="matrix",
                   help="'matrix' puts one pattern per row and one density per column; "
                        "'pack' just fills the plate in the order given")
    g.add_argument("--gap", type=float, default=4.0, help="gap between neighbouring parts")
    g.add_argument("--columns", type=int, default=0,
                   help="force a column count (pack layout only; 0 = auto)")
    g.add_argument("--replicates", type=int, default=1, help="copies of each combination")
    g.add_argument("--reference", type=int, default=1, metavar="N",
                   help="fully dense (100%%) reference electrodes, on a row of their own")

    g = p.add_argument_group("output")
    g.add_argument("--svg-mode", choices=("stroke", "outline", "both"), default="stroke",
                   help="'stroke' emits toolpath centrelines; 'outline' emits "
                        "deposited areas")
    g.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True,
                   help="add labels in a separate non-print group")
    g.add_argument("--label-size", type=float, default=1.6, help="label font size in mm")
    g.add_argument("--fiducials", action=argparse.BooleanOptionalAction, default=True,
                   help="print three alignment marks round the electrodes, the same "
                        "in every layer, to register each print on the first")
    g.add_argument("--plate-outline", action=argparse.BooleanOptionalAction, default=True,
                   help="draw the plate border in a separate non-print group")
    g.add_argument("--split-layers", action="store_true",
                   help="also write one SVG per layer next to the main output")
    g.add_argument("--no-flip-y", action="store_true",
                   help="keep SVG +Y pointing down the plate instead of flipping to "
                        "plate coords")
    g.add_argument("--manifest", type=Path, default=None,
                   help="where to write the run manifest (default: output with .json)")
    g.add_argument("--quiet", "-q", action="store_true")
    return p


def print_patterns() -> None:
    print("Infill patterns, ordered by how much the toolpath crosses itself.\n")
    print(f"  {'pattern':<11s} {'overlap':<9s} description")
    for pattern in all_patterns():
        print(f"  {pattern.name:<11s} {pattern.overlap_class:<9s} {pattern.description}")
    print(
        "\n  none      no bead is laid over another; the surface stays flat.\n"
        "  junction  bead ends meet at lattice nodes; a modest pile-up.\n"
        "  crossing  the nozzle drives over earlier beads and leaves a discrete\n"
        "            hump at every intersection.\n"
        "\nRun the same densities across both ends of that axis to separate the\n"
        "effect of surface texture from the effect of bulk density."
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarise(result, say) -> None:
    """The table that says how each pattern differs, not just that it does."""
    rows = result.unique()
    if not rows:
        return
    say("")
    say("  X = bead-over-bead crossings (the humps under test); junc = bead ends"
        " meeting;\n  ovl% = all doubled-up footprint; Xvol = material piled at"
        " crossings, in nanolitres;\n  isl = separate blobs of ink"
        " and dead% = the share of the\n  ink not connected to a perimeter,"
        " which is the part that carries no current.")
    say(f"  {'pattern':<11s} {'overlap':<9s} {'target':>7s} {'actual':>7s} "
        f"{'path/mm':>8s} {'X/mm2':>7s} {'X tot':>6s} {'junc':>5s} "
        f"{'ovl%':>6s} {'Xvol/nL':>8s} {'hump/mm':>8s} {'isl':>4s} {'dead%':>6s}")
    for (pattern, density), per_layer in rows.items():
        top = per_layer[-1]
        n = len(per_layer)
        # Average only over the layers actually carrying the swept pattern, so
        # solid skin layers do not drag the figure towards 100%.
        swept = [lp for lp in per_layer if lp.density_target == density] or per_layer
        overlap_class = get_pattern(pattern).overlap_class
        say(f"  {pattern:<11s} {overlap_class:<9s} "
            f"{density * 100:6.1f}% "
            f"{sum(lp.density_actual for lp in swept) / len(swept) * 100:6.1f}% "
            f"{sum(lp.length for lp in per_layer):8.1f} "
            f"{top.overlap.crossings_per_mm2:7.2f} "
            f"{sum(lp.overlap.crossings for lp in per_layer):6d} "
            f"{sum(lp.overlap.junctions for lp in per_layer):5d} "
            f"{sum(lp.overlap.overlap_fraction for lp in per_layer) / n * 100:5.1f}% "
            f"{sum(lp.overlap.crossing_excess_volume_mm3 for lp in per_layer) * 1e3:8.2f} "
            f"{top.overlap.hump_height_mm:8.3f} "
            f"{max(lp.overlap.islands for lp in per_layer):4d} "
            f"{(1 - min(lp.overlap.connected_fraction for lp in per_layer)) * 100:5.0f}%")

    # At a matched density, a pattern that crosses itself lays material
    # without adding coverage, so it walks further.  Flag it: an impedance
    # difference between patterns that differ this much in deposited silver
    # is not attributable to surface texture alone.
    by_density: dict[float, list[float]] = {}
    for (_, density), per_layer in rows.items():
        by_density.setdefault(density, []).append(sum(lp.length for lp in per_layer))
    flagged = [
        (d, max(v) / min(v)) for d, v in sorted(by_density.items())
        if len(v) > 1 and min(v) > 0 and max(v) / min(v) > 1.15
    ]
    if flagged:
        say("")
        detail = ", ".join(f"{d * 100:.0f}%: {r:.2f}x" for d, r in flagged)
        say(f"  note    : ink used varies between patterns at the same density "
            f"({detail}).\n            Control for path_length_mm in the manifest, or "
            f"compare at a density\n            where the spread is small.")
    say("")
    if any(lp.flex.solved for per in rows.values() for lp in per):
        _flexibility_table(rows, say)


def _flexibility_table(rows, say) -> None:
    """Relative stiffness of each pattern, from its structure alone."""
    reference = None
    for per_layer in rows.values():
        if per_layer[-1].flex.solved:
            reference = per_layer[-1].flex
            break
    if reference is None:
        return
    say("  Structure-only stiffness, relative to the first pattern listed. Young's")
    say("  modulus and the bead section cancel out of a linear frame, so only the")
    say("  ratios mean anything. bend/stretch above 1 = more compliant than the")
    say("  reference; braced% = nodes where three or more beads fuse.")
    # Mark the best available trades.  Humps and compliance pull against each
    # other, so there is no single winner -- but anything another row beats on
    # both counts is simply worse, and can be dropped from consideration.
    scored = []
    for (pattern, density), per_layer in rows.items():
        flex = per_layer[-1].flex
        if flex.solved:
            scored.append((pattern, density, per_layer[-1].overlap.crossings_per_mm2,
                           flex.bend_compliance, flex))
    best = {
        (p, d) for p, d, x, b, _ in scored
        if not any(ox >= x and ob >= b and (ox, ob) != (x, b) for _, _, ox, ob, _ in scored)
    }

    say(f"  {'pattern':<11s} {'target':>7s} {'bend':>7s} {'stretch':>8s} "
        f"{'anis':>6s} {'braced':>7s} {'X/mm2':>7s}  best trade")
    for pattern, density, crossings, _, flex in scored:
        bend = flex.bend_compliance / reference.bend_compliance
        stretch = (flex.stretch_compliance / reference.stretch_compliance
                   if reference.stretch_compliance else 0.0)
        say(f"  {pattern:<11s} {density * 100:6.1f}% {bend:>7.2f} {stretch:>8.2f} "
            f"{flex.bend_anisotropy:>6.1f} {flex.braced_fraction * 100:>6.0f}% "
            f"{crossings:>7.2f}  {'*' if (pattern, density) in best else ''}")
    say("")
    say("  * nothing else in this sweep gives both more humps and more bend.")
    say("    Humps need crossings, crossings brace the network, and a braced")
    say("    network resists bending -- so the two cannot both be maximised.")
    say("")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_patterns:
        print_patterns()
        return 0

    def say(*parts) -> None:
        if not args.quiet:
            print(*parts, file=sys.stderr)

    printer = get_printer(args.printer)
    plate = args.plate
    printer = printer.with_overrides(
        plate_x=plate[0] if plate else None,
        plate_y=plate[1] if plate else None,
        z_travel=plate[2] if plate else None,
        line_width=args.line_width,
        layer_height=args.layer_height,
        first_layer_height=args.first_layer_height,
        margin=args.margin,
        resolution=args.resolution,
    )

    config = SliceConfig(
        perimeters=args.perimeters,
        infill_angle=args.infill_angle,
        infill_overlap=args.infill_overlap,
        alternate_angles=args.alternate_angles,
        solid_layers=args.solid_layers,
        slice_at=args.slice_at,
        calibrate=args.calibrate,
        density_metric=args.density_metric,
        tolerance=args.tolerance,
        tpms_resolution=args.tpms_resolution,
        flexibility=args.flexibility,
        stitch_ratio=args.stitch_ratio,
    )
    layout_config = LayoutConfig(
        mode=args.layout, gap=args.gap, columns=args.columns,
        labels=args.labels, label_size=args.label_size, fiducials=args.fiducials,
    )
    style = SvgStyle(
        mode=args.svg_mode, labels=args.labels, label_size=args.label_size,
        plate_outline=args.plate_outline, flip_y=not args.no_flip_y,
    )
    if style.mode == "outline":
        say("warning : --svg-mode outline unions the beads into flat regions, which "
            "discards\n          the crossings this sweep is measuring; use 'stroke' "
            "to keep them.")

    try:
        loaded = load_mesh(args.stl, args.input_units, args.scale, args.rotate_deg)
        say(f"model   : {args.stl} ({loaded.units}"
            f"{' guessed' if loaded.guessed else ''}, x{loaded.scale:g}) -> "
            f"{loaded.width:.3f} x {loaded.depth:.3f} x {loaded.height:.3f} mm")

        request = SweepRequest(
            patterns=args.infill_pattern, densities=args.infill_density,
            replicates=args.replicates, reference=args.reference,
        )
        result = run_sweep(loaded, printer, request, config, layout_config, style)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc).strip("'")) from exc

    say(f"slicing : {result.n_layers} layer(s) at {printer.layer_height:g} mm "
        f"(first {printer.first_layer:g} mm), section taken at layer {config.slice_at}")
    shape = (f"{result.grid.cols} across x {result.grid.rows} up"
             if layout_config.mode == "matrix" else
             f"{result.grid.cols} x {result.grid.rows}")
    say(f"layout  : {len(result.variants)} electrode(s), {layout_config.mode} {shape}, "
        f"cells of {result.grid.cell_w:.2f} x {result.grid.cell_h:.2f} mm on a "
        f"{printer.plate_x:g} x {printer.plate_y:g} mm plate")
    for warning in result.warnings:
        say(f"warning : {warning}")

    summarise(result, say)

    write_svg(args.output, result.svg(title=args.output.stem))
    say(f"wrote   : {args.output}")

    if args.split_layers:
        for idx in range(result.n_layers):
            part = args.output.with_name(
                f"{args.output.stem}_layer{idx:03d}{args.output.suffix}"
            )
            write_svg(part, result.svg(only_layer=idx, title=part.stem))
            say(f"wrote   : {part}")

    manifest_path = args.manifest or args.output.with_suffix(".json")
    write_manifest(manifest_path, result.manifest(args.stl))
    say(f"wrote   : {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
