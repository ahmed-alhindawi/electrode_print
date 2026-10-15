#!/usr/bin/env python3
"""Regenerate every figure in docs/methods.md.

    uv run python docs/make_figures.py

Each figure is one function.  They share a cached sweep so the whole set
rebuilds in about a minute.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point

from electrode_slicer import Printer, SweepRequest, load_mesh, run_sweep
from electrode_slicer.mechanics import _assemble, _solve, build_network
from electrode_slicer.metrics import crossing_points
from electrode_slicer.patterns import FillContext, get_pattern
from electrode_slicer.slicer import SliceConfig, Slicer

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Palette.  Categorical slots are used in fixed order and never cycled; charts
# that need all-pairs separation (the scatter) use only the first three.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
        "#008300", "#4a3aa7", "#e34948"]
# Domain colours for geometry drawings, which depict parts rather than series.
WALL, FILL, HUMP = "#1f4e79", "#c0392b", "#e8a33d"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.8, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200,
})

PRINTER = Printer(name="voltera-nova", plate_x=220.0, plate_y=300.0, z_travel=40.0,
                  line_width=0.1, layer_height=0.1)
MODEL = Path(__file__).resolve().parents[1] / "model" / "electrode.stl"
_SCALES: dict = {}
_LAYERS: dict = {}
_LOADED = None


def electrode():
    global _LOADED
    if _LOADED is None:
        _LOADED = load_mesh(MODEL)
    return _LOADED


def layers_for(pattern: str, density: float, flexibility: bool = False,
               stitch_ratio: float | None = None):
    """One electrode's layers, cached across figures."""
    result = run_sweep(
        electrode(), PRINTER,
        SweepRequest(patterns=[pattern], densities=[density], reference=0),
        SliceConfig(flexibility=flexibility, stitch_ratio=stitch_ratio),
        scale_cache=_SCALES, layer_cache=_LAYERS,
    )
    return result.layers_by_variant[0]


def draw_paths(ax, layer, humps=False, lw=0.9):
    for line in layer.perimeters:
        ax.plot(*line.xy, color=WALL, lw=lw, solid_capstyle="round")
    for line in layer.infill:
        ax.plot(*line.xy, color=FILL, lw=lw, solid_capstyle="round")
    if humps:
        pts = crossing_points(layer.infill)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                    color=HUMP, ms=2.4, mec=SURFACE, mew=0.3, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")


def finish(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(HERE.parent)}")


# ---------------------------------------------------------------------------


def fig_pattern_library():
    """Every fill at one density, grouped by how it treats its own beads."""
    groups = [
        ("no crossings", ["lines", "serpentine", "hilbert", "gyroid"]),
        ("bead ends meet", ["honeycomb"]),
        ("beads cross", ["stitched", "grid", "triangles", "stars"]),
    ]
    names = [n for _, group in groups for n in group]
    fig, axes = plt.subplots(1, len(names), figsize=(11.5, 1.9))
    for ax, name in zip(axes, names, strict=True):
        layer = layers_for(name, 0.4)[-1]
        draw_paths(ax, layer, humps=True, lw=0.7)
        crossings = len(crossing_points(layer.infill))
        ax.set_title(f"{name}\n{crossings} humps", fontsize=7.5, color=INK, pad=3)
    # One bracket per group, under the panels it covers.
    start = 0
    for label, group in groups:
        span = axes[start].get_position().x0, axes[start + len(group) - 1].get_position().x1
        mid = (span[0] + span[1]) / 2
        fig.lines.append(plt.Line2D([span[0], span[1]], [0.02, 0.02],
                                    transform=fig.transFigure, color=AXIS, lw=0.8))
        fig.text(mid, -0.03, label, ha="center", va="top", fontsize=7.5, color=INK2)
        start += len(group)
    fig.suptitle("Infill patterns at 40% density, ordered by how often a bead "
                 "crosses one already laid down", fontsize=9, color=INK, y=1.14)
    finish(fig, "01-pattern-library.png")


def fig_hump_measurement():
    """What a hump is, and how the stack is read."""
    fig = plt.figure(figsize=(11.5, 3.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.08)

    # A: the two cases side by side, on schematic beads.
    ax = fig.add_subplot(grid[0])
    # left: a crossing -- one bead passes over another, so material stacks
    ax.plot([0, 8], [7, 7], color=FILL, lw=8, solid_capstyle="round")
    ax.plot([4, 4], [3, 11], color=FILL, lw=8, solid_capstyle="round",
            path_effects=None)
    ax.plot([4], [7], "o", color=HUMP, ms=11, mec=SURFACE, mew=1.6, zorder=5)
    ax.text(4, 1.4, "crossing\none bead over another,\nso it stands proud",
            fontsize=7.5, color=INK2, ha="center", va="top")
    # right: a junction -- three ends arrive, all at the same height
    for angle in (90, 210, 330):
        rad = math.radians(angle)
        ax.plot([15 + 4 * math.cos(rad), 15], [7 + 4 * math.sin(rad), 7],
                color=FILL, lw=8, solid_capstyle="round")
    ax.text(15, 1.4, "junction\nthree ends meet,\nnothing stacks",
            fontsize=7.5, color=INK2, ha="center", va="top")
    ax.set_xlim(-2, 21); ax.set_ylim(-5, 12.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("A  only the left case is counted", loc="left",
                 fontsize=8.5, color=INK)

    # B: the same test applied to a real fill.
    ax = fig.add_subplot(grid[1])
    layer = layers_for("grid", 0.4)[-1]
    draw_paths(ax, layer, humps=True, lw=1.1)
    ax.set_title(f"B  grid at 40%: {len(crossing_points(layer.infill))} humps found",
                 loc="left", fontsize=8.5, color=INK)

    # C: the stack, with the exposed humps standing proud.
    ax = fig.add_subplot(grid[2])
    per_layer = layers_for("grid", 0.4)
    zex, k, az, el = 26, 1.0, -0.6, 0.95
    ex, ey, ez = electrode().extents

    def to(x, y, z):
        ca, sa = math.cos(az), math.sin(az)
        rx, ry = (x - ex / 2) * ca - (y - ey / 2) * sa, (x - ex / 2) * sa + (y - ey / 2) * ca
        zz = (z - ez / 2) * zex
        ce, se = math.cos(el), math.sin(el)
        return rx * k, (ry * se - zz * ce) * k, ry * ce + zz * se

    skin = per_layer[-1].index
    items = []
    for lp in per_layer:
        top_z = lp.z + PRINTER.layer_height / 2
        for role, paths in (("wall", lp.perimeters), ("fill", lp.infill)):
            for line in paths:
                proj = [to(x, y, top_z) for x, y in line.coords]
                items.append((sum(p[2] for p in proj) / len(proj), role, proj))
        if lp.index == skin:
            for x, y, n in crossing_points(lp.infill):
                foot, head = to(x, y, top_z), to(x, y, top_z + PRINTER.layer_height * (n - 1))
                items.append((head[2], "hump", [foot, head]))
    for _, role, proj in sorted(items, key=lambda t: t[0]):
        if role == "hump":
            ax.plot([proj[0][0], proj[1][0]], [proj[0][1], proj[1][1]],
                    color=HUMP, lw=0.7, zorder=3)
            ax.plot(proj[1][0], proj[1][1], "o", color=HUMP, ms=2.6, mec="none", zorder=4)
        else:
            ax.plot([p[0] for p in proj], [p[1] for p in proj],
                    color=WALL if role == "wall" else FILL, lw=0.8, solid_capstyle="round")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    ax.set_title("C  the printed stack, Z exaggerated 26x", loc="left",
                 fontsize=8.5, color=INK)
    ax.text(0.5, -0.02, "only the top layer's humps are exposed; the rest are buried",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)
    finish(fig, "02-hump-measurement.png")


def fig_calibration():
    """Why a density means one thing across patterns."""
    region = Point(0, 0).buffer(2.0, quad_segs=64)
    slicer = Slicer(PRINTER, SliceConfig())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    shown = ["lines", "grid", "stars"]      # three slots: all-pairs safe
    for i, name in enumerate(shown):
        pattern = get_pattern(name)
        scales = np.geomspace(0.1, 1.6, 26)
        cover = [slicer.measure(pattern, region, s, 0.0, 0.05) * 100 for s in scales]
        ax1.plot(scales, cover, color=SLOT[i], lw=1.8, label=name)
        solved, got = slicer.solve_scale(pattern, region, 0.40, 0.0, 0.05)
        ax1.plot([solved], [got * 100], "o", color=SLOT[i], ms=8,
                 mec=SURFACE, mew=1.5, zorder=5)
    ax1.axhline(40, color=AXIS, lw=0.8, ls="-")
    ax1.text(0.105, 42, "requested 40%", fontsize=7.5, color=INK2)
    ax1.set_xscale("log")
    ax1.set_xlabel("scale knob: bead spacing, cell size or period (mm)")
    ax1.set_ylabel("measured coverage (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", lw=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=7.5, labelcolor=INK2)
    ax1.set_title("A  each pattern is driven to the same measured coverage",
                  loc="left", fontsize=8.5, color=INK)

    names = ["lines", "serpentine", "hilbert", "gyroid", "honeycomb",
             "stitched", "grid", "triangles", "stars"]
    targets = [20, 40, 60, 80]
    errors = []
    for name in names:
        for t in targets:
            layer = layers_for(name, t / 100)[-1]
            errors.append((layer.density_actual * 100) - t)
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.axhspan(-0.5, 0.5, color=SLOT[0], alpha=0.10, lw=0)
    xs = np.arange(len(names))
    for j, _t in enumerate(targets):
        vals = [errors[i * len(targets) + j] for i in range(len(names))]
        ax2.plot(xs + (j - 1.5) * 0.16, vals, "o", ms=5, color=SLOT[0],
                 mec=SURFACE, mew=0.8, alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("achieved − requested (percentage points)")
    ax2.set_ylim(-1, 1)
    ax2.grid(axis="y", lw=0.8)
    ax2.set_axisbelow(True)
    ax2.text(0.02, 0.93, "shaded band = ±0.5 pt tolerance", transform=ax2.transAxes,
             fontsize=7.5, color=INK2)
    ax2.set_title("B  residual at 20 / 40 / 60 / 80%", loc="left",
                  fontsize=8.5, color=INK)
    finish(fig, "03-density-calibration.png")


def _rejected_fills(region):
    """Rebuild the two fills that were dropped, so the evidence can be shown.

    Neither ships any more -- that is the point of the figure -- so they are
    reconstructed here from the same definitions the package used.
    """
    import numpy as np
    from electrode_slicer.geometry import as_polygons, ring_lines
    from electrode_slicer.patterns.tpms import TpmsPattern

    width = PRINTER.line_width

    def nested_rings(spacing):
        out, current = [], region.buffer(-width / 2)
        for _ in range(200):
            polys = [q for q in as_polygons(current)
                     if not q.buffer(-width / 2).is_empty]
            if not polys:
                break
            out.extend(ring_lines(polys))
            from shapely.ops import unary_union
            current = unary_union(polys).buffer(-spacing, join_style="mitre")
        return out

    class SchwarzP(TpmsPattern):
        name = "schwarz-p (rebuilt for this figure)"
        description = name

        def field(self, u, v, w):
            return np.cos(u) + np.cos(v) + math.cos(w)

    schwarz = SchwarzP().generate(
        FillContext(region=region, scale=0.30, line_width=width))
    return {"nested rings": nested_rings(width / 0.8), "Schwarz-P": schwarz}


def fig_dead_ink():
    """Why two crossing-free patterns were dropped."""
    from electrode_slicer.geometry import as_polygons, robust_union

    kept = layers_for("serpentine", 0.8)[-1]
    slicer = Slicer(PRINTER, SliceConfig())
    zs = [z for z, _ in slicer.layer_plan(electrode().height)]
    walls, region = slicer.walls(slicer.sections(electrode().mesh, zs)[-1])
    rejected = _rejected_fills(region)

    cases = [(name, walls + fill, "rejected") for name, fill in rejected.items()]
    cases.append(("serpentine", kept.all_paths, "kept"))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1))
    for ax, (name, paths, verdict) in zip(axes, cases, strict=True):
        beads = [ln.buffer(PRINTER.line_width / 2, quad_segs=8) for ln in paths]
        blobs = sorted(as_polygons(robust_union(beads)), key=lambda p: p.area, reverse=True)
        total = sum(b.area for b in blobs)
        dead = 1 - blobs[0].area / total if blobs else 0
        for i, blob in enumerate(blobs):
            colour = SLOT[0] if i == 0 else SLOT[7]
            ax.fill(*blob.exterior.xy, color=colour, lw=0)
            for ring in blob.interiors:
                ax.fill(*ring.xy, color=SURFACE, lw=0)
        ax.set_aspect("equal"); ax.axis("off")
        blob_word = "blob" if len(blobs) == 1 else "blobs"
        ax.set_title(f"{name} — {verdict}\n{len(blobs)} {blob_word}, "
                     f"{dead*100:.0f}% of ink dead", fontsize=8, color=INK, pad=4)
    fig.suptitle("Connected components of the deposited ink at ~80% density.  Blue is "
                 "the blob joined to the perimeter; red carries no current.",
                 fontsize=8.5, color=INK, y=1.04)
    finish(fig, "04-dead-ink.png")


def fig_frame_model():
    """The structural model and its validation."""
    fig = plt.figure(figsize=(11, 3.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.25)

    # A: validation against closed form.
    ax = fig.add_subplot(grid[0])
    width = height = 0.1
    area = width * height
    i_out = width * height ** 3 / 12
    lengths = np.linspace(0.5, 3.0, 8)
    computed, exact = [], []
    for L in lengths:
        pts = [(L * i / 8, 0.0) for i in range(9)]
        net = build_network([LineString(pts)])
        order = np.argsort(net.points[:, 0])
        root, tip = int(order[0]), int(order[-1])
        K = _assemble(net, True, area, i_out, 0.05)
        fixed = np.zeros(net.n_nodes * 3, dtype=bool)
        fixed[root * 3:root * 3 + 3] = True
        f = np.zeros(net.n_nodes * 3); f[tip * 3] = 1.0
        computed.append(abs(_solve(K, net, fixed, f)[tip * 3]))
        exact.append(L ** 3 / (3 * i_out))
    ax.plot(exact, exact, color=AXIS, lw=1.8, label="PL³/3EI (exact)")
    ax.plot(exact, computed, "o", ms=8, color=SLOT[0], mec=SURFACE, mew=1.2,
            label="frame model")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("closed form (mm/N, E=1)")
    ax.set_ylabel("model")
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.5, labelcolor=INK2)
    ax.set_title("A  solver validation", loc="left", fontsize=8.5, color=INK)

    # B and C: the network, chain versus braced.
    for col, name in ((1, "serpentine"), (2, "grid")):
        ax = fig.add_subplot(grid[col])
        layer = layers_for(name, 0.4)[-1]
        net = build_network(layer.all_paths, weld=PRINTER.line_width)
        for a, b in net.elements:
            ax.plot(*zip(net.points[a], net.points[b]), color=AXIS, lw=0.5, zorder=1)
        braced = net.degree >= 3
        ax.plot(net.points[braced, 0], net.points[braced, 1], "o", ms=3.2,
                color=SLOT[1], mec="none", zorder=3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{'B' if col == 1 else 'C'}  {name}: "
                     f"{braced.sum()} braced joints of {net.n_nodes}",
                     loc="left", fontsize=8.5, color=INK)
    fig.text(0.5, -0.04, "Orange marks nodes where three or more beads fuse.  A braced "
             "joint resists rotation, which is what makes a crossing pattern stiff.",
             ha="center", fontsize=7.5, color=INK2)
    finish(fig, "05-frame-model.png")


def fig_tradeoff():
    """The result: humps against bending, and where the cliff is."""
    families = {
        "no crossings": ["lines", "serpentine", "hilbert", "gyroid"],
        "bead ends meet": ["honeycomb"],
        "beads cross": ["stitched", "grid", "triangles", "stars"],
    }
    densities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    points = []
    for group, names in families.items():
        for name in names:
            for d in densities:
                layer = layers_for(name, d, flexibility=True)[-1]
                if layer.flex.solved:
                    points.append((group, name, d, layer.overlap.crossings_per_mm2,
                                   layer.flex.bend_compliance))
    for ratio in (2.0, 4.0, 8.0, 16.0):
        for d in densities:
            layer = layers_for("stitched", d, flexibility=True, stitch_ratio=ratio)[-1]
            if layer.flex.solved:
                points.append(("beads cross", f"stitched/{ratio:.0f}", d,
                               layer.overlap.crossings_per_mm2, layer.flex.bend_compliance))
    ref = next(b for g, n, d, x, b in points if n == "lines" and d == 0.2)
    points = [(g, n, d, x, b / ref) for g, n, d, x, b in points]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                  gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.28})
    for i, group in enumerate(families):
        xs = [x for g, n, d, x, b in points if g == group]
        ys = [b for g, n, d, x, b in points if g == group]
        ax.plot(xs, ys, "o", ms=7, color=SLOT[i], mec=SURFACE, mew=1.0,
                label=group, alpha=0.9, zorder=3)
    front = [a for a in points
             if not any(b[3] >= a[3] and b[4] >= a[4] and (b[3], b[4]) != (a[3], a[4])
                        for b in points)]
    front.sort(key=lambda t: t[3])
    ax.plot([x for *_, x, b in front], [b for *_, b in front],
            color=MUTED, lw=1.2, ls="--", zorder=2, label="best available trade")
    def locate(name: str, density: float) -> tuple[float, float]:
        """Find a labelled point in the data rather than hard-coding where it
        sits.  A typed-in coordinate goes stale the moment a number moves, and
        then the arrow points confidently at nothing."""
        for _, n, d, x, b in points:
            if n == name and abs(d - density) < 1e-9:
                return x, b
        raise KeyError(f"no point for {name} at {density:.0%}")

    for label, name, density, off in [
        # The stitch ratio is what distinguishes one stitched point from
        # another, so a label naming only the density would be ambiguous:
        # five stitched points share 40%, spanning 0.96 to 3.97 humps/mm2.
        ("serpentine, 80%", "serpentine", 0.80, (34, -22)),
        ("stitched, 40%\nstitch ratio 16", "stitched/16", 0.40, (14, 26)),
        ("stars, 80%", "stars", 0.80, (10, 40)),
    ]:
        ax.annotate(label, xy=locate(name, density), xytext=off,
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    ha="left" if off[0] > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    # The cliff, measured from the plotted data rather than typed in: the best
    # crossing-free point against the best one carrying any crossings at all.
    flat = max((q for q in points if q[3] == 0), key=lambda q: q[4])
    first = max((q for q in points if q[3] > 0), key=lambda q: q[4])
    band_hi = min(q[3] for q in points if q[3] >= 1.0)
    ax.axvspan(first[3] * 0.85, band_hi * 1.15, color=SLOT[3], alpha=0.12,
               lw=0, zorder=0)
    ax.text(math.sqrt(first[3] * band_hi), 0.30,
            f"the cliff:\nthe first crossings\ncost {flat[4] / first[4]:.1f}x the give",
            fontsize=7.5, color=INK2, ha="center", va="center")
    # symlog, not log: a crossing-free pattern has exactly zero humps, and a
    # log axis cannot show that without inventing a value for it.
    ax.set_xscale("symlog", linthresh=0.2, linscale=0.45)
    ax.set_yscale("log")
    ax.set_xlim(-0.03, 90)
    ax.axvline(0.2, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel("exposed humps per mm² on the skin-facing layer")
    ax.set_ylabel("bending compliance (relative to lines at 20%)")
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.42, 1.0), fontsize=7.5,
              labelcolor=INK2)
    ax.set_title("A  humps and bending pull against each other",
                 loc="left", fontsize=8.5, color=INK)

    # B: what each experimental axis drags along with it.
    ratios = [16.0, 8.0, 4.0, 2.0]
    stitch = [layers_for("stitched", 0.6, flexibility=True, stitch_ratio=r)[-1]
              for r in ratios]
    swap_names = ["serpentine", "stitched", "grid", "stars"]
    swap = [layers_for(n, 0.6, flexibility=True)[-1] for n in swap_names]

    def spread(items, get):
        vals = [get(i) for i in items]
        return max(vals) / min(vals)

    bars = [
        ("ink", spread(swap, lambda lp: lp.length),
         spread(stitch, lambda lp: lp.length)),
        ("bending", spread(swap, lambda lp: lp.flex.bend_compliance),
         spread(stitch, lambda lp: lp.flex.bend_compliance)),
    ]
    y = np.arange(len(bars))
    for k, (colour, label) in enumerate([(SLOT[0], "swap the pattern"),
                                         (SLOT[1], "sweep the stitch ratio")]):
        vals = [b[1 + k] for b in bars]
        ax2.barh(y + (0.5 - k) * 0.22, vals, height=0.2, color=colour, label=label)
        for yy, v in zip(y + (0.5 - k) * 0.22, vals, strict=True):
            ax2.text(v + 0.25, yy, f"{v:.1f}×", va="center", fontsize=7.5, color=INK2)
    ax2.axvline(1, color=AXIS, lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax2.set_xlabel("spread across the sweep (×, lower is better controlled)")
    ax2.set_xlim(0, 17)
    ax2.grid(axis="x", lw=0.8); ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", fontsize=7.5, labelcolor=INK2)
    ax2.set_title("B  both move humps; one drags less with it",
                  loc="left", fontsize=8.5, color=INK)
    fig.text(0.5, -0.04, "B compares two ways of moving hump count at a fixed 60% "
             "density, over comparable hump ranges: swapping the pattern family\n"
             "(serpentine to stars) against sweeping the stitch ratio (16 to 2). "
             "Both change the humps; only one leaves the rest of the electrode alone.",
             ha="center", fontsize=7.5, color=INK2)
    finish(fig, "06-tradeoff.png")


if __name__ == "__main__":
    print("writing figures:")
    fig_pattern_library()
    fig_hump_measurement()
    fig_calibration()
    fig_dead_ink()
    fig_frame_model()
    fig_tradeoff()
