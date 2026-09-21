"""Arranging a sweep of electrodes on the build plate.

The matrix layout puts one infill pattern per row and one density per column,
with replicates side by side inside their density block, so the plate reads
like the experiment it is.  Fully dense reference electrodes get a row of
their own underneath, because they belong to no pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .patterns import get_pattern

__all__ = ["Variant", "Grid", "build_variants", "assign_cells", "layout", "PATTERN_TAG"]

# Short tags used in the per-part captions, where there is no room for names.
PATTERN_TAG = {
    "lines": "lin",
    "grid": "grd",
    "triangles": "tri",
    "stars": "str",
    "solid": "sol",
    "none": "non",
    "serpentine": "snk",
    "hilbert": "hil",
    "honeycomb": "hex",
    "gyroid": "gyr",
}


@dataclass
class Variant:
    """One electrode on the plate: a pattern, a density and a position."""

    index: int
    pattern: str
    density: float
    replicate: int = 0
    #: True when this is one of several copies, so the id carries its number.
    replicated: bool = False
    col: int = 0
    row: int = 0
    dx: float = 0.0
    dy: float = 0.0
    is_reference: bool = False

    @property
    def tag(self) -> str:
        return PATTERN_TAG.get(self.pattern, self.pattern[:3])

    @property
    def label(self) -> str:
        return f"{self.tag} {self.density * 100:.0f}%"

    @property
    def ident(self) -> str:
        base = f"{self.pattern}_{self.density * 100:.0f}pct"
        if self.is_reference:
            base = f"reference_{base}"
        # Number every copy or none of them: an unnumbered id next to numbered
        # siblings reads like a different thing rather than the first of a set.
        return f"{base}_r{self.replicate}" if self.replicated else base

    @property
    def overlap_class(self) -> str:
        return get_pattern(self.pattern).overlap_class


@dataclass
class Grid:
    """Where the sweep sits on the plate, and what labels its axes."""

    cell_w: float
    cell_h: float
    cols: int
    rows: int
    x0: float = 0.0
    y0: float = 0.0
    col_label_y: float = 0.0
    col_labels: list[tuple[float, str]] = field(default_factory=list)
    row_labels: list[tuple[float, str]] = field(default_factory=list)


@dataclass
class LayoutConfig:
    """How to arrange the sweep; separate from what the sweep contains."""

    mode: str = "matrix"          # "matrix" or "pack"
    gap: float = 4.0
    columns: int = 0              # pack mode only; 0 means fit as many as possible
    labels: bool = True
    label_size: float = 1.6

    def __post_init__(self) -> None:
        if self.mode not in ("matrix", "pack"):
            raise ValueError("layout mode must be 'matrix' or 'pack'")
        if self.gap < 0:
            raise ValueError("gap cannot be negative")


def build_variants(patterns, densities, replicates: int = 1,
                   reference: int = 1) -> list[Variant]:
    """The cartesian product of patterns and densities, plus the references."""
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    if reference < 0:
        raise ValueError("reference must be >= 0")
    variants: list[Variant] = []
    for pattern in patterns:
        for density in densities:
            for rep in range(replicates):
                variants.append(Variant(
                    index=len(variants), pattern=pattern, density=density,
                    replicate=rep, replicated=replicates > 1,
                ))
    for rep in range(reference):
        variants.append(Variant(
            index=len(variants), pattern="solid", density=1.0,
            replicate=rep, replicated=reference > 1, is_reference=True,
        ))
    return variants


def assign_cells(variants: list[Variant], replicates: int = 1) -> tuple[int, int]:
    """Give every variant a (col, row) in the matrix; returns the grid shape."""
    sweep = [v for v in variants if not v.is_reference]
    refs = [v for v in variants if v.is_reference]

    patterns = list(dict.fromkeys(v.pattern for v in sweep))
    densities = sorted({v.density for v in sweep})
    reps = max(1, replicates)
    for var in sweep:
        var.row = patterns.index(var.pattern)
        var.col = densities.index(var.density) * reps + var.replicate
    for i, var in enumerate(refs):
        var.row, var.col = len(patterns), i
    cols = max((v.col for v in variants), default=0) + 1
    rows = max((v.row for v in variants), default=0) + 1
    return cols, rows


def layout(variants: list[Variant], footprint, printer, cfg: LayoutConfig,
           replicates: int = 1) -> Grid:
    """Place each variant on the plate and work out the axis labels."""
    part_w, part_h = footprint
    caption_room = (cfg.label_size * 1.8) if cfg.labels else 0.0
    cell_w = part_w + cfg.gap
    cell_h = part_h + caption_room + cfg.gap

    # Room for the pattern names down the left and the density headings along
    # the top, so the plate can be read without the manifest.
    axes = cfg.labels and cfg.mode == "matrix"
    longest = max((len(v.pattern) for v in variants), default=8)
    gutter = (max(longest, len("100% ref")) * cfg.label_size * 0.62) if axes else 0.0
    header = (cfg.label_size * 2.2) if axes else 0.0

    usable_w, usable_h = printer.usable_x, printer.usable_y
    if part_w + gutter > usable_w or part_h + caption_room + header > usable_h:
        raise ValueError(
            f"part footprint {part_w:.2f} x {part_h + caption_room:.2f} mm does not fit "
            f"the usable plate area {usable_w:.1f} x {usable_h:.1f} mm "
            f"(reduce the margin or scale the part down)"
        )

    if cfg.mode == "pack":
        cols = cfg.columns or max(1, int((usable_w + cfg.gap) // cell_w))
        cols = max(1, min(cols, len(variants)))
        rows = math.ceil(len(variants) / cols)
        for i, var in enumerate(variants):
            var.col, var.row = i % cols, i // cols
    else:
        cols, rows = assign_cells(variants, replicates)

    if cols * cell_w - cfg.gap + gutter > usable_w:
        fit = max(1, int((usable_w - gutter + cfg.gap) // cell_w))
        raise ValueError(
            f"the sweep is {cols} columns of {cell_w:.2f} mm but only {fit} fit across "
            f"the plate; drop some densities or replicates, or shrink the gap"
        )
    if rows * cell_h - cfg.gap + header > usable_h:
        fit = max(1, int((usable_h - header + cfg.gap) // cell_h))
        raise ValueError(
            f"the sweep is {rows} rows of {cell_h:.2f} mm but only {fit} fit up the "
            f"plate; drop some patterns, or shrink the gap"
        )

    # Centre the labelled block, then hang the grid off its top-left corner so
    # that row 0 is the first pattern given and reads at the top of the SVG.
    block_w = cols * cell_w - cfg.gap + gutter
    block_h = rows * cell_h - cfg.gap + header
    x0 = (printer.plate_x - block_w) / 2.0 + gutter
    y0 = (printer.plate_y - block_h) / 2.0

    def row_base(row: int) -> float:
        return y0 + (rows - 1 - row) * cell_h

    for var in variants:
        var.dx = x0 + var.col * cell_w
        var.dy = row_base(var.row) + caption_room

    grid = Grid(cell_w=cell_w, cell_h=cell_h, cols=cols, rows=rows, x0=x0, y0=y0)
    grid.col_label_y = row_base(0) + caption_room + part_h + cfg.label_size * 0.9
    if axes:
        seen_col: dict[int, float] = {}
        for var in variants:
            if not var.is_reference:
                seen_col.setdefault(var.col, var.density)
        reps = max(1, replicates)
        for col, density in sorted(seen_col.items()):
            if col % reps:      # label the density block once, not each replicate
                continue
            span = (reps - 1) * cell_w / 2.0
            grid.col_labels.append(
                (x0 + col * cell_w + part_w / 2.0 + span, f"{density * 100:.0f}%")
            )
        rows_seen: dict[int, str] = {}
        for var in variants:
            rows_seen.setdefault(var.row, "100% ref" if var.is_reference else var.pattern)
        for row, name in sorted(rows_seen.items()):
            grid.row_labels.append((row_base(row) + caption_room + part_h / 2.0, name))
    return grid
