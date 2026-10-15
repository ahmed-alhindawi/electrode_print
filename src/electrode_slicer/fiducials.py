"""Registration marks, so every print pass can be lined up on the first.

A plate is printed as many files -- each layer, and each layer's passes (see
:mod:`~electrode_slicer.passes`) -- and every one has to land exactly on what
is already there.  Each file therefore carries the same marks at the same
place on the plate, to align to the ones printed by the first.

A mark is a ring one bead wide with a dot at its centre, standing clear of
it.  Nothing in it meets anything else, for the same reason the passes are
split: a region slicer such as Arachne prints a lone strip or dot faithfully
and mangles any junction, and a crosshair is all junction, right where the
alignment has to be read.  The centre of a ring is easy to find by eye or by
camera, and the dot marks it.

Three marks go round the block of electrodes in an L -- bottom left, bottom
right, top right -- so the set fixes rotation and mirroring as well as
position: no turn or flip of the plate maps it onto itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString

__all__ = ["Fiducial", "place_fiducials"]


@dataclass(frozen=True)
class Fiducial:
    """One registration mark: a ring with a dot at its centre."""

    x: float
    y: float
    diameter: float          # outside of the ring
    dot: float               # diameter of the centre dot
    line_width: float        # the ring's bead

    def ring(self, segments: int = 72) -> LineString:
        """The ring's centreline, closed."""
        r = (self.diameter - self.line_width) / 2.0
        return LineString([
            (self.x + r * math.cos(2 * math.pi * k / segments),
             self.y + r * math.sin(2 * math.pi * k / segments))
            for k in range(segments + 1)
        ])

    @property
    def gap(self) -> float:
        """Clear space between the dot and the inside of the ring."""
        return (self.diameter / 2.0 - self.line_width) - self.dot / 2.0


def place_fiducials(variants, footprint, printer, clearance: float = 1.0,
                    diameter: float | None = None) -> tuple[list[Fiducial], list[str]]:
    """Three marks in an L round the electrodes, and any warning about them.

    The ring is ten beads across -- 1 mm at a 0.1 mm bead, small but well
    clear of anything that could be mistaken for it -- with a dot three beads
    wide, leaving a gap of more than a bead between them.  Each sits
    `clearance` beyond the corner of the block of electrodes.
    """
    if not variants:
        return [], []
    width = printer.line_width
    diameter = 10.0 * width if diameter is None else diameter
    dot = 3.0 * width
    part_w, part_d = footprint
    minx = min(v.dx for v in variants)
    miny = min(v.dy for v in variants)
    maxx = max(v.dx for v in variants) + part_w
    maxy = max(v.dy for v in variants) + part_d
    reach = clearance + diameter / 2.0
    marks = [
        Fiducial(minx - reach, miny - reach, diameter, dot, width),
        Fiducial(maxx + reach, miny - reach, diameter, dot, width),
        Fiducial(maxx + reach, maxy + reach, diameter, dot, width),
    ]
    warnings = []
    lo_x, lo_y = printer.margin, printer.margin
    hi_x, hi_y = printer.plate_x - printer.margin, printer.plate_y - printer.margin
    r = diameter / 2.0
    if any(f.x - r < lo_x or f.y - r < lo_y or f.x + r > hi_x or f.y + r > hi_y for f in marks):
        warnings.append(
            "the alignment marks fall inside the plate margin; there is no room for "
            "them round this many electrodes. Shrink the gap or the margin, or drop "
            "some electrodes, if the margin really is out of reach."
        )
    return marks, warnings
