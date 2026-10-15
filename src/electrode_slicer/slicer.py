"""Turning a mesh into per-layer toolpaths.

The slicer owns the parts of the job that are the same whatever pattern is
being used: where the layers sit, what the cross section is, how many walls go
round it, and -- the part that makes a pattern comparison meaningful -- how to
drive a pattern's one scale knob until the density it actually achieves
matches the density that was asked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import trimesh
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry import box as rectangle
from shapely.ops import unary_union

from .geometry import (
    as_polygons,
    extend_ends,
    ring_lines,
    robust_intersection,
    simplify_paths,
    toolpath_area,
)
from .mechanics import Flexibility, analyse_flexibility
from .metrics import OverlapStats, analyse_overlap
from .patterns import FillContext, InfillPattern, get_pattern
from .printer import Printer

__all__ = ["Modifier", "SliceConfig", "LayerPaths", "Slicer", "layer_heights", "coverage"]


MODIFIER_SHAPES = ("box", "cylinder")
OUTSIDE_MODES = ("solid", "skip")


@dataclass(frozen=True)
class Modifier:
    """A volume of the model that the sweep's pattern and density apply to.

    An electrode is more than its sensing face: a probe or tab carries the
    current away, and sweeping its infill only muddies the comparison.  Place
    one of these over the sensing face and the sweep fills just that; what
    lies outside every modifier is printed solid, or left out (see
    ``SliceConfig.outside``).

    Coordinates are millimetres in the frame of the loaded model, which sits
    with its bounding box's corner at the origin.  ``center`` is the middle of
    the volume and ``size`` its width, depth and height; a cylinder is the
    upright one inscribed in that box, so equal width and depth give a disc.
    """

    shape: str = "box"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.shape not in MODIFIER_SHAPES:
            raise ValueError(
                f"modifier shape must be one of {', '.join(MODIFIER_SHAPES)}, "
                f"not {self.shape!r}"
            )
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("a modifier needs a 3D centre and a 3D size")
        if not all(math.isfinite(v) for v in (*self.center, *self.size)):
            raise ValueError("modifier centre and size must be finite numbers")
        if min(self.size) <= 0:
            raise ValueError("modifier width, depth and height must all be positive")

    def footprint(self, z: float):
        """Its cross section at height `z`: empty unless `z` lies inside it."""
        (cx, cy, cz), (w, d, h) = self.center, self.size
        if abs(z - cz) > h / 2.0:
            return Polygon()
        if self.shape == "box":
            return rectangle(cx - w / 2.0, cy - d / 2.0, cx + w / 2.0, cy + d / 2.0)
        disc = Point(cx, cy).buffer(0.5, quad_segs=64)
        return affinity.scale(disc, xfact=w, yfact=d, origin=(cx, cy))


@dataclass
class SliceConfig:
    """Everything about the job that is not the machine and not the shape."""

    perimeters: int = 2
    infill_angle: float = 45.0
    infill_overlap: float = 0.15      # bead ends run this far into the walls
    alternate_angles: bool = True     # cross-hatch line infill between layers
    solid_layers: int = 0             # fully dense top and bottom layers
    slice_at: str = "mid"             # where in a layer to take the section
    calibrate: bool = True            # bisect to the requested density
    density_metric: str = "area"      # "area" is exact, "length" is a fast guess
    tolerance: float = 0.005          # calibration tolerance, coverage fraction
    max_iterations: int = 16
    flexibility: bool = False         # run the frame model on each layer
    stitch_ratio: float | None = None  # row spacings between stitched's stitches
    tpms_resolution: float | None = None
    # Where the sweep applies; none means the whole part.  Outside them the
    # part is printed "solid", or "skip"ped altogether.
    modifiers: tuple[Modifier, ...] = ()
    outside: str = "solid"

    def __post_init__(self) -> None:
        self.modifiers = tuple(self.modifiers)
        if self.outside not in OUTSIDE_MODES:
            raise ValueError(f"outside must be one of {', '.join(OUTSIDE_MODES)}")
        if self.perimeters < 0:
            raise ValueError("perimeters must be >= 0")
        if self.solid_layers < 0:
            raise ValueError("solid_layers must be >= 0")
        if self.slice_at not in ("mid", "bottom", "top"):
            raise ValueError("slice_at must be 'mid', 'bottom' or 'top'")
        if self.density_metric not in ("area", "length"):
            raise ValueError("density_metric must be 'area' or 'length'")


@dataclass
class LayerPaths:
    """The toolpaths and measurements for one layer of one electrode."""

    index: int
    z: float
    perimeters: list[LineString] = field(default_factory=list)
    infill: list[LineString] = field(default_factory=list)
    solid: list[LineString] = field(default_factory=list)   # outside the modifiers
    # The area the walls and solid fill cover between them -- the slice less
    # the region the pattern fills -- for a tool that fills areas itself.
    walls_area: Polygon | MultiPolygon = field(default_factory=Polygon)
    region_area: float = 0.0
    slice_area: float = 0.0
    density_target: float = 0.0
    density_actual: float = 0.0
    scale: float = 0.0
    pattern: str = ""
    overlap: OverlapStats = field(default_factory=OverlapStats)
    flex: Flexibility = field(default_factory=Flexibility)

    @property
    def all_paths(self) -> list[LineString]:
        return self.perimeters + self.solid + self.infill

    @property
    def length(self) -> float:
        return sum(ln.length for ln in self.all_paths)


def layer_heights(
    total: float, layer_h: float, first_h: float, slice_at: str = "mid"
) -> list[tuple[float, float]]:
    """Return ``(slice_z, layer_top_z)`` for every layer covering `total`."""
    out: list[tuple[float, float]] = []
    bottom = 0.0
    # Tolerate the float noise a mesh's bounding box carries, so a 0.30 mm part
    # at 0.1 mm layers gives three layers rather than three plus a sliver.
    eps = max(layer_h, first_h) * 1e-3
    while bottom < total - eps:
        h = first_h if not out else layer_h
        top = min(bottom + h, total)
        if slice_at == "bottom":
            z = bottom + min(1e-4, (top - bottom) * 0.01)
        elif slice_at == "top":
            z = top - min(1e-4, (top - bottom) * 0.01)
        else:
            z = 0.5 * (bottom + top)
        out.append((z, top))
        bottom = top
    return out


def coverage(lines: list[LineString], region, line_width: float, metric: str = "area") -> float:
    """Fraction of `region` covered by the deposited beads."""
    if not lines or region.is_empty or region.area <= 0:
        return 0.0
    if metric == "length":
        # Cheap, but double counts crossings so it over-reads on lattices.
        return sum(ln.length for ln in lines) * line_width / region.area
    return robust_intersection(toolpath_area(lines, line_width), region).area / region.area


class Slicer:
    """Slices a mesh and fills each layer with a chosen pattern."""

    def __init__(self, printer: Printer, config: SliceConfig | None = None,
                 scale_cache: dict | None = None) -> None:
        self.printer = printer
        self.config = config or SliceConfig()
        # Layers with identical geometry and settings share a solved scale; a
        # thin flat part repeats the same cross section many times over.  The
        # caller may pass a cache in to keep it across runs -- the key covers
        # every input to the solve, so a shared one cannot go stale.
        self._scale_cache: dict[tuple, float] = {} if scale_cache is None else scale_cache

    # -- geometry ---------------------------------------------------------

    def layer_plan(self, height: float) -> list[tuple[float, float]]:
        return layer_heights(
            height, self.printer.layer_height, self.printer.first_layer, self.config.slice_at
        )

    def sections(self, mesh, zs: list[float]) -> list:
        """Cross-section the mesh at each height, in a shared world XY frame."""
        if not zs:
            return []
        planes = mesh.section_multiplane(
            plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=zs
        )
        out = []
        for section in planes:
            if section is None:
                out.append(MultiPolygon())
                continue
            polys = [p for p in section.polygons_full if p.area > 0]
            out.append(unary_union(polys) if polys else MultiPolygon())
        return out

    def walls(self, slice_geom):
        """Offset perimeter loops, plus the region left over for infill."""
        count = self.config.perimeters
        width = self.printer.line_width
        loops: list[LineString] = []
        for i in range(count):
            inset = slice_geom.buffer(-(i + 0.5) * width, join_style="round")
            polys = as_polygons(inset)
            if not polys:
                break
            loops.extend(ring_lines(polys))
        inner = slice_geom.buffer(-count * width, join_style="round") if count else slice_geom
        return loops, (inner if not inner.is_empty else Polygon())

    def zone(self, z: float):
        """Where the sweep applies at height `z`, or None for everywhere."""
        if not self.config.modifiers:
            return None
        return unary_union([m.footprint(z) for m in self.config.modifiers])

    # -- density calibration ----------------------------------------------

    def _context(self, pattern, region, scale, angle, z) -> FillContext:
        return FillContext(
            region=region, scale=scale, angle_deg=angle, z=z,
            line_width=self.printer.line_width,
            resolution=self.config.tpms_resolution,
            stitch_ratio=self.config.stitch_ratio,
        )

    def measure(self, pattern: InfillPattern, region, scale, angle, z) -> float:
        lines = pattern.generate(self._context(pattern, region, scale, angle, z))
        return coverage(lines, region, self.printer.line_width, self.config.density_metric)

    def solve_scale(
        self, pattern: InfillPattern, region, target: float, angle: float, z: float
    ) -> tuple[float, float]:
        """Find the scale knob whose achieved coverage matches `target`.

        Returns ``(scale, achieved_coverage)``.  Coverage falls monotonically
        as the knob grows, so bracket the target and then bisect in log space:
        the knob is a length, and multiplicative steps behave far better than
        additive ones.
        """
        width = self.printer.line_width
        tolerance = self.config.tolerance
        guess = pattern.initial_scale(target, width)
        # Each pattern knows where its own knob saturates; below that there is
        # nothing left to find, and probing further is where the geometry
        # explodes.
        floor_scale = pattern.min_scale(width)

        guess_cov = self.measure(pattern, region, guess, angle, z)
        best_scale, best_cov = guess, guess_cov
        if abs(best_cov - target) <= tolerance:
            return best_scale, best_cov

        def remember(s: float, c: float) -> None:
            nonlocal best_scale, best_cov
            if abs(c - target) < abs(best_cov - target):
                best_scale, best_cov = s, c

        # Bracket: lo is the dense end (coverage at or above target), hi the
        # sparse end.  Keep each end's coverage too -- the search below
        # interpolates with them rather than just halving the interval.
        lo_s = hi_s = None
        lo_c = hi_c = 0.0
        if guess_cov >= target:
            lo_s, lo_c, probe = guess, guess_cov, guess
            for _ in range(14):
                probe *= 1.5
                cov = self.measure(pattern, region, probe, angle, z)
                remember(probe, cov)
                if cov < target:
                    hi_s, hi_c = probe, cov
                    break
                lo_s, lo_c = probe, cov
        else:
            hi_s, hi_c, probe = guess, guess_cov, guess
            for _ in range(14):
                # Clamp to the floor and still measure there.  Stopping short
                # of it would throw away a perfectly good bracket -- the floor
                # is solid by definition, so it satisfies any target -- and
                # leave dense infill quietly under-delivered.
                probe = max(probe / 1.5, floor_scale)
                cov = self.measure(pattern, region, probe, angle, z)
                remember(probe, cov)
                if cov >= target:
                    lo_s, lo_c = probe, cov
                    break
                hi_s, hi_c = probe, cov
                if probe <= floor_scale:
                    break
        if lo_s is None or hi_s is None:
            return best_scale, best_cov

        # False position in log space, with the Illinois tweak.  Coverage is
        # close to linear in log(scale) -- it is k*w/s for a line fill -- so
        # interpolating lands near the answer in a few steps where bisection
        # would spend a dozen.  Every one of those steps is a full fill plus a
        # polygon union, which is the whole cost of a sweep.  Illinois halves
        # the stale end's weight when one side stops moving, which keeps the
        # guaranteed convergence that plain false position can lose.
        log_lo, log_hi = math.log(lo_s), math.log(hi_s)
        f_lo, f_hi = lo_c - target, hi_c - target
        stalled_lo = stalled_hi = 0
        for _ in range(self.config.max_iterations):
            if f_lo == f_hi:
                mid_log = 0.5 * (log_lo + log_hi)
            else:
                mid_log = log_lo + (log_hi - log_lo) * f_lo / (f_lo - f_hi)
                # Never let the step crawl into a corner; fall back to the
                # midpoint if interpolation lands outside the middle 98%.
                span = log_hi - log_lo
                edge = abs(span) * 0.01
                lo_edge, hi_edge = min(log_lo, log_hi) + edge, max(log_lo, log_hi) - edge
                if not lo_edge <= mid_log <= hi_edge:
                    mid_log = 0.5 * (log_lo + log_hi)
            mid = math.exp(mid_log)
            cov = self.measure(pattern, region, mid, angle, z)
            remember(mid, cov)
            if abs(cov - target) <= tolerance:
                return mid, cov
            if cov >= target:
                log_lo, f_lo = mid_log, cov - target
                stalled_lo, stalled_hi = 0, stalled_hi + 1
                if stalled_hi >= 2:
                    f_hi *= 0.5
            else:
                log_hi, f_hi = mid_log, cov - target
                stalled_hi, stalled_lo = 0, stalled_lo + 1
                if stalled_lo >= 2:
                    f_lo *= 0.5
        return best_scale, best_cov

    # -- one layer ---------------------------------------------------------

    def infill_angle(self, pattern: InfillPattern, index: int) -> float:
        """Cross-hatching successive layers bonds them together.

        Lattice patterns already self-cross and TPMS cross sections already
        vary through Z, so only single-family line fills are rotated.
        """
        angle = self.config.infill_angle
        if self.config.alternate_angles and pattern.families == 1 and not pattern.z_dependent:
            angle += 90.0 * (index % 2)
        return angle

    def build_layer(self, index: int, z: float, slice_geom, pattern_name: str,
                    density: float) -> LayerPaths:
        cfg, printer = self.config, self.printer
        layer = LayerPaths(index=index, z=z, density_target=density, pattern=pattern_name)
        zone = self.zone(z)
        if zone is not None and cfg.outside == "skip":
            slice_geom = _polygonal(robust_intersection(slice_geom, zone))
        if slice_geom.is_empty:
            return layer
        layer.slice_area = slice_geom.area

        loops, region = self.walls(slice_geom)
        layer.perimeters = simplify_paths(loops, printer.resolution)
        if zone is not None and cfg.outside == "solid":
            # Fill everything outside the modifiers solid, and leave the sweep
            # only what is inside them -- the density is measured there alone.
            rest = _polygonal(region.difference(zone))
            region = _polygonal(robust_intersection(region, zone))
            layer.solid = self.fill_solid(rest, index, z, slice_geom)
        layer.region_area = region.area
        layer.walls_area = (slice_geom if region.is_empty
                            else _polygonal(slice_geom.difference(region)))

        def finish() -> LayerPaths:
            layer.overlap = analyse_overlap(
                layer.infill, layer.all_paths, printer.line_width,
                printer.layer_height, layer.region_area or layer.slice_area,
            )
            if cfg.flexibility:
                layer.flex = analyse_flexibility(
                    layer.all_paths, printer.line_width, printer.layer_height
                )
            return layer

        if region.is_empty or pattern_name == "none" or density <= 0:
            return finish()

        # At full density every pattern degenerates to beads laid edge to
        # edge, so route it through the solid fill rather than asking a
        # lattice to resolve something it cannot.
        is_solid = pattern_name == "solid" or density >= 0.999
        pattern = get_pattern("solid" if is_solid else pattern_name)
        target = 1.0 if is_solid else density
        # Record what this layer actually aimed at.  Leaving the requested
        # density here would have the checks read a deliberate 100% pad as a
        # calibration that missed by eighty points.
        layer.density_target = target
        layer.pattern = pattern.name
        if is_solid:
            layer.walls_area = slice_geom
        angle = self.infill_angle(pattern, index)

        key = (
            pattern.name, round(target, 6), round(angle, 4),
            round(z, 6) if pattern.z_dependent else None,
            printer.line_width, cfg.density_metric, cfg.tolerance,
            cfg.tpms_resolution, cfg.stitch_ratio, region.wkb,
        )
        if key in self._scale_cache:
            scale = self._scale_cache[key]
        elif is_solid or not cfg.calibrate:
            scale = pattern.initial_scale(target, printer.line_width)
        else:
            scale, _ = self.solve_scale(pattern, region, target, angle, z)
        self._scale_cache[key] = scale

        layer.scale = scale
        layer.infill = pattern.generate(self._context(pattern, region, scale, angle, z))
        layer.density_actual = coverage(
            layer.infill, region, printer.line_width, cfg.density_metric
        )
        # Weld the infill into the walls only now: the extensions sit outside
        # the measured region, so they cannot skew the calibrated density.
        layer.infill = extend_ends(
            layer.infill, cfg.infill_overlap * printer.line_width, region, slice_geom
        )
        layer.infill = simplify_paths(layer.infill, printer.resolution)
        return finish()

    def fill_solid(self, region, index: int, z: float, limit) -> list[LineString]:
        """Beads laid edge to edge over `region`, welded into what borders it."""
        if region.is_empty:
            return []
        width = self.printer.line_width
        # Slivers narrower than a bead hold no pass of their own; the walls and
        # the neighbouring fill already cover them.
        if region.buffer(-width / 2.0).is_empty:
            return []
        pattern = get_pattern("solid")
        angle = self.config.infill_angle + 90.0 * (index % 2)
        lines = pattern.generate(
            self._context(pattern, region, pattern.initial_scale(1.0, width), angle, z)
        )
        lines = extend_ends(lines, self.config.infill_overlap * width, region, limit)
        return simplify_paths(lines, self.printer.resolution)

    # -- a whole electrode --------------------------------------------------

    def build(self, slices: list, zs: list[float], pattern_name: str,
              density: float) -> list[LayerPaths]:
        """Every layer of one electrode at one pattern and density."""
        n = len(slices)
        out: list[LayerPaths] = []
        for index, (z, geom) in enumerate(zip(zs, slices, strict=True)):
            name, dens = pattern_name, density
            solid_n = self.config.solid_layers
            if solid_n and (index < solid_n or index >= n - solid_n):
                name, dens = "solid", 1.0
            out.append(self.build_layer(index, z, geom, name, dens))
        return out


def _polygonal(geom):
    """An overlay result as a Polygon or MultiPolygon, dropping stray lines."""
    polys = as_polygons(geom)
    if not polys:
        return Polygon()
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def mesh_is_usable(mesh) -> bool:
    return isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty and len(mesh.faces) > 0
