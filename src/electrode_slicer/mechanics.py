"""How stiff a toolpath is, from its structure alone.

An electrode has to conform to skin, so how readily the printed network bends
matters as much as how it conducts.  That is usually a materials question, but
it does not have to be: in a linear frame model the Young's modulus and the
bead cross section appear as scalar multipliers on the whole stiffness matrix,
so setting both to one leaves a number that depends only on how the beads are
arranged.  Absolute values are therefore meaningless; ratios between patterns
at matched density are not, and those are what this reports.

The model is a planar frame of Euler-Bernoulli beams:

* every bead becomes a chain of beam elements, split wherever beads meet;
* joints are rigid, because two beads that cross are fused by the ink, and a
  rigid joint resists relative rotation where a pinned one would not;
* the in-plane (stretch, shear) and out-of-plane (bend, twist) problems
  decouple exactly for a planar frame, so both are solved from one assembly.

What it does not model: bonding between layers.  Each layer is analysed as its
own sheet, which compares the networks against each other but is not the
bending stiffness of the finished three-layer laminate.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .geometry import as_linestrings

__all__ = ["FrameNetwork", "Flexibility", "build_network", "analyse_flexibility"]


@dataclass
class FrameNetwork:
    """A bead network reduced to nodes and straight beam elements."""

    points: np.ndarray            # (n, 2) node coordinates in mm
    elements: np.ndarray          # (m, 2) node index pairs
    degree: np.ndarray            # (n,) how many elements meet at each node

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def branch_nodes(self) -> int:
        """Nodes where three or more beads meet, i.e. a braced joint."""
        return int(np.count_nonzero(self.degree >= 3))

    @property
    def free_ends(self) -> int:
        return int(np.count_nonzero(self.degree == 1))

    @property
    def lengths(self) -> np.ndarray:
        delta = self.points[self.elements[:, 1]] - self.points[self.elements[:, 0]]
        return np.hypot(delta[:, 0], delta[:, 1])


def weld_touching_ends(lines: list[LineString], distance: float) -> list[LineString]:
    """Join bead ends to the beads they land on top of.

    Infill runs a little way into the perimeter so the two fuse when printed,
    but their *centrelines* stop short of crossing.  Noding alone therefore
    leaves the fill floating free of the walls -- a structure with no load
    path at all.  Adding a stub from each free end to the bead it overlaps
    models the fused join, and without it the stiffness matrix is singular.
    """
    if distance <= 0 or len(lines) < 2:
        return lines
    tree = STRtree(lines)
    welds: list[LineString] = []
    for i, line in enumerate(lines):
        for end in (Point(line.coords[0]), Point(line.coords[-1])):
            reach = end.buffer(distance)
            best = None
            for j in tree.query(reach):
                j = int(j)
                if j == i:
                    continue
                gap = end.distance(lines[j])
                if gap <= distance and (best is None or gap < best[0]):
                    best = (gap, lines[j])
            if best is None or best[0] <= 1e-12:
                continue        # already touching, or nothing within reach
            landing = best[1].interpolate(best[1].project(end))
            welds.append(LineString([end, landing]))
    return lines + welds


def build_network(lines: list[LineString], snap: float = 1e-4,
                  simplify: float = 0.0, weld: float = 0.0) -> FrameNetwork:
    """Turn toolpaths into a frame: weld, node, then split into segments.

    ``unary_union`` splits every bead where it meets another, so a crossing
    becomes a shared node rather than two paths passing through each other --
    which is what makes a crossing brace the structure.
    """
    usable = [ln for ln in lines if ln.length > 0]
    if not usable:
        return FrameNetwork(np.empty((0, 2)), np.empty((0, 2), dtype=int), np.empty(0, int))
    usable = weld_touching_ends(usable, weld)
    noded = as_linestrings(unary_union(usable))
    if simplify > 0:
        noded = [ln.simplify(simplify, preserve_topology=False) for ln in noded]

    index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []

    def node_of(xy) -> int:
        key = (round(xy[0] / snap), round(xy[1] / snap))
        found = index.get(key)
        if found is None:
            found = index[key] = len(points)
            points.append((key[0] * snap, key[1] * snap))
        return found

    for line in noded:
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:], strict=False):
            ia, ib = node_of(a), node_of(b)
            if ia != ib:
                edges.append((ia, ib))

    if not edges:
        return FrameNetwork(np.empty((0, 2)), np.empty((0, 2), dtype=int), np.empty(0, int))
    elements = np.unique(np.sort(np.array(edges, dtype=int), axis=1), axis=0)
    coords = np.array(points, dtype=float)
    if weld > 0:
        elements = _bridge_components(coords, elements, weld * 1.5)
    degree = np.bincount(elements.ravel(), minlength=len(points))
    return FrameNetwork(coords, elements, degree)


def _bridge_components(points: np.ndarray, elements: np.ndarray,
                       max_gap: float) -> np.ndarray:
    """Join network pieces whose beads are fused but whose centrelines are not.

    Welding endpoints catches a bead that stops on another one, but two beads
    running side by side a fraction of a bead apart are fused along their
    length without either end being near anything.  The ink is one conductor;
    the frame has to be one structure too, or it has free-floating pieces and
    no solution.
    """
    if len(elements) == 0:
        return elements
    added: list[tuple[int, int]] = []
    for _ in range(64):
        graph = coo_matrix(
            (np.ones(len(elements)), (elements[:, 0], elements[:, 1])),
            shape=(len(points), len(points)),
        )
        count, labels = connected_components(graph, directed=False)
        used = np.unique(elements)
        if count - (len(points) - len(used)) <= 1:
            break                       # one piece, ignoring unused nodes
        sizes = np.bincount(labels[used], minlength=count)
        main = int(np.argmax(sizes))
        anchors = used[labels[used] == main]
        others = used[labels[used] != main]
        if len(anchors) == 0 or len(others) == 0:
            break
        tree = cKDTree(points[anchors])
        gaps, nearest = tree.query(points[others])
        best = int(np.argmin(gaps))
        if gaps[best] > max_gap:
            break                       # genuinely separate ink; leave it be
        added.append((int(others[best]), int(anchors[nearest[best]])))
        elements = np.unique(
            np.sort(np.vstack([elements, np.array(added[-1:])]), axis=1), axis=0)
    return elements


# --------------------------------------------------------------------------
# element stiffness
# --------------------------------------------------------------------------


def _beam_out_of_plane(length: float, ei: float, gj: float) -> np.ndarray:
    """6x6 for [w1, twist1, slope1, w2, twist2, slope2] in the beam's own frame."""
    k = np.zeros((6, 6))
    inv = ei / length ** 3
    bend = np.array([
        [12.0, 6 * length, -12.0, 6 * length],
        [6 * length, 4 * length ** 2, -6 * length, 2 * length ** 2],
        [-12.0, -6 * length, 12.0, -6 * length],
        [6 * length, 2 * length ** 2, -6 * length, 4 * length ** 2],
    ]) * inv
    for a, ia in enumerate((0, 2, 3, 5)):
        for b, ib in enumerate((0, 2, 3, 5)):
            k[ia, ib] = bend[a, b]
    twist = gj / length
    k[1, 1] = k[4, 4] = twist
    k[1, 4] = k[4, 1] = -twist
    return k


def _beam_in_plane(length: float, ea: float, ei: float) -> np.ndarray:
    """6x6 for [u1, v1, rz1, u2, v2, rz2] in the beam's own frame."""
    k = np.zeros((6, 6))
    axial = ea / length
    k[0, 0] = k[3, 3] = axial
    k[0, 3] = k[3, 0] = -axial
    inv = ei / length ** 3
    bend = np.array([
        [12.0, 6 * length, -12.0, 6 * length],
        [6 * length, 4 * length ** 2, -6 * length, 2 * length ** 2],
        [-12.0, -6 * length, 12.0, -6 * length],
        [6 * length, 2 * length ** 2, -6 * length, 4 * length ** 2],
    ]) * inv
    for a, ia in enumerate((1, 2, 4, 5)):
        for b, ib in enumerate((1, 2, 4, 5)):
            k[ia, ib] = bend[a, b]
    return k


def _assemble(net: FrameNetwork, out_of_plane: bool, ea: float, ei: float,
              gj: float) -> coo_matrix:
    n_dof = net.n_nodes * 3
    rows, cols, vals = [], [], []
    delta = net.points[net.elements[:, 1]] - net.points[net.elements[:, 0]]
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    for (na, nb), (dx, dy), length in zip(net.elements, delta, lengths, strict=True):
        if length <= 0:
            continue
        c, s = dx / length, dy / length
        if out_of_plane:
            local = _beam_out_of_plane(length, ei, gj)
            # (w, rx, ry) -> (w, twist about the beam, slope along it)
            rot = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
        else:
            local = _beam_in_plane(length, ea, ei)
            # (u, v, rz) -> (along the beam, across it, rz)
            rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        transform = np.zeros((6, 6))
        transform[:3, :3] = rot
        transform[3:, 3:] = rot
        k = transform.T @ local @ transform
        dofs = [na * 3, na * 3 + 1, na * 3 + 2, nb * 3, nb * 3 + 1, nb * 3 + 2]
        for a in range(6):
            for b in range(6):
                if k[a, b]:
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(k[a, b])
    return coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def _solve(matrix, net: FrameNetwork, fixed: np.ndarray, load: np.ndarray):
    """Solve K u = f with the fixed DOFs removed."""
    free = ~fixed
    if not free.any() or not fixed.any():
        return None
    reduced = matrix[free][:, free]
    try:
        with warnings.catch_warnings():
            # A disconnected frame is a real possibility and is reported
            # through `solved`; scipy's warning about it is noise here.
            warnings.simplefilter("ignore", MatrixRankWarning)
            solution = spsolve(reduced.tocsc(), load[free])
    except Exception:                                    # pragma: no cover
        return None
    if solution is None or not np.all(np.isfinite(solution)):
        return None
    full = np.zeros(net.n_nodes * 3)
    full[free] = solution
    return full


def _grip_masks(net: FrameNetwork, axis: int, grip: float):
    """Nodes to clamp at one end and to pull at the other."""
    coord = net.points[:, axis]
    low, high = coord.min(), coord.max()
    span = high - low
    if span <= 0:
        return None, None
    held = coord <= low + span * grip
    pulled = coord >= high - span * grip
    if not held.any() or not pulled.any() or (held & pulled).all():
        return None, None
    return held, pulled


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


@dataclass
class Flexibility:
    """Structural compliance of one layer's network, material factored out."""

    nodes: int = 0
    elements: int = 0
    branch_nodes: int = 0          # welded joints that brace the network
    free_ends: int = 0
    mean_degree: float = 0.0
    braced_fraction: float = 0.0   # share of nodes that are braced joints
    anisotropy: float = 0.0        # 0 = beads evenly spread, 1 = all parallel
    principal_angle_deg: float = 0.0
    bend_compliance_x: float = 0.0     # deflection per unit load, curving along x
    bend_compliance_y: float = 0.0
    bend_compliance: float = 0.0       # geometric mean of the two
    bend_anisotropy: float = 1.0       # stiffest direction over most compliant
    stretch_compliance_x: float = 0.0
    stretch_compliance_y: float = 0.0
    stretch_compliance: float = 0.0
    #: False when the frame could not be solved.  Zero compliance would read
    #: as infinitely stiff, which is the opposite of what a failure means.
    solved: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _typical(displacements: np.ndarray) -> float:
    """How far the loaded edge moves, ignoring floppy whiskers.

    A toolpath ends in the odd pendant stub -- a bead that found nothing to
    fuse to.  Structurally it carries no load between any two other points,
    so it says nothing about how stiff the network is, but it deflects
    enormously and can drag a mean up by eight orders of magnitude.  The
    median describes the edge that is actually doing the work.
    """
    if displacements.size == 0:
        return 0.0
    return float(np.median(np.abs(displacements)))


def _orientation(net: FrameNetwork) -> tuple[float, float]:
    """Length-weighted spread of bead directions, as an anisotropy and angle."""
    delta = net.points[net.elements[:, 1]] - net.points[net.elements[:, 0]]
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    keep = lengths > 0
    if not keep.any():
        return 0.0, 0.0
    unit = delta[keep] / lengths[keep, None]
    weight = lengths[keep]
    # Second-order orientation tensor; beads have no head or tail, so this is
    # the right object rather than a mean direction, which would cancel out.
    tensor = (unit[:, :, None] * unit[:, None, :] * weight[:, None, None]).sum(0)
    tensor /= weight.sum()
    values, vectors = np.linalg.eigh(tensor)
    anisotropy = float(values[-1] - values[0])          # 0 isotropic, 1 parallel
    major = vectors[:, -1]
    return anisotropy, float(math.degrees(math.atan2(major[1], major[0])) % 180.0)


def analyse_flexibility(
    lines: list[LineString],
    line_width: float,
    layer_height: float,
    grip: float = 0.15,
    simplify: float = 0.0,
    max_nodes: int = 60_000,
) -> Flexibility:
    """Relative structural compliance of a bead network.

    Young's modulus is taken as 1 and the bead as a solid rectangle of
    ``line_width`` by ``layer_height``, so the numbers carry no material and
    are only meaningful against each other.
    """
    report = Flexibility()
    net = build_network(lines, simplify=simplify, weld=line_width)
    if net.n_elements == 0:
        return report
    report.nodes = net.n_nodes
    report.elements = net.n_elements
    report.branch_nodes = net.branch_nodes
    report.free_ends = net.free_ends
    report.mean_degree = float(net.degree.mean()) if net.n_nodes else 0.0
    report.braced_fraction = net.branch_nodes / net.n_nodes if net.n_nodes else 0.0
    report.anisotropy, report.principal_angle_deg = _orientation(net)
    if net.n_nodes > max_nodes:      # the solve would dominate the whole run
        return report

    # Unit modulus, rectangular bead.  Poisson's ratio only enters through the
    # torsion constant, where 0.35 is unremarkable for a cured conductive ink
    # and barely moves the comparison anyway.
    width, height = line_width, layer_height
    area = width * height
    second_moment = width * height ** 3 / 12.0          # bending out of plane
    second_moment_plane = height * width ** 3 / 12.0    # bending in plane
    shear_modulus = 1.0 / (2.0 * (1.0 + 0.35))
    short, long_ = min(width, height), max(width, height)
    torsion = long_ * short ** 3 * (
        1 / 3 - 0.21 * short / long_ * (1 - short ** 4 / (12 * long_ ** 4))
    )

    out_matrix = _assemble(net, True, area, second_moment, shear_modulus * torsion)
    in_matrix = _assemble(net, False, area, second_moment_plane, 0.0)

    for axis, name in ((0, "x"), (1, "y")):
        held, pulled = _grip_masks(net, axis, grip)
        if held is None:
            continue
        fixed = np.repeat(held, 3)

        # Out of plane: press the far edge down and see how far it goes.
        load = np.zeros(net.n_nodes * 3)
        load[np.flatnonzero(pulled) * 3] = 1.0 / np.count_nonzero(pulled)
        solved = _solve(out_matrix, net, fixed, load)
        if solved is not None:
            setattr(report, f"bend_compliance_{name}",
                    _typical(solved[np.flatnonzero(pulled) * 3]))

        # In plane: pull the far edge away and see how far it stretches.
        load = np.zeros(net.n_nodes * 3)
        load[np.flatnonzero(pulled) * 3 + axis] = 1.0 / np.count_nonzero(pulled)
        solved = _solve(in_matrix, net, fixed, load)
        if solved is not None:
            setattr(report, f"stretch_compliance_{name}",
                    _typical(solved[np.flatnonzero(pulled) * 3 + axis]))

    report.solved = all(
        getattr(report, f"{kind}_compliance_{axis}") > 0
        for kind in ("bend", "stretch") for axis in ("x", "y")
    )
    for kind in ("bend", "stretch"):
        a = getattr(report, f"{kind}_compliance_x")
        b = getattr(report, f"{kind}_compliance_y")
        if a > 0 and b > 0:
            setattr(report, f"{kind}_compliance", math.sqrt(a * b))
            if kind == "bend":
                report.bend_anisotropy = max(a, b) / min(a, b)
        else:
            setattr(report, f"{kind}_compliance", max(a, b))
    return report
