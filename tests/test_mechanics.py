"""The structural model.

Everything here rests on the frame solver being right, so it is checked
against closed-form results first.  Young's modulus and the bead section are
scalar multipliers on the whole stiffness matrix, so setting both to one
leaves numbers that depend only on how the beads are arranged -- which makes
absolute values meaningless and ratios between patterns the whole point.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from electrode_slicer import SweepRequest, run_sweep
from electrode_slicer.mechanics import (
    _assemble,
    _solve,
    analyse_flexibility,
    build_network,
    weld_touching_ends,
)

WIDTH = HEIGHT = 0.1
AREA = WIDTH * HEIGHT
I_OUT = WIDTH * HEIGHT ** 3 / 12.0
I_IN = HEIGHT * WIDTH ** 3 / 12.0


def cantilever(length, n_elements, out_of_plane, load_dof):
    """A straight beam, clamped at one end, unit load at the other."""
    pts = [(length * i / n_elements, 0.0) for i in range(n_elements + 1)]
    net = build_network([LineString(pts)])
    order = np.argsort(net.points[:, 0])
    root, tip = int(order[0]), int(order[-1])
    matrix = _assemble(net, out_of_plane, AREA, I_OUT if out_of_plane else I_IN, 0.05)
    fixed = np.zeros(net.n_nodes * 3, dtype=bool)
    fixed[root * 3:root * 3 + 3] = True
    load = np.zeros(net.n_nodes * 3)
    load[tip * 3 + load_dof] = 1.0
    solution = _solve(matrix, net, fixed, load)
    return abs(solution[tip * 3 + load_dof])


# --------------------------------------------------------------------------
# the solver, against results that can be worked out by hand
# --------------------------------------------------------------------------


def test_cantilever_tip_deflection_matches_pl3_over_3ei():
    assert cantilever(2.0, 8, True, 0) == pytest.approx(2.0 ** 3 / (3 * I_OUT), rel=1e-9)


def test_in_plane_cantilever_matches_the_same_formula():
    assert cantilever(2.0, 8, False, 1) == pytest.approx(2.0 ** 3 / (3 * I_IN), rel=1e-9)


def test_axial_extension_matches_pl_over_ea():
    assert cantilever(2.0, 8, False, 0) == pytest.approx(2.0 / AREA, rel=1e-9)


@pytest.mark.parametrize("n_elements", [1, 2, 4, 16, 64])
def test_the_answer_does_not_depend_on_how_finely_the_beam_is_split(n_elements):
    """Euler-Bernoulli elements are exact for a straight beam, so refining
    the mesh must change nothing at all."""
    assert cantilever(2.0, n_elements, True, 0) == pytest.approx(
        2.0 ** 3 / (3 * I_OUT), rel=1e-6)


def test_deflection_scales_as_the_cube_of_length():
    short = cantilever(1.0, 8, True, 0)
    long_ = cantilever(2.0, 8, True, 0)
    assert long_ / short == pytest.approx(8.0, rel=1e-6)


def test_a_diagonal_brace_stiffens_a_square_frame():
    """The mechanism behind the whole comparison: a crossing is a brace."""
    def shear(with_diagonal):
        sides = [LineString([(0, 0), (2, 0)]), LineString([(2, 0), (2, 2)]),
                 LineString([(2, 2), (0, 2)]), LineString([(0, 2), (0, 0)])]
        if with_diagonal:
            sides.append(LineString([(0, 0), (2, 2)]))
        net = build_network(sides)
        matrix = _assemble(net, False, AREA, I_IN, 0.0)
        right = np.flatnonzero(net.points[:, 0] >= 2 - 1e-9)
        fixed = np.repeat(net.points[:, 0] <= 1e-9, 3)
        load = np.zeros(net.n_nodes * 3)
        load[right * 3 + 1] = 1.0 / len(right)
        return abs(_solve(matrix, net, fixed, load)[right * 3 + 1].mean())

    assert shear(False) / shear(True) > 10


# --------------------------------------------------------------------------
# building the network
# --------------------------------------------------------------------------


def test_crossing_beads_share_a_node():
    net = build_network([LineString([(0, 0), (10, 0)]), LineString([(5, -5), (5, 5)])])
    assert net.branch_nodes == 1
    assert net.n_elements == 4, "both beads are split at the crossing"


def test_beads_that_only_come_close_do_not_share_a_node():
    net = build_network([LineString([(0, 0), (10, 0)]), LineString([(0, 1), (10, 1)])])
    assert net.branch_nodes == 0


def test_welding_joins_a_bead_end_to_the_bead_it_lands_on():
    """Infill runs into the perimeter so the two fuse, but their centrelines
    stop short of crossing.  Without the weld the fill floats free and the
    stiffness matrix is singular."""
    wall = LineString([(0, 0), (10, 0)])
    fill = LineString([(5, 0.05), (5, 5)])          # ends just above the wall
    assert build_network([wall, fill]).branch_nodes == 0
    assert build_network([wall, fill], weld=0.1).branch_nodes == 1


def test_welding_leaves_already_touching_beads_alone():
    touching = [LineString([(0, 0), (5, 0)]), LineString([(5, 0), (10, 0)])]
    assert len(weld_touching_ends(touching, 0.1)) == 2


def test_an_empty_network_reports_nothing_rather_than_failing():
    report = analyse_flexibility([], 0.1, 0.1)
    assert report.elements == 0
    assert not report.solved


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------


def test_parallel_beads_read_as_fully_anisotropic():
    lines = [LineString([(0, y), (10, y)]) for y in range(5)]
    report = analyse_flexibility(lines, 0.1, 0.1)
    assert report.anisotropy == pytest.approx(1.0, abs=0.01)
    assert report.principal_angle_deg == pytest.approx(0.0, abs=1)


def test_a_square_grid_reads_as_isotropic():
    lines = ([LineString([(0, y), (10, y)]) for y in range(11)]
             + [LineString([(x, 0), (x, 10)]) for x in range(11)])
    assert analyse_flexibility(lines, 0.1, 0.1).anisotropy == pytest.approx(0.0, abs=0.02)


def test_the_principal_angle_follows_the_beads():
    lines = [LineString([(0, 0), (10, 10)]), LineString([(0, 2), (8, 10)])]
    assert analyse_flexibility(lines, 0.1, 0.1).principal_angle_deg == pytest.approx(45, abs=2)


# --------------------------------------------------------------------------
# what it says about the real patterns
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stiffness(electrode):
    """Bending and stretch compliance of each pattern at a matched density.

    Module-scoped because the frame solve is the expensive part; the printer
    is built here rather than taken from the function-scoped fixture.
    """
    from electrode_slicer import Printer
    from electrode_slicer.slicer import SliceConfig

    machine = Printer(name="test-rig", plate_x=220.0, plate_y=300.0, z_travel=40.0,
                      line_width=0.1, layer_height=0.1)
    out = {}
    scales, layers = {}, {}
    for name in ("lines", "serpentine", "hilbert", "grid", "stars", "honeycomb"):
        result = run_sweep(
            electrode, machine,
            SweepRequest(patterns=[name], densities=[0.6], reference=0),
            SliceConfig(flexibility=True), scale_cache=scales, layer_cache=layers,
        )
        out[name] = result.layers_by_variant[0][-1].flex
    return out


def test_every_pattern_produces_a_solvable_frame(stiffness):
    for name, flex in stiffness.items():
        assert flex.solved, f"{name} did not solve"
        assert flex.bend_compliance > 0
        assert flex.elements > 0


def test_crossing_patterns_are_the_stiffest(stiffness):
    """The finding that matters: the fills that make the most humps are also
    the least able to conform, so texture and compliance cannot be varied
    independently."""
    crossing = max(stiffness[n].bend_compliance for n in ("grid", "stars"))
    single_stroke = min(stiffness[n].bend_compliance for n in ("serpentine", "hilbert"))
    assert single_stroke > 5 * crossing


def test_single_stroke_fills_are_the_most_compliant(stiffness):
    ranked = sorted(stiffness, key=lambda n: stiffness[n].bend_compliance, reverse=True)
    assert set(ranked[:2]) == {"serpentine", "hilbert"}


def test_crossings_show_up_as_braced_joints(stiffness):
    """The mechanism, not just the outcome."""
    for crossing in ("grid", "stars", "honeycomb"):
        assert stiffness[crossing].braced_fraction > 0.3
    for chain in ("serpentine", "hilbert"):
        assert stiffness[chain].braced_fraction < 0.1


def test_stretch_separates_them_even_more_sharply(stiffness):
    assert stiffness["serpentine"].stretch_compliance > \
        100 * stiffness["stars"].stretch_compliance


def test_flexibility_is_off_unless_asked_for(electrode, printer):
    result = run_sweep(electrode, printer,
                       SweepRequest(patterns=["grid"], densities=[0.4], reference=0))
    assert not result.layers_by_variant[0][-1].flex.solved
