"""Shared fixtures.

``pythonpath = ["src"]`` in pyproject.toml puts the package on the path, so
the tests import ``electrode_slicer`` the same way any consumer would.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import box

from electrode_slicer import Printer, SliceConfig, Slicer, load_mesh
from electrode_slicer.layout import LayoutConfig

ROOT = Path(__file__).resolve().parents[1]
STL = ROOT / "model" / "electrode_6mm.stl"

needs_model = pytest.mark.skipif(not STL.exists(), reason="model/electrode_6mm.stl not present")


@pytest.fixture(scope="session")
def stl_path() -> Path:
    if not STL.exists():
        pytest.skip("model/electrode_6mm.stl not present")
    return STL


@pytest.fixture
def square():
    """A 10 x 10 mm region, big enough for every pattern to show structure."""
    return box(0, 0, 10, 10)


@pytest.fixture
def printer() -> Printer:
    """A test printer with a round bead width, so sums stay checkable."""
    return Printer(
        name="test-rig", plate_x=220.0, plate_y=300.0, z_travel=40.0,
        line_width=0.25, layer_height=0.1, margin=10.0, resolution=0.005,
    )


@pytest.fixture
def config() -> SliceConfig:
    return SliceConfig()


@pytest.fixture
def slicer(printer, config) -> Slicer:
    return Slicer(printer, config)


@pytest.fixture
def layout_config() -> LayoutConfig:
    return LayoutConfig()


@pytest.fixture(scope="session")
def electrode(stl_path):
    """The real electrode, loaded once and shared -- loading is not free."""
    return load_mesh(stl_path)
