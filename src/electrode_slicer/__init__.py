"""Slice an STL into SVG toolpaths for a dispensing printer.

The package is split so that each concern lives in one place:

* :mod:`~electrode_slicer.printer` -- the machine: build volume, bead width,
  layer height.
* :mod:`~electrode_slicer.io` -- reading STLs and writing SVGs and manifests.
* :mod:`~electrode_slicer.patterns` -- the abstract infill pattern and every
  implementation of it.
* :mod:`~electrode_slicer.slicer` -- layers, walls and density calibration.
* :mod:`~electrode_slicer.metrics` -- how much the toolpath crosses itself.
* :mod:`~electrode_slicer.layout` -- arranging a sweep on the plate.
* :mod:`~electrode_slicer.sweep` -- the pipeline the CLI and web app share.
"""

from __future__ import annotations

from .io import LoadedMesh, SvgStyle, load_mesh, load_mesh_bytes
from .layout import LayoutConfig, Variant, build_variants, layout
from .metrics import OverlapStats, analyse_overlap
from .patterns import FillContext, InfillPattern, all_patterns, get_pattern, pattern_names
from .printer import PRINTERS, VOLTERA_NOVA, Printer, get_printer
from .slicer import LayerPaths, SliceConfig, Slicer
from .sweep import SweepRequest, SweepResult, run_sweep

__version__ = "0.4.0"

__all__ = [
    "FillContext",
    "InfillPattern",
    "LayerPaths",
    "LayoutConfig",
    "LoadedMesh",
    "OverlapStats",
    "PRINTERS",
    "Printer",
    "SliceConfig",
    "Slicer",
    "SvgStyle",
    "SweepRequest",
    "SweepResult",
    "VOLTERA_NOVA",
    "Variant",
    "all_patterns",
    "analyse_overlap",
    "build_variants",
    "get_pattern",
    "get_printer",
    "layout",
    "load_mesh",
    "load_mesh_bytes",
    "pattern_names",
    "run_sweep",
    "__version__",
]
