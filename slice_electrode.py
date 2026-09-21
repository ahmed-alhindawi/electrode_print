#!/usr/bin/env python3
"""Slice an STL into SVG toolpaths for the Voltera NOVA.

A thin wrapper so the tool stays runnable as a script.  The implementation
lives in the ``electrode_slicer`` package under ``src/``; see its docstring
for how the pieces fit together.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from electrode_slicer.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
