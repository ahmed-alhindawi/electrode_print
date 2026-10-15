"""``python -m electrode_slicer.web.pages [out]``: build the static site."""

from __future__ import annotations

import sys

from .build import build

if __name__ == "__main__":
    out = build(sys.argv[1] if len(sys.argv) > 1 else "build/pages")
    print(f"static site written to {out}")
