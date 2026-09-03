"""Shared test setup for the src-package layout: put src/ and scripts/ on
sys.path (mirroring what scripts/train.py does at runtime) and force a
non-interactive matplotlib backend.

Order matters: src/ must come BEFORE the project root, because the root-level
models/ directory (checkpoint storage, no __init__.py) would otherwise shadow
the src/models code package as a namespace package.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
