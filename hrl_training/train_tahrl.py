#!/usr/bin/env python3
"""Compatibility entry point for the canonical root HRL trainer."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    runpy.run_path(str(ROOT / "train_tahrl.py"), run_name="__main__")


if __name__ == "__main__":
    main()
