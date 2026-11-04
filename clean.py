#!/usr/bin/env python3
"""Remove generated release-local engine and Python environment files."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
TARGETS = (
    ROOT / "engine/llama.cpp-n3k/build",
    ROOT / "engine/llama.cpp-n3k",
    ROOT / "ui/.venv",
)


for target in TARGETS:
    if target.is_symlink():
        raise RuntimeError(f"refusing to remove symlink: {target}")
    if target.exists():
        if not target.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"refusing to remove path outside release tree: {target}")
        shutil.rmtree(target)
        print(f"removed {target.relative_to(ROOT)}")

print("Models in models/ were left untouched.")
