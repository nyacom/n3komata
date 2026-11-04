#!/usr/bin/env python3
"""Create a clean, patched llama.cpp source tree for the N3komata MoE engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "engine"
TARGET = ENGINE_DIR / "llama.cpp-n3k"
PATCH = ENGINE_DIR / "patches/llama.cpp/moe-correction.patch"
UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"
BASE = "f5027fb21307eb264aed483242603d68c87b0fe3"


def run(args: list[str], *, cwd: Path | None = None, stdout=None) -> None:
    subprocess.run(args, cwd=cwd, check=True, stdout=stdout)


def archive_base(source_repo: Path, archive_path: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", f"{BASE}^{{commit}}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if revision != BASE:
        raise RuntimeError(f"source repository resolved base to unexpected commit: {revision}")
    with archive_path.open("wb") as output:
        run(["git", "-C", str(source_repo), "archive", "--format=tar", BASE], stdout=output)


def materialize(source_repo: Path | None) -> Path:
    if TARGET.exists():
        raise FileExistsError(
            f"refusing to replace existing engine tree: {TARGET}\n"
            "Move it yourself after preserving any local changes, then rerun this command."
        )
    if not PATCH.is_file():
        raise FileNotFoundError(f"engine patch not found: {PATCH}")

    with tempfile.TemporaryDirectory(prefix=".llama.cpp-n3k-", dir=ENGINE_DIR) as temporary:
        stage = Path(temporary)
        source = source_repo.resolve() if source_repo is not None else stage / "upstream"
        if source_repo is None:
            run(["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(source)])
            run(["git", "-C", str(source), "checkout", "--detach", BASE])

        archive_path = stage / "upstream.tar"
        archive_base(source, archive_path)

        tree = stage / "tree"
        tree.mkdir()
        with tarfile.open(archive_path, "r:") as archive:
            try:
                archive.extractall(tree, filter="data")
            except TypeError:  # Python versions before the extraction filter was added.
                archive.extractall(tree)

        run(["git", "apply", "--check", str(PATCH)], cwd=tree)
        run(["git", "apply", str(PATCH)], cwd=tree)

        patch_sha256 = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        metadata = {
            "name": "llama.cpp-n3k",
            "upstream": UPSTREAM,
            "base_commit": BASE,
            "patch": "engine/patches/llama.cpp/moe-correction.patch",
            "patch_sha256": patch_sha256,
        }
        (tree / "N3KOMATA-ENGINE.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tree, TARGET)

    return TARGET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="use an existing llama.cpp Git checkout instead of cloning upstream",
    )
    args = parser.parse_args()
    target = materialize(args.source_repo)
    print(f"Materialized patched N3komata engine at {target}")
    print(f"Base: {BASE}")
    print(f"Patch: {PATCH.relative_to(ROOT)}")
    print(
        "Build with: cmake -S engine/llama.cpp-n3k -B engine/llama.cpp-n3k/build "
        "-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release"
    )


if __name__ == "__main__":
    main()
