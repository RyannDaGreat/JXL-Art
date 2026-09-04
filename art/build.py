#!/usr/bin/env python3
"""
Compile JXL-art tree sources into .jxl files (the deliverables) and .png previews.

Usage:
    python3.10 art/build.py            # build every art/trees/*.tree
    python3.10 art/build.py digits     # build one piece
    python3.10 art/build.py digits --max_bytes 512

Each .jxl must be <= max_bytes (code-golf limit, default 1024). Violations raise loudly.
"""
import subprocess
from pathlib import Path

import fire

ART_DIR = Path(__file__).resolve().parent
TREE_DIR = ART_DIR / "trees"
OUT_DIR = ART_DIR / "out"
GOLF_LIMIT_BYTES = 1024


def compile_tree(tree_path, jxl_path, png_path):
    """
    Command. Runs jxl_from_tree then djxl. Returns the .jxl size in bytes.

    Examples:
        >>> # compile_tree(Path('art/trees/digits.tree'), Path('art/out/digits.jxl'), Path('art/out/digits.png'))
        >>> # 213
    """
    subprocess.run(["jxl_from_tree", str(tree_path), str(jxl_path)], check=True)
    subprocess.run(["djxl", str(jxl_path), str(png_path)], check=True, capture_output=True)
    return jxl_path.stat().st_size


def build(name=None, max_bytes=GOLF_LIMIT_BYTES):
    """
    Command. Builds one piece (by stem name) or all pieces, printing sizes; raises if any exceeds max_bytes.

    Examples:
        >>> # build()            -> prints "digits.jxl  213 bytes  OK" per piece
        >>> # build('flag', 900) -> raises AssertionError if flag.jxl > 900 bytes
    """
    OUT_DIR.mkdir(exist_ok=True)
    trees = sorted(TREE_DIR.glob("*.tree")) if name is None else [TREE_DIR / f"{name}.tree"]
    if not trees:
        raise FileNotFoundError(f"no .tree files found for {name!r} in {TREE_DIR}")
    oversized = []
    for tree in trees:
        size = compile_tree(tree, OUT_DIR / f"{tree.stem}.jxl", OUT_DIR / f"{tree.stem}.png")
        verdict = "OK" if size <= max_bytes else f"TOO BIG (> {max_bytes})"
        print(f"{tree.stem}.jxl  {size} bytes  {verdict}")
        if size > max_bytes:
            oversized.append((tree.stem, size))
    assert not oversized, f"code-golf limit exceeded: {oversized}"


if __name__ == "__main__":
    fire.Fire(build)
