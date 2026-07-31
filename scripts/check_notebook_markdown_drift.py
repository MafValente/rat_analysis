from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a notebook against its git HEAD version and report markdown "
            "cells that were added, removed, or changed."
        )
    )
    parser.add_argument("notebooks", nargs="+", help="Notebook paths to inspect.")
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref to compare against. Default: HEAD",
    )
    return parser.parse_args()


def load_notebook_from_disk(path: Path) -> dict:
    return json.loads(path.read_text())


def load_notebook_from_git(path: Path, ref: str) -> dict:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(
            f"Could not read {path} from git ref {ref}: {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def markdown_cells(nb: dict) -> list[tuple[int, str]]:
    cells: list[tuple[int, str]] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            cells.append((idx, "".join(cell.get("source", []))))
    return cells


def short_preview(text: str) -> str:
    first_line = text.strip().splitlines()
    if not first_line:
        return "<empty markdown cell>"
    preview = first_line[0]
    return preview if len(preview) <= 100 else preview[:97] + "..."


def compare_markdown(path: Path, ref: str) -> int:
    current = markdown_cells(load_notebook_from_disk(path))
    baseline = markdown_cells(load_notebook_from_git(path, ref))

    changed = False
    max_len = max(len(current), len(baseline))

    for pos in range(max_len):
        base = baseline[pos] if pos < len(baseline) else None
        curr = current[pos] if pos < len(current) else None

        if base == curr:
            continue

        changed = True
        if base is None and curr is not None:
            print(
                f"{path}: added markdown cell at notebook index {curr[0]}: "
                f"{short_preview(curr[1])}"
            )
            continue
        if curr is None and base is not None:
            print(
                f"{path}: removed markdown cell formerly at notebook index {base[0]}: "
                f"{short_preview(base[1])}"
            )
            continue

        assert base is not None and curr is not None
        print(
            f"{path}: changed markdown cell "
            f"(git index {base[0]} -> current index {curr[0]})"
        )
        print(f"  git:     {short_preview(base[1])}")
        print(f"  current: {short_preview(curr[1])}")

    if not changed:
        print(f"{path}: no markdown drift relative to {ref}")
        return 0
    return 1


def main() -> int:
    args = parse_args()
    exit_code = 0

    for notebook in args.notebooks:
        path = Path(notebook)
        if not path.exists():
            print(f"{path}: file not found", file=sys.stderr)
            exit_code = 2
            continue
        try:
            exit_code = max(exit_code, compare_markdown(path, args.ref))
        except Exception as exc:  # pragma: no cover - CLI error reporting
            print(f"{path}: {exc}", file=sys.stderr)
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
