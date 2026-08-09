"""Validate notebook structure and compile every code cell without executing it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat


def validate_notebook(
    path: Path,
    *,
    require_empty_outputs: bool = True,
    max_size_bytes: int = 1_000_000,
) -> dict[str, object]:
    size_bytes = path.stat().st_size
    if size_bytes > max_size_bytes:
        raise ValueError(
            f"Notebook is {size_bytes} bytes; limit is {max_size_bytes} bytes"
        )
    raw_text = path.read_text(encoding="utf-8")
    secret_patterns = {
        "Hugging Face token": r"\bhf_[A-Za-z0-9]{20,}\b",
        "GitHub token": r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b",
        "bearer credential": r"(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}",
        "signed or expiring URL": (
            r"https?://[^\s\"']+\?(?:[^\s\"']*&)?"
            r"(?:token|signature|x-amz-signature|x-amz-credential|expires)="
        ),
    }
    for label, pattern in secret_patterns.items():
        if re.search(pattern, raw_text, flags=re.IGNORECASE):
            raise ValueError(f"Notebook contains a possible {label}")
    notebook = nbformat.read(str(path), as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(code_cells, start=1):
        compile(cell.source, f"{path}:code-cell-{index}", "exec")
    outputs_empty = all(not cell.get("outputs", []) for cell in code_cells)
    execution_counts_empty = all(cell.get("execution_count") is None for cell in code_cells)
    if require_empty_outputs and not outputs_empty:
        raise ValueError("Notebook contains saved code-cell outputs")
    if require_empty_outputs and not execution_counts_empty:
        raise ValueError("Notebook contains saved execution counts")
    return {
        "status": "valid",
        "path": str(path),
        "cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "size_bytes": size_bytes,
        "outputs_empty": outputs_empty,
        "execution_counts_empty": execution_counts_empty,
        "secret_scan": "passed",
        "execution_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a notebook and compile its code cells without execution"
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--allow-outputs", action="store_true")
    args = parser.parse_args()
    result = validate_notebook(
        args.notebook, require_empty_outputs=not args.allow_outputs
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
