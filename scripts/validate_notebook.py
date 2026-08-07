"""Validate notebook structure and compile every code cell without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


def validate_notebook(path: Path, *, require_empty_outputs: bool = True) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(code_cells, start=1):
        compile(cell.source, f"{path}:code-cell-{index}", "exec")
    outputs_empty = all(not cell.get("outputs", []) for cell in code_cells)
    if require_empty_outputs and not outputs_empty:
        raise ValueError("Notebook contains saved code-cell outputs")
    return {
        "status": "valid",
        "path": str(path),
        "cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "outputs_empty": outputs_empty,
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
