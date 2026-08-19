from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "capstone_demo.ipynb"
GENERATOR = ROOT / "scripts" / "update_capstone_notebook.py"
COLAB_REQUIREMENTS = ROOT / "requirements-colab.txt"


def load_generator():
    spec = importlib.util.spec_from_file_location("update_capstone_notebook", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_notebook(path: Path = NOTEBOOK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_setup_function(name: str):
    setup = "".join(load_notebook()["cells"][1]["source"])
    tree = ast.parse(setup)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    metadata = SimpleNamespace(
        PackageNotFoundError=importlib.metadata.PackageNotFoundError,
        version=Mock(),
    )
    namespace = {
        "importlib": SimpleNamespace(metadata=metadata, invalidate_caches=Mock()),
        "subprocess": SimpleNamespace(run=Mock()),
        "sys": SimpleNamespace(executable="python"),
    }
    exec(  # noqa: S102 -- execute one controlled function from the generated notebook
        compile(ast.Module(body=[function], type_ignores=[]), "<setup>", "exec"),
        namespace,
    )
    return namespace[name], namespace


def test_colab_overlay_does_not_pin_preloaded_binary_stack() -> None:
    requirements = {
        line.split("==", 1)[0].lower()
        for line in COLAB_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert {
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "soundfile",
        "librosa",
        "sentencepiece",
        "sacrebleu",
    } <= requirements
    assert requirements.isdisjoint(
        {"torch", "numpy", "scipy", "pandas", "matplotlib", "pyarrow"}
    )


def test_generated_setup_is_branch_and_binary_restart_safe() -> None:
    notebook = load_notebook()
    badge = "".join(notebook["cells"][0]["source"])
    setup = "".join(notebook["cells"][1]["source"])

    assert "blob/main/capstone_demo.ipynb" in badge
    assert 'REPOSITORY_CANDIDATE_REFS = ("main", "codex/c1-integration")' in setup
    assert '"--branch", REPOSITORY_REF' in setup
    assert '"checkout", "--detach", "FETCH_HEAD"' in setup
    assert '"requirements-colab.txt"' in setup
    assert "changed_binary_packages" in setup
    assert "os.kill(os.getpid(), signal.SIGKILL)" in setup
    assert "remove_incompatible_torchao()" in setup
    assert "is_torchao_available()" in setup
    assert '"main/artifacts/comparison-v2"' not in setup
    assert '"main/artifacts/gpu-handoff"' not in setup
    ast.parse(setup)


def test_setup_removes_incompatible_optional_torchao() -> None:
    remove, namespace = load_setup_function("remove_incompatible_torchao")
    missing = importlib.metadata.PackageNotFoundError("torchao")
    namespace["importlib"].metadata.version.side_effect = ["0.10.0", missing]

    assert remove() == "0.10.0"
    namespace["subprocess"].run.assert_called_once_with(
        ["python", "-m", "pip", "uninstall", "-y", "torchao"],
        check=True,
    )
    namespace["importlib"].invalidate_caches.assert_called_once_with()


def test_setup_accepts_missing_optional_torchao() -> None:
    remove, namespace = load_setup_function("remove_incompatible_torchao")
    namespace["importlib"].metadata.version.side_effect = (
        importlib.metadata.PackageNotFoundError("torchao")
    )

    assert remove() is None
    namespace["subprocess"].run.assert_not_called()
    namespace["importlib"].invalidate_caches.assert_not_called()


def test_setup_preserves_compatible_torchao() -> None:
    remove, namespace = load_setup_function("remove_incompatible_torchao")
    namespace["importlib"].metadata.version.return_value = "0.17.0"

    assert remove() is None
    namespace["subprocess"].run.assert_not_called()
    namespace["importlib"].invalidate_caches.assert_not_called()


def test_setup_rejects_incompatible_torchao_that_remains_installed() -> None:
    remove, namespace = load_setup_function("remove_incompatible_torchao")
    namespace["importlib"].metadata.version.side_effect = ["0.10.0", "0.10.0"]

    with pytest.raises(
        RuntimeError,
        match="Failed to remove incompatible torchao 0.10.0",
    ):
        remove()

    namespace["subprocess"].run.assert_called_once_with(
        ["python", "-m", "pip", "uninstall", "-y", "torchao"],
        check=True,
    )
    namespace["importlib"].invalidate_caches.assert_called_once_with()


def test_notebook_generator_is_deterministic_and_clears_outputs(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    generated = tmp_path / "capstone_demo.ipynb"
    generated.write_bytes(NOTEBOOK.read_bytes())
    generator.NOTEBOOK = generated

    generator.main()
    first = generated.read_bytes()
    generator.main()
    second = generated.read_bytes()

    assert first == second
    notebook = load_notebook(generated)
    assert len(notebook["cells"]) == 26
    assert len({cell["id"] for cell in notebook["cells"]}) == 26
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
