"""`evaluate.py` reads config through the discovery chain, not raw YAML.

`routellm/evals/evaluate.py` imports heavy optional dependencies
(matplotlib, pandarallel, ...) at module load time, so it is never
imported here — even a collection-time import can fail in an
environment that lacks them. The source is read as text and parsed
with `ast` instead, mirroring how `calibrate_threshold.py` was already
converted to `load_config(explicit=args.config).data`.
"""

from __future__ import annotations

import ast
from pathlib import Path

EVALUATE_PY = Path(__file__).resolve().parents[1] / "evals" / "evaluate.py"


def _source() -> str:
    return EVALUATE_PY.read_text()


def test_evaluate_py_does_not_raw_load_the_config_file():
    """No more `yaml.safe_load(open(` — that skipped the discovery chain."""
    assert "yaml.safe_load(open(" not in _source()


def test_evaluate_py_uses_load_config_for_the_explicit_layer():
    """The controller's config comes from `load_config(explicit=...)`."""
    assert "load_config(explicit=args.config)" in _source()


def test_evaluate_py_imports_load_config_from_routellm_config():
    """The import is present at module level, via an AST walk."""
    tree = ast.parse(_source(), filename=str(EVALUATE_PY))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "routellm.config"
        for alias in node.names
    }
    assert "load_config" in imported


def test_evaluate_py_no_longer_imports_yaml_directly():
    """`yaml` was only used for the raw load; the bare import is gone too."""
    tree = ast.parse(_source(), filename=str(EVALUATE_PY))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "yaml" not in imported_modules
