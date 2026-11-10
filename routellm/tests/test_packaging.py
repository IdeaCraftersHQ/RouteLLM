"""What `pyproject.toml` declares, and the CI that exercises it.

These read the manifest and the workflow as data. They cannot prove an
install resolves, but they catch the drift that matters: a dependency a
code path imports but nobody declared, a range that lost a bound, and a
CI matrix that no longer tests the bounds the manifest advertises.
"""

import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"


@pytest.fixture(scope="module")
def meta():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]


def _requirement(requirements, name):
    found = [Requirement(r) for r in requirements if Requirement(r).name.lower() == name]
    assert len(found) == 1, f"{name}: expected one requirement, found {len(found)}"
    return found[0]


def test_mcp_is_declared_in_the_serve_extra(meta):
    # litellm's OpenAI path imports `mcp.types` on any request that carries `tools`.
    req = _requirement(meta["optional-dependencies"]["serve"], "mcp")
    assert req.specifier.contains("2.3.0")


def test_litellm_has_a_tested_range(meta):
    spec = _requirement(meta["dependencies"], "litellm").specifier
    lower = [s for s in spec if s.operator == ">="]
    upper = [s for s in spec if s.operator in ("<", "<=")]
    assert [s.version for s in lower] == ["1.106.4"]
    assert len(upper) == 1


def test_numpy_is_not_capped_below_two(meta):
    # pyarrow 26 refuses to import under numpy 1.x, and `datasets` pulls pyarrow
    # with no upper bound, so the `<2` cap made the `ml` extra uninstallable
    # as a working environment.
    spec = _requirement(meta["dependencies"], "numpy").specifier
    assert spec.contains(Version("2.0.0"))


def test_ci_matrix_tests_both_ends_of_the_litellm_range(meta):
    spec = _requirement(meta["dependencies"], "litellm").specifier
    lower = next(s.version for s in spec if s.operator == ">=")

    workflow = yaml.safe_load(WORKFLOW.read_text())
    pins = [
        entry["litellm"]
        for job in workflow["jobs"].values()
        for entry in job.get("strategy", {}).get("matrix", {}).get("include", [])
        if "litellm" in entry
    ]
    assert f"litellm=={lower}" in pins
    # The newest allowed is the manifest's own range, so the two cannot drift.
    assert any(Requirement(pin).specifier == spec for pin in pins)


def test_ci_runs_without_extras_and_with_pairing():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    installs = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "pip install" in step.get("run", "")
    ]
    assert any('".[dev,serve]"' in run for run in installs)
    assert any("pairing" in run for run in installs)
