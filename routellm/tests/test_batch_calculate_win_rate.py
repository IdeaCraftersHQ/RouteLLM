"""`Controller.batch_calculate_win_rate` scores a prompt series for evals.

`calibrate_threshold.py` and the eval benchmarks call it to get one strong
win rate per prompt before choosing thresholds. The scripts must also build
a controller with kwargs `Controller` accepts.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from routellm.controller import Controller, RoutingError

CALIBRATE_PY = Path(__file__).resolve().parents[1] / "calibrate_threshold.py"


@pytest.fixture
def controller():
    return Controller(routers=["random"], strong_model="strong", weak_model="weak")


def test_scores_one_win_rate_per_prompt(controller):
    prompts = pd.Series(["a", "b", "c"], index=[10, 11, 12])

    rates = controller.batch_calculate_win_rate(prompts, "random")

    assert list(rates.index) == [10, 11, 12]
    assert all(0 <= r <= 1 for r in rates)


def test_progress_bar_path_scores_too():
    c = Controller(routers=["random"], strong_model="strong", weak_model="weak", progress_bar=True)

    assert len(c.batch_calculate_win_rate(pd.Series(["a", "b"]), "random")) == 2


def test_unknown_router_is_refused(controller):
    with pytest.raises(RoutingError):
        controller.batch_calculate_win_rate(pd.Series(["a"]), "nope")


def test_calibrate_threshold_passes_only_controller_kwargs():
    import inspect

    accepted = set(inspect.signature(Controller.__init__).parameters)
    tree = ast.parse(CALIBRATE_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Controller":
            used = {k.arg for k in node.keywords}
            assert used <= accepted, used - accepted
