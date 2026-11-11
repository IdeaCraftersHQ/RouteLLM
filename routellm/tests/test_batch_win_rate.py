"""`Controller.batch_calculate_win_rate` parallel path matches the serial one."""

import sys

import pandas as pd

from routellm.controller import Controller


class _Router:
    NO_PARALLEL = False

    def calculate_strong_win_rate(self, prompt):
        return (sum(map(ord, prompt)) % 100) / 100


def _controller(no_parallel=False, progress_bar=False):
    c = Controller.__new__(Controller)
    r = _Router()
    r.NO_PARALLEL = no_parallel
    c.routers = {"stub": r}
    c.progress_bar = progress_bar
    c.max_workers = None
    return c


def _prompts():
    return pd.Series([f"prompt {i}" * (i % 5 + 1) for i in range(40)], index=range(100, 140))


def test_parallel_matches_serial():
    prompts = _prompts()
    serial = _controller(no_parallel=True).batch_calculate_win_rate(prompts, "stub")
    parallel = _controller().batch_calculate_win_rate(prompts, "stub", max_workers=4)
    pd.testing.assert_series_equal(parallel, serial)


def test_parallel_with_progress_bar():
    prompts = _prompts()
    serial = _controller(no_parallel=True).batch_calculate_win_rate(prompts, "stub")
    parallel = _controller(progress_bar=True).batch_calculate_win_rate(prompts, "stub")
    pd.testing.assert_series_equal(parallel, serial)


def test_no_pandarallel_needed():
    _controller().batch_calculate_win_rate(_prompts(), "stub")
    assert "pandarallel" not in sys.modules
