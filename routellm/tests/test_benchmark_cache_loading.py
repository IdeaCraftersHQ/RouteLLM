"""Tests for how the eval benchmarks load their optional .npy cache.

Each benchmark starts from a cached score table when one is present and
from empty when it is not. Absent, truncated and wrong-shaped files are
all cold starts. A bare `except` there also hid KeyboardInterrupt and
any bug in the loader; the narrowed list does not.
"""

import numpy as np
import pytest

from routellm.evals.benchmarks import _load_cache


def test_a_missing_cache_file_is_a_cold_start(tmp_path):
    assert _load_cache(str(tmp_path / "absent.npy")) == ({}, False)


def test_a_truncated_cache_file_is_a_cold_start(tmp_path):
    path = tmp_path / "corrupt.npy"
    path.write_bytes(b"not an npy at all")

    assert _load_cache(str(path)) == ({}, False)


def test_an_array_shaped_cache_is_a_cold_start(tmp_path):
    """`.item()` on a multi-element array raises ValueError."""
    path = tmp_path / "arr.npy"
    np.save(path, np.array([1, 2, 3]))

    assert _load_cache(str(path)) == ({}, False)


def test_a_good_cache_is_returned(tmp_path):
    path = tmp_path / "good.npy"
    np.save(path, np.array({"router": {"prompt": 0.5}}, dtype=object))

    assert _load_cache(str(path)) == ({"router": {"prompt": 0.5}}, True)


def test_an_empty_cache_that_loaded_is_not_a_cold_start(tmp_path):
    """A cache holding {} loaded fine; only a failure reports False."""
    path = tmp_path / "empty.npy"
    np.save(path, np.array({}, dtype=object))

    assert _load_cache(str(path)) == ({}, True)


def test_an_interrupt_during_the_load_is_not_swallowed(tmp_path, monkeypatch):
    """A bare `except` would turn Ctrl-C into a silent empty cache.

    KeyboardInterrupt does not inherit from Exception, so only the bare
    form caught it. The narrowed list lets it through.
    """
    path = tmp_path / "good.npy"
    np.save(path, np.array({"router": {}}, dtype=object))

    def boom(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "load", boom)

    with pytest.raises(KeyboardInterrupt):
        _load_cache(str(path))
