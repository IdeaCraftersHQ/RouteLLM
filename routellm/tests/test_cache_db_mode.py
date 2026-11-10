"""The cache database holds prompts in plaintext, so it stays owner-only."""

import os
import stat

from routellm.caching import Cache, CacheConfig


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_new_database_is_owner_only(tmp_path):
    path = tmp_path / "cache.db"

    Cache(CacheConfig(enabled=True, db_path=str(path)))

    assert _mode(path) == 0o600


def test_existing_world_readable_database_is_tightened(tmp_path):
    path = tmp_path / "cache.db"
    path.write_bytes(b"")
    os.chmod(path, 0o644)

    Cache(CacheConfig(enabled=True, db_path=str(path)))

    assert _mode(path) == 0o600


def test_disabled_cache_leaves_an_existing_file_alone(tmp_path):
    path = tmp_path / "cache.db"
    path.write_bytes(b"")
    os.chmod(path, 0o644)

    Cache(CacheConfig(enabled=False, db_path=str(path)))

    assert _mode(path) == 0o644
