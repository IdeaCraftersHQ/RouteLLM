"""Entry-point plugins found through a really installed distribution.

The fixture distribution under `fixtures/plugin_dist` is installed into
a temp directory (built offline with setuptools, unpacked like a wheel
install), and a subprocess with that directory on
`sys.path` runs every scenario, so `importlib.metadata` resolves the
`routellm.decision_providers`, `routellm.decision_recognizers` and
`routellm.transforms` groups the way a user's environment does.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

HERE = Path(__file__).parent
DIST = HERE / "fixtures" / "plugin_dist"
REPO = HERE.parent.parent


BUILD = "from setuptools import build_meta; import sys; print(build_meta.build_wheel(sys.argv[1]))"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    work = tmp_path_factory.mktemp("build")
    src, wheels, site = work / "src", work / "wheels", work / "site"
    shutil.copytree(DIST, src)
    wheels.mkdir()
    site.mkdir()
    build = subprocess.run(
        [sys.executable, "-c", BUILD, str(wheels)],
        capture_output=True,
        text=True,
        cwd=src,
        timeout=120,
    )
    if build.returncode != 0:
        pytest.skip(f"cannot build the test plugin offline: {build.stderr[-300:]}")
    # Unpacking the wheel is what an install does: package + dist-info.
    with zipfile.ZipFile(wheels / build.stdout.strip().splitlines()[-1]) as wheel:
        wheel.extractall(site)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(REPO)])}
    run = subprocess.run(
        [sys.executable, str(HERE / "plugin_driver.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=site,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.splitlines()[-1])


# -- providers --------------------------------------------------------------


def test_config_naming_a_provider_plugin_loads_it(ran):
    assert ran["provider_ok"]["source"] == "model"
    assert ran["provider_ok"]["data_class"] == "private"


def test_a_provider_answer_outside_the_label_sets_is_refused(ran):
    bad = ran["provider_bad_labels"]
    assert (bad["source"], bad["error"]) == ("error", "invalid_answer")
    assert bad["data_class"] == "private"  # most restrictive, never the plugin's label
    assert "invalid_answer" in ran["provider_bad_labels_refuse"]


def test_unknown_provider_names_itself_and_the_group(ran):
    err = ran["provider_unknown"]
    assert err["type"] == "ValueError"
    assert "rllm_nope" in err["message"] and "routellm.decision_providers" in err["message"]


def test_a_provider_plugin_failing_to_load_is_named(ran):
    err = ran["provider_boom"]
    assert err["type"] == "ValueError"
    assert "rllm_boom" in err["message"]
    assert "simulated missing dependency" in err["message"]


# -- recognizers ------------------------------------------------------------


def test_a_recognizer_plugin_hit_sets_the_source(ran):
    hit = ran["recognizer_hit"]
    assert (hit["source"], hit["data_class"]) == ("recognizer", "private")
    assert hit["detail"] == {"recognizer": "wordy", "word": "swordfish"}


def test_a_recognizer_plugin_miss_falls_through_to_the_provider(ran):
    assert ran["recognizer_miss"]["source"] == "model"


def test_unknown_recognizer_type_names_itself_and_the_group(ran):
    err = ran["recognizer_unknown"]
    assert err["type"] == "ValueError"
    assert "rllm_nope" in err["message"] and "routellm.decision_recognizers" in err["message"]


def test_a_recognizer_plugin_failing_to_load_is_named(ran):
    err = ran["recognizer_boom"]
    assert err["type"] == "ValueError"
    assert "rllm_boom" in err["message"]
    assert "simulated missing dependency" in err["message"]


# -- transforms -------------------------------------------------------------


def test_config_naming_a_transform_plugin_loads_it_with_options(ran):
    assert ran["transform_ok"] == {"type": "TagTransform", "tag": "x"}


def test_unknown_transform_is_a_config_error(ran):
    err = ran["transform_unknown"]
    assert err["type"] == "ConfigError"
    assert "rllm_nope" in err["message"] and "routellm.transforms" in err["message"]


def test_a_transform_plugin_failing_to_load_is_named(ran):
    err = ran["transform_boom"]
    assert err["type"] == "ConfigError"
    assert "rllm_boom" in err["message"]
    assert "simulated missing dependency" in err["message"]
