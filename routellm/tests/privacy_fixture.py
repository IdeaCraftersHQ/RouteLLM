"""Privacy suite fixture: two data classes, three reaches, fake leaves.

Classes
    `private` data must stay on the host; `general` data may travel.
    Today a class is expressed by the tier a request is addressed to:
    the `private` tier reaches local leaves only, the `general` tier
    reaches nothing but cloud and gated leaves. Later cases add the other
    ways of declaring a class (header, recognizer, classifier) on top of
    this same world.

Reaches
    local        a loopback engine: private data may go here
    cloud        a hosted provider reached directly
    gated        a metered gateway reached behind an egress allow-list

Every leaf is a `FakeUpstream`. A leaf that is not local counts its
hits; a privacy case asserts that count stays at zero for the private
class. The `general` tier is the control: it must reach the cloud and
gated fakes, or a zero elsewhere proves nothing.

Cases are plain functions registered with `case(...)`. A case drives
the app through `World` and returns the observation it wants checked;
`run_cases` runs a batch in one subprocess (the server parses
`sys.argv` at import, so one process holds one app) and returns every
case's observation plus the hits each leaf took. Adding a case is one
decorated function in a module the subprocess imports.

Nothing here leaves the loopback interface.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.tests.fake_upstream import FakeUpstream  # noqa: F401  used by the subprocess script

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Leaf name -> reach. A leaf is local, cloud or gated.
LEAVES = {
    "local_main": "local",
    "local_small": "local",
    "local_audio": "local",
    "cloud_main": "cloud",
    "gated_main": "gated",
}
LOCAL = tuple(n for n, r in LEAVES.items() if r == "local")
OFF_HOST = tuple(n for n, r in LEAVES.items() if r != "local")

CASE_MODULES = [
    "routellm.tests.privacy_cases_structural",
    "routellm.tests.privacy_cases_clients",
    "routellm.tests.privacy_cases_selectors",
]


#: Quality per leaf, so a selector tier orders deterministically. The
#: cloud leaf is the best on purpose: a selector that lets it through wins it.
QUALITY = {
    "local_main": 80,
    "local_small": 40,
    "local_audio": 60,
    "cloud_main": 90,
    "gated_main": 70,
}


def endpoints(bases: dict[str, str]) -> dict:
    """Return the endpoint section: one leaf per name in `LEAVES`.

    `local_main` takes images and a long context, `local_small` takes
    short text only, `local_audio` hears audio. Every leaf is strict, so
    a request a leaf does not declare is refused, never guessed at.
    """

    def leaf(name, tags, caps):
        return {
            "quality": QUALITY[name],
            "model": f"openai/model-{name.replace('_', '-')}",
            "api_base": bases[name],
            "api_key_env": "FAKE_KEY",
            "tags": tags,
            "strict": True,
            "extra": {"max_retries": 0},
            "capabilities": caps,
        }

    return {
        "local_main": leaf(
            "local_main",
            ["local", "tools"],
            {"vision": True, "tools": True, "context": 65536, "modalities_in": ["text", "image"]},
        ),
        "local_small": leaf(
            "local_small",
            ["local", "tools"],
            {"vision": False, "tools": True, "context": 8192, "modalities_in": ["text"]},
        ),
        "local_audio": leaf(
            "local_audio",
            ["local"],
            {"tools": False, "context": 65536, "modalities_in": ["text", "audio"]},
        ),
        "cloud_main": leaf(
            "cloud_main",
            ["cloud", "tools"],
            {"vision": True, "tools": True, "context": 200000, "modalities_in": ["text", "image"]},
        ),
        "gated_main": leaf(
            "gated_main",
            ["cloud", "gated", "tools"],
            {"vision": True, "tools": True, "context": 200000, "modalities_in": ["text", "image"]},
        ),
    }


#: The local-only selector: reach local, never cloud.
SELECTOR = "tag:local !tag:cloud"


def tiers() -> dict:
    """Return the tier section.

    `private` has a local primary and a local sibling; `private_audio`
    swaps in the leaf that hears audio; `default` is what the legacy
    `router-<name>-<threshold>` form lands on and is local-only;
    `general` is the control that reaches off the host;
    `private_selector` is `private` written as selectors, local and not
    cloud on both sides, so it must land on the same two leaves.
    """
    pair = {"router": "random", "threshold": 0.0}
    return {
        "private": {**pair, "strong": "local_main", "weak": "local_small"},
        "private_audio": {**pair, "strong": "local_audio", "weak": "local_small"},
        "default": {**pair, "strong": "private", "weak": "local_small"},
        "general": {**pair, "strong": "cloud_main", "weak": "gated_main"},
        "private_selector": {
            **pair,
            "strong": {"select": SELECTOR, "order": "quality_desc"},
            "weak": {"select": SELECTOR, "order": "quality_asc"},
        },
    }


def build_config(bases: dict[str, str]) -> dict:
    return {"endpoints": endpoints(bases), "tiers": tiers()}


@dataclass
class Case:
    fn: object
    status: int | None = None
    local: dict | None = field(default_factory=dict)  # leaf -> expected hits; None: use total
    local_total: int | None = None  # with local=None: hits across all local leaves
    reaches_off_host: bool = False  # the control: off-host leaves ARE expected to be hit


CASES: dict[str, Case] = {}


def case(name, status=None, local=None, total=None, reaches_off_host=False):
    """Register a case. `status` and `local` are what the test asserts.

    `local` maps a local leaf to its expected hit count; a leaf left out
    is expected at zero. `total` replaces it when the case cannot say
    which local leaf answers. Off-host leaves are expected at zero unless
    the case is a control (`reaches_off_host=True`).
    """

    def deco(fn):
        CASES[name] = Case(
            fn, status, None if total is not None else dict(local or {}), total, reaches_off_host
        )
        return fn

    return deco


class World:
    """The fakes plus the live app, as a case sees them."""

    def __init__(self, fakes, client, server):
        self.fakes = fakes
        self.client = client
        self.server = server

    def reset(self):
        for fake in self.fakes.values():
            fake.mode = "ok"
            fake.hits.clear()
        self.server.CONTROLLER.resilience.cb_map.clear()

    def hits(self) -> dict[str, int]:
        return {name: len(fake.hits) for name, fake in self.fakes.items()}

    def fail(self, *names, mode="500"):
        for name in names or LOCAL:
            self.fakes[name].mode = mode

    def post(self, model, messages=None, **kw):
        body = {"model": model, "messages": messages or TEXT, **kw}
        return self.client.post("/v1/chat/completions", json=body)

    def stream(self, model, messages=None, **kw):
        body = {"model": model, "messages": messages or TEXT, "stream": True, **kw}
        with self.client.stream("POST", "/v1/chat/completions", json=body) as r:
            events = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
            return r.status_code, events


TEXT = [{"role": "user", "content": "private text"}]
IMAGE = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "what is this"},
        ],
    }
]
AUDIO = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
        ],
    }
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "d",
            "parameters": {"type": "object", "properties": {"p": {"type": "string"}}},
        },
    }
]

_SCRIPT = """
import importlib, json, sys
from routellm.tests import privacy_fixture as pf
fakes = {{n: pf.FakeUpstream(n) for n in pf.LEAVES}}
cfg = pf.build_config({{n: f.base for n, f in fakes.items()}})
cfg.update(json.loads({extra!r}))
open({config!r}, "w").write(json.dumps(cfg))
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]
for module in pf.CASE_MODULES:
    importlib.import_module(module)
import routellm.openai_server as server
from fastapi.testclient import TestClient
out = {{}}
with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.cache.config.enabled = False
    server.CONTROLLER.resilience.config.max_retries = 0
    for name, setting in json.loads({settings!r}).items():
        for key, value in setting.items():
            setattr(server.CONTROLLER.resilience.config, key, value)
    world = pf.World(fakes, client, server)
    for name in {names!r}:
        world.reset()
        obs = pf.CASES[name].fn(world)
        out[name] = {{"obs": obs, "hits": world.hits()}}
print("@@RESULT@@" + json.dumps(out))
"""


def run_cases(tmp_path, names, extra_config=None, resilience=None):
    """Run the named cases in one subprocess; return {case: {obs, hits}}.

    `extra_config` is merged over the fixture config (top-level keys),
    `resilience` sets `ResilienceConfig` fields before the first request.
    """
    # Selector tiers consult the models.dev catalog on construction; an
    # empty snapshot keeps that off the network (tags need no catalog).
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    script = _SCRIPT.format(
        config=str(tmp_path / "config.json"),
        extra=json.dumps(extra_config or {}),
        settings=json.dumps({"r": resilience or {}}),
        names=list(names),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            CATALOG_CACHE_ENV: str(catalog),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "FAKE_KEY": "sk-fake",
        },
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    return json.loads(line.split("@@RESULT@@", 1)[1])


def check(name, result):
    """Assert one case's observation against its registered expectation.

    The privacy claim lives here: every off-host leaf took zero hits
    unless the case is the control.
    """
    spec = CASES[name]
    hits, obs = result["hits"], result["obs"]
    if spec.status is not None:
        assert obs["status"] == spec.status, obs
    if spec.local is None:
        assert sum(hits[leaf] for leaf in LOCAL) == spec.local_total, hits
    else:
        for leaf in LOCAL:
            assert hits[leaf] == spec.local.get(leaf, 0), (leaf, hits)
    if spec.reaches_off_host:
        assert any(hits[leaf] for leaf in OFF_HOST), hits
    else:
        assert {leaf: hits[leaf] for leaf in OFF_HOST} == dict.fromkeys(OFF_HOST, 0), hits
