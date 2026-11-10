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

import hashlib
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
    "routellm.tests.privacy_cases_class_policy",
    "routellm.tests.privacy_cases_egress_gate",
    "routellm.tests.privacy_cases_policy_routing",
    "routellm.tests.privacy_cases_aux",
]

#: The caller every class-policy case sends as, and its key.
POLICY_KEY = "sk-suite-0000"


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
            "tags": [*tags, "jur:ca"],
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


def policy_tiers() -> dict:
    """Extra tiers for the class-policy world, each a way a leaf could be reached.

    None is in any class's tiers at startup (the startup walk would refuse
    it); a case admits one to the class it attacks with `World.admit`, the
    stand-in for a config change the startup check never saw.
    """
    pair = {"router": "pin_strong", "threshold": 0.0}
    return {
        **gate_tiers(),
        "leaky_both": {**pair, "strong": "cloud_main", "weak": "gated_main"},
        "leaky_forced": {**pair, "strong": "cloud_main", "weak": "local_audio"},
        "leaky_forced_local": {**pair, "strong": "cloud_main", "weak": "local_main"},
        "leaky_sibling": {**pair, "strong": "local_main", "weak": "cloud_main"},
        "leaky_pool": {
            **pair,
            "strong": {"select": "tag:jur:ca", "order": "quality_asc", "pool": 3},
            "weak": "cloud_main",
        },
    }


def gate_tiers() -> dict:
    """Tiers with the gated leaf on every path a request can reach it by.

    The `leaky` class may use all of them by policy; the gate world's
    `egress_gates` entry is what refuses it the gated leaf.
    """
    pair = {"router": "pin_strong", "threshold": 0.0}
    return {
        "gate_primary": {**pair, "strong": "gated_main", "weak": "local_small"},
        "gate_forced": {**pair, "strong": "gated_main", "weak": "local_audio"},
        "gate_sibling": {**pair, "strong": "local_main", "weak": "gated_main"},
        "gate_pool": {
            **pair,
            "strong": {"select": "tag:jur:ca", "order": "quality_asc", "pool": 4},
            "weak": "local_main",
        },
    }


#: The gate world's extra config: a gate on the `gated` tag that refuses
#: the `leaky` class, and a detector whose endpoint is the gated leaf.
GATE_CONFIG = {
    "egress_gates": [
        {
            "name": "gated",
            "select": "tag:gated",
            "deny_classes": ["leaky"],
            "models": {"allow": ["openai/model-gated-*"]},
        }
    ],
    "intents": {"detector": "litellm", "model": "gated_main", "tiers": {"code": "gate_sibling"}},
}


#: The auxiliary-call world's extra config: a detector, and (added by
#: the subprocess, which knows the fakes' ports) an `embedding` endpoint
#: and a router-that-embeds tier, all pointing at the cloud fake. Every
#: aux call is therefore a remote call a restricted class must never make.
AUX_CONFIG = {
    "aux": "cloud_main",
    "intents": {"detector": "litellm", "model": "cloud_main", "tiers": {"code": "default"}},
}


#: Route world: header names, labels and projects of its `policy_routing:`.
ROUTE_PROJECT_HEADER = "X-Route-Project"
ROUTE_CLIENT_HEADER = "X-Route-Client"
OPEN_PROJECT = "open-project"
RESTRICTED_PROJECT = "restricted-project"  # a policy the suite caller may not select
TAGGED_PROJECT = "tagged-project"  # general class, but no `cloud`-tagged endpoint is a candidate


def route_config() -> dict:
    """The route world's extra config: an entry tier and a `policy_routing:` section.

    The default policy is the fail-closed one: its floor is `private` and
    it may use the private tier only, so a request with no project, an
    unknown one or a malformed one stays on the host. `open-project`
    lowers the floor to `general` and may also use `general`. The client
    rule is listed before the class rule on purpose: it is how a
    misconfigured policy would try to send a private request to the cloud.
    """
    pair = {"router": "random", "threshold": 0.0}
    return {
        "tiers": {
            **tiers(),
            **policy_tiers(),
            "route_entry": {**pair, "strong": "local_main", "weak": "local_small"},
            "general_local": {**pair, "strong": "local_main", "weak": "local_small"},
        },
        "comparison": {"enabled": True, "compare_allowed_models": ["local_main", "cloud_main"]},
        "policy_routing": {
            "attributes": {
                "project": {"header": ROUTE_PROJECT_HEADER},
                "client": {"header": ROUTE_CLIENT_HEADER},
            },
            "key": "project",
            "entry": ["route_entry"],
            "rules": [
                {"when": {"client": "ci"}, "tier": "general"},
                {"when": {"class": "private"}, "tier": "private"},
            ],
            "else": "general",
            "policies": {
                "default": {"data_class": "private", "tiers": ["private"], "fallback": "private"},
                OPEN_PROJECT: {"data_class": "general", "tiers": ["private", "general"]},
                RESTRICTED_PROJECT: {"data_class": "general", "tiers": ["private", "general"]},
                TAGGED_PROJECT: {
                    "data_class": "general",
                    "tiers": ["private", "general_local"],
                    "fallback": "general_local",
                    "deny_tags": ["cloud"],
                },
            },
        },
    }


def class_policy_files(tmp_path: Path, route: bool = False) -> dict:
    """Write the policy and the keys file; return the config keys naming them.

    `route` writes the two-class policy of the route world, ordered least
    to most restrictive (`general`, `private`), which is the order policy
    routing reads restrictiveness from.
    """
    leaky = list(policy_tiers())
    gate = list(gate_tiers())
    policy = {
        "classes": {
            "private": {
                "treatment": "block",
                "reach": ["local"],
                "jur": ["ca"],
                "tiers": ["private", "private_audio", "private_selector", "default"],
            },
            "leaky": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": leaky},
            "general": {
                "treatment": "raw",
                "reach": ["local", "cloud"],
                "tiers": ["general", *gate],
            },
        },
        "principals": {"suite": {"classes": ["private", "leaky", "general"]}},
    }
    if route:
        policy["classes"] = {
            "general": {
                "treatment": "raw",
                "reach": ["local", "cloud"],
                "tiers": ["general", "general_local"],
            },
            "private": policy["classes"]["private"],
        }
        policy["principals"] = {
            "suite": {
                "classes": ["general", "private"],
                "projects": [OPEN_PROJECT, TAGGED_PROJECT],
            }
        }
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    digest = hashlib.sha256(POLICY_KEY.encode()).hexdigest()
    (tmp_path / "keys.yaml").write_text(
        f'keys:\n  - key_sha256: "{digest}"\n    principal: suite\n'
    )
    return {
        "policy": str(tmp_path / "policy.json"),
        "auth": {"keys_file": str(tmp_path / "keys.yaml")},
        "tiers": {**tiers(), **policy_tiers()},
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
    policy: bool = False  # runs in the class-policy world (see `run_cases`)
    gate: bool = False  # runs in the egress-gate world (policy world plus gates)
    route: bool = False  # runs in the route world (two classes plus `policy_routing:`)
    aux: bool = False  # runs in the auxiliary-call world (policy world plus `AUX_CONFIG`)


CASES: dict[str, Case] = {}


def case(
    name,
    status=None,
    local=None,
    total=None,
    reaches_off_host=False,
    policy=False,
    gate=False,
    route=False,
    aux=False,
):
    """Register a case. `status` and `local` are what the test asserts.

    `local` maps a local leaf to its expected hit count; a leaf left out
    is expected at zero. `total` replaces it when the case cannot say
    which local leaf answers. Off-host leaves are expected at zero unless
    the case is a control (`reaches_off_host=True`). `policy=True` runs
    it only in the class-policy world, `gate=True` only in the egress-gate
    world (which is the policy world plus `GATE_CONFIG`), `route=True` only
    in the route world (see `route_config`).
    """

    def deco(fn):
        CASES[name] = Case(
            fn,
            status,
            None if total is not None else dict(local or {}),
            total,
            reaches_off_host,
            policy or gate or route or aux,
            gate,
            route,
            aux,
        )
        return fn

    return deco


class World:
    """The fakes plus the live app, as a case sees them."""

    def __init__(self, fakes, client, server, class_policy=False):
        self.fakes = fakes
        self.client = client
        self.server = server
        self.class_policy = class_policy
        self._admitted: list[tuple[str, str]] = []

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {POLICY_KEY}"} if self.class_policy else {}

    def admit(self, data_class, tier):
        """Add `tier` to a class's tiers after startup (undone by `reset`)."""
        self.server.CONTROLLER.policy.classes[data_class].tiers.append(tier)
        self._admitted.append((data_class, tier))

    def reset(self):
        for data_class, tier in self._admitted:
            self.server.CONTROLLER.policy.classes[data_class].tiers.remove(tier)
        self._admitted.clear()
        for fake in self.fakes.values():
            fake.mode = "ok"
            fake.hits.clear()
        self.server.CONTROLLER.resilience.cb_map.clear()

    def hits(self) -> dict[str, int]:
        return {name: len(fake.hits) for name, fake in self.fakes.items()}

    def fail(self, *names, mode="500"):
        for name in names or LOCAL:
            self.fakes[name].mode = mode

    def _headers(self, data_class, extra=None):
        return {
            **self.headers,
            **({"X-Data-Class": data_class} if data_class else {}),
            **(extra or {}),
        }

    def post(self, model, messages=None, data_class=None, headers=None, **kw):
        body = {"model": model, "messages": messages or TEXT, **kw}
        return self.client.post(
            "/v1/chat/completions", json=body, headers=self._headers(data_class, headers)
        )

    def compare(self, models, data_class=None, headers=None, **kw):
        body = {"models": models, "messages": kw.pop("messages", None) or TEXT, **kw}
        return self.client.post(
            "/v1/compare", json=body, headers=self._headers(data_class, headers)
        )

    def stream(self, model, messages=None, data_class=None, **kw):
        body = {"model": model, "messages": messages or TEXT, "stream": True, **kw}
        headers = self._headers(data_class)
        with self.client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
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
aux = cfg.pop("aux", None)
if aux:
    leaf = dict(cfg["endpoints"][aux])
    cfg["endpoints"]["embedding"] = {{
        "model": "openai/embed", "api_base": leaf["api_base"], "api_key_env": "FAKE_KEY",
        "tags": ["cloud", "jur:us"], "extra": {{"max_retries": 0}},
    }}
    cfg["tiers"]["private_emb"] = {{
        "router": "emb", "threshold": 0.5, "strong": "local_main", "weak": "local_small",
    }}
open({config!r}, "w").write(json.dumps(cfg))
import os
# governance (the policy) is honoured from the system layer only
os.environ["ROUTELLM_SYSTEM_CONFIG"] = {config!r}
sys.argv = ["x", "--routers", "random"]
from routellm.routers.routers import ROUTER_CLS
class _Pin:
    def route(self, prompt, threshold, pair):
        return pair.strong
ROUTER_CLS["pin_strong"] = lambda **_: _Pin()
from routellm.routers.base import Router
from routellm.routers.embeddings import embed
class _Emb(Router):
    requires_embedding = True
    def calculate_strong_win_rate(self, prompt):
        return 1.0 if embed(prompt) else 0.0
ROUTER_CLS["emb"] = _Emb
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
    world = pf.World(fakes, client, server, class_policy={policy!r})
    for name in {names!r}:
        world.reset()
        obs = pf.CASES[name].fn(world)
        out[name] = {{"obs": obs, "hits": world.hits()}}
print("@@RESULT@@" + json.dumps(out))
"""


def run_cases(tmp_path, names, extra_config=None, resilience=None, class_policy=False, route=False):
    """Run the named cases in one subprocess; return {case: {obs, hits}}.

    `extra_config` is merged over the fixture config (top-level keys),
    `resilience` sets `ResilienceConfig` fields before the first request.
    `class_policy` starts the app under a policy file and an auth key, with
    the extra tiers of `policy_tiers`; cases then declare a class.
    """
    if route:
        class_policy = True
        extra_config = {**route_config(), **(extra_config or {})}
    if class_policy:
        extra_config = {**class_policy_files(tmp_path, route), **(extra_config or {})}
    # Selector tiers consult the models.dev catalog on construction; an
    # empty snapshot keeps that off the network (tags need no catalog).
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    script = _SCRIPT.format(
        config=str(tmp_path / "config.json"),
        extra=json.dumps(extra_config or {}),
        settings=json.dumps({"r": resilience or {}}),
        names=list(names),
        policy=class_policy,
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
