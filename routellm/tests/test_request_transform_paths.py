"""Request transforms beyond the main call: shadow, compare, the loader, the real app.

Shares the test transforms and the patched provider of `test_request_transform`.
"""

import asyncio
import json
from pathlib import Path

import pytest

from routellm.caching import CacheConfig
from routellm.config import ConfigError
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.quality import FineTuneConfig, QualityManager
from routellm.server_config import build_comparison_config, build_transforms
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.test_request_transform import (  # noqa: F401  (ROUTER_CLS side effect)
    TOKEN,
    Always,
    Boom,
    Swap,
    _hold,
    messages,
)
from routellm.transform import (
    HoldbackRestorer,
    IdentityTransform,
    TransformRunner,
    load_transforms,
)

# --- shadow and compare ------------------------------------------------------------


@pytest.fixture
def ups():
    made = {name: FakeUpstream(name) for name in "abc"}
    yield made
    for fake in made.values():
        fake.server.shutdown()
        fake.server.server_close()


def shadow_controller(ups, tmp_path, transforms):
    endpoints = {
        name: {
            "model": f"openai/model-{name}",
            "api_base": ups[name].base,
            "api_key_env": "FAKE_KEY",
        }
        for name in ups
    }
    controller = Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(
            {
                "endpoints": endpoints,
                "tiers": {
                    "default": {
                        "router": "pin_strong",
                        "threshold": 0.0,
                        "strong": "a",
                        "weak": "b",
                    }
                },
            }
        ),
        cache_config=CacheConfig(enabled=False),
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(
                enabled=True, trace_dir=str(tmp_path / "traces"), capture_content=True
            )
        ),
        comparison_config=build_comparison_config(
            {"comparison": {"enabled": True, "allow_header": True}}
        ),
        transforms=transforms,
    )
    controller.resilience.config.max_retries = 0
    return controller


def bodies(fake):
    return json.dumps([h[2]["messages"] for h in fake.hits])


def test_a_shadow_call_is_transformed_and_its_trace_holds_that_form(ups, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    controller = shadow_controller(ups, tmp_path, [Swap()])

    async def go():
        await controller.acompletion(model="default", messages=messages(), compare="c")
        await controller.comparison.drain()

    asyncio.run(go())
    assert ups["c"].hits, "the shadow never ran"
    assert "SECRET" not in bodies(ups["c"]) and TOKEN in bodies(ups["c"])
    assert "SECRET" not in bodies(ups["a"])
    stored = "".join(p.read_text() for p in Path(tmp_path / "traces").glob("*.json"))
    assert "SECRET" not in stored and TOKEN in stored


def test_a_raising_transform_means_no_shadow_call_either(ups, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    controller = shadow_controller(ups, tmp_path, [Boom()])
    result = asyncio.run(
        controller.comparison.compare(["b", "c"], {"messages": messages()}, operator=True)
    )
    assert not ups["b"].hits and not ups["c"].hits
    assert {r["error"] for r in result["results"]} == {"transform_failed"}


def test_a_compare_candidate_is_transformed_and_its_output_restored(ups, tmp_path, monkeypatch):
    class Shout(Swap):
        def inverse(self, text, state):
            return text.replace("hello", "HELLO")

    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    controller = shadow_controller(ups, tmp_path, [Shout()])
    result = asyncio.run(
        controller.comparison.compare(["b", "c"], {"messages": messages()}, operator=True)
    )
    assert TOKEN in bodies(ups["b"]) and "SECRET" not in bodies(ups["b"])
    assert [r["output"] for r in result["results"]] == ["HELLO from b", "HELLO from c"]
    stored = "".join(p.read_text() for p in Path(tmp_path / "traces").glob("*.json"))
    assert "hello from b" in stored and "HELLO" not in stored and "SECRET" not in stored


# --- protocol pieces ---------------------------------------------------------------


def test_holdback_restorer_buffers_only_what_may_become_a_token():
    r = HoldbackRestorer(lambda t: t.replace(TOKEN, "X"), _hold)
    out = [r.feed(p) for p in ("ab[", "[S_", "1]", "]cd[[")]
    assert "".join(out) + r.finish() == "abXcd[["
    assert out[0] == "ab"


def test_runner_inactive_without_transforms_and_skips_identity():
    assert not TransformRunner().active
    assert not TransformRunner([IdentityTransform()]).active


class Named:
    def __init__(self, tag="t"):
        self.tag = tag

    def forward(self, msgs, ctx):
        return msgs, None

    def inverse(self, text, state):
        return text


def test_loader_builds_from_import_path_and_options():
    built = load_transforms(
        [f"{__name__}:Named", {"use": f"{__name__}:Named", "options": {"tag": "z"}}]
    )
    assert [type(b) for b in built] == [Named, Named]
    assert built[1].tag == "z"


def test_loader_resolves_an_entry_point_name(monkeypatch):
    class EP:
        name = "mine"

        def load(self):
            return Named

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group=None: [EP()] if group == "routellm.transforms" else [],
    )
    assert isinstance(load_transforms(["mine"])[0], Named)
    assert isinstance(load_transforms(["identity"])[0], IdentityTransform)


@pytest.mark.parametrize(
    "spec",
    [
        "not-a-list",
        [{"options": {}}],
        [{"use": "nope.nowhere:X"}],
        [{"use": "ghost-name"}],
        [{"use": f"{__name__}:Named", "extra": 1}],
        [{"use": f"{__name__}:TOKEN"}],
    ],
)
def test_loader_refuses_bad_entries(spec):
    with pytest.raises(ConfigError):
        load_transforms(spec)


def test_server_config_reads_the_transforms_section():
    assert build_transforms(None) == []
    assert len(build_transforms({"transforms": [f"{__name__}:Named"]})) == 1


# --- through the real app ---------------------------------------------------------


@pytest.mark.slow
def test_a_raising_transform_is_500_with_its_reason_and_no_upstream_call(tmp_path):
    from routellm.tests.test_request_class import _run

    out = _run(
        tmp_path,
        """
body = {**BODY, "messages": [{"role": "user", "content": "tell SECRET"}]}
r = client.post("/v1/chat/completions", json=body)
s = client.post("/v1/chat/completions", json={**body, "stream": True})
emit({"r": [r.status_code, r.json()], "s": s.status_code, "hits": hits()})
""",
        policy=False,
        auth=False,
        config_extra={"transforms": ["routellm.tests.test_request_transform:Boom"]},
    )
    status, body = out["r"]
    assert status == 500 and out["s"] == 500
    assert body["code"] == "transform_failed" and body["object"] == "error"
    assert "Boom failed" in body["message"] and "SECRET" not in json.dumps(body)
    assert out["hits"] == 0


@pytest.mark.slow
def test_the_app_sends_the_rewrite_upstream_from_the_configured_list(tmp_path):
    from routellm.tests.test_request_class import _run

    out = _run(
        tmp_path,
        """
body = {**BODY, "messages": [{"role": "user", "content": "tell SECRET"}]}
r = client.post("/v1/chat/completions", json=body)
sent = json.dumps([h[2]["messages"] for h in A.hits + B.hits])
emit({"status": r.status_code, "sent": sent})
""",
        policy=False,
        auth=False,
        config_extra={"transforms": [{"use": "routellm.tests.test_request_transform:Always"}]},
    )
    assert out["status"] == 200
    assert TOKEN in out["sent"] and "SECRET" not in out["sent"]
