"""The backward-compatibility contract for capability routing, pinned.

Every case here runs against a registry whose endpoints carry NO
`capabilities:` block, which is what every config in the wild looks
like today. The contract is that such a config loads, routes, and
produces the same decision path it always did, and that a request which
carries no images, no tools and no `response_format` still pays for
exactly one router call per level.

These are the tests every later change to the capability code is
measured against, which is why they live in their own file rather than
as a clause in a docstring.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import routellm.controller
from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.pairing import Selector, rank_candidates
from routellm.requirements import Requirements
from routellm.routers.base import Router

# Reuse the example's fixtures rather than copying its fake catalog.
from routellm.tests.test_examples import (  # noqa: F401
    EXAMPLE,
    catalog,
    config,
    resolved,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The exact key set a plain request's path entry carries. This is the
#: additive-path guarantee: it goes red the moment someone writes a
#: capability key unconditionally.
PLAIN_PATH_KEYS = {
    "tier",
    "router",
    "router_from",
    "threshold",
    "threshold_from",
    "win_rate",
    "picked",
}

#: A registry that knows nothing about any endpoint's capabilities.
BARE_CONFIG = {
    "endpoints": {
        "cloud_strong": {"model": "gpt-4o", "quality": 90},
        "frontier_local": {"model": "ollama_chat/qwen3:32b", "tags": ["local"]},
        "local_fast": {"model": "ollama_chat/qwen3:8b", "tags": ["local"]},
    },
    "tiers": {
        "premium": {
            "router": "hi",
            "threshold": 0.33,
            "strong": "cloud_strong",
            "weak": "frontier_local",
        },
        "default": {
            "router": "hi",
            "threshold": 0.12,
            "strong": "premium",
            "weak": "local_fast",
        },
    },
}


class _StubRouter(Router):
    """Router returning a fixed win rate and counting its calls."""

    win_rate = 0.9
    calls: list[str] = []

    def calculate_strong_win_rate(self, prompt):
        type(self).calls.append(prompt)
        return type(self).win_rate


@pytest.fixture
def hi_router(monkeypatch):
    """Stub router scoring 0.9, registered under the name 'hi'."""

    class Stub(_StubRouter):
        pass

    Stub.calls = []
    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "hi", lambda **kw: Stub())
    return Stub


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """Keep every test off the network and out of the real cache."""
    import routellm.pairing as pairing

    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: [])


def _controller():
    """A controller on the capability-free registry, caching off."""
    return Controller(
        routers=["hi"],
        strong_model=None,
        weak_model=None,
        endpoints=EndpointRegistry.from_config(BARE_CONFIG),
        cache_config=CacheConfig(enabled=False),
    )


def _text(content="explain recursion"):
    """One plain user message."""
    return [{"role": "user", "content": content}]


def _vision():
    """One user message carrying an image part."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ],
        }
    ]


def _route(controller, messages, reqs):
    """Run the routing half of one request with the given requirements."""
    request = {"model": "default", "messages": messages}
    tier, router, threshold = controller._parse_model_name("default")
    from routellm import requirements as requirements_module

    prompt = requirements_module._prompt_text(messages)
    return controller._route(prompt, request, tier, router, threshold, reqs)


# ---------------------------------------------------------------------------
# The no-op path
# ---------------------------------------------------------------------------


def test_config_without_capabilities_loads_and_routes(hi_router):
    controller = _controller()

    picked, path, _ = _route(controller, _text(), Requirements())

    assert picked == "cloud_strong"
    assert len(path) == 2


def test_plain_request_takes_the_same_leaf_as_before(hi_router):
    from routellm import requirements as requirements_module

    controller = _controller()
    messages = _text()
    derived = requirements_module.derive(messages, {}, "default")

    with_check, path_with, _ = _route(controller, messages, derived)
    hi_router.calls = []
    without_check, path_without, _ = _route(controller, messages, Requirements())

    assert with_check == without_check
    assert path_with == path_without


def test_path_keys_are_unchanged_for_a_plain_request(hi_router):
    from routellm import requirements as requirements_module

    controller = _controller()
    messages = _text()
    derived = requirements_module.derive(messages, {}, "default")

    _, path, _ = _route(controller, messages, derived)

    for entry in path:
        assert set(entry) == PLAIN_PATH_KEYS


def test_router_is_still_called_once_per_level_for_a_plain_request(hi_router):
    from routellm import requirements as requirements_module

    controller = _controller()
    messages = _text()
    derived = requirements_module.derive(messages, {}, "default")

    _, path, _ = _route(controller, messages, derived)

    assert len(hi_router.calls) == len(path) == 2


def test_no_capability_block_anywhere_means_no_refusal_ever(hi_router):
    from routellm import requirements as requirements_module

    controller = _controller()
    messages = _vision()
    derived = requirements_module.derive(messages, {}, "default")

    assert derived.vision is True

    picked, path, _ = _route(controller, messages, derived)

    # Unknown serves by default, so the routers still decide.
    assert picked == "cloud_strong"
    for entry in path:
        assert "capability_forced" not in entry


def test_selectors_without_capability_terms_pick_what_they_picked(catalog):
    registry = EndpointRegistry.from_config(BARE_CONFIG)

    ranked = [
        name for name, _ in rank_candidates(registry, Selector(select="tag:local"))
    ]

    assert set(ranked) == {"frontier_local", "local_fast"}


def test_examples_multitier_still_resolves_every_selector(resolved):
    for tier_name in resolved.tier_names():
        tier = resolved.get_tier(tier_name)
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            assert isinstance(value, str)
            assert resolved.has_tier(value) or value in resolved.names()


@pytest.mark.slow
def test_token_counting_imports_no_torch(tmp_path):
    # In a subprocess so the heavy stack a sibling test imported cannot
    # make this pass or fail spuriously.
    script = textwrap.dedent(
        """
        import sys
        from routellm.requirements import derive

        derive(
            [{"role": "user", "content": "hello there"}],
            {"max_tokens": 16},
            "default",
        )
        assert "torch" not in sys.modules, sorted(
            n for n in sys.modules if "torch" in n
        )
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
