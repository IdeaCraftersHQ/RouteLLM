"""Areas: naming them in config and pairing on per-area quality.

An area is a named group of tiers. Pairing resolves once, at startup,
so a selector knows only the tier it sits in; the tier's name is
therefore the only handle on its area, and the mapping is explicit in
config rather than inferred.
"""

import pytest

from routellm.endpoints import Endpoint, EndpointRegistry, Selector, Tier
from routellm.pairing import describe_candidate, rank_candidates, resolve_pairing


def _config(areas=None, tiers=None):
    config = {
        "endpoints": {
            "alpha": {"model": "m/alpha", "quality": 90, "tags": ["chat"]},
            "beta": {"model": "m/beta", "quality": 10, "tags": ["chat"]},
        },
        "tiers": tiers
        if tiers is not None
        else {
            "coding": {"strong": "alpha", "weak": "beta"},
            "coding_quality": {"strong": "alpha", "weak": "beta"},
        },
    }
    if areas is not None:
        config["areas"] = areas
    return config


def test_areas_invert_into_tier_to_area():
    registry = EndpointRegistry.from_config(_config(areas={"coding": ["coding", "coding_quality"]}))

    assert registry.areas == {"coding": "coding", "coding_quality": "coding"}
    assert registry.area_of("coding_quality") == "coding"
    assert registry.area_of("nothing") is None
    assert registry.area_of(None) is None


def test_a_tier_in_two_areas_names_both():
    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(
            _config(
                areas={
                    "coding": ["coding", "coding_quality"],
                    "writing": ["coding_quality"],
                }
            )
        )

    message = str(excinfo.value)
    assert "coding_quality" in message
    assert "coding" in message and "writing" in message


def test_an_area_naming_a_missing_tier_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(areas_config := _config(areas={"coding": ["ghost"]}))

    message = str(excinfo.value)
    assert "ghost" in message
    assert "coding" in message


def _area_registry():
    """A registry where alpha wins overall and beta wins inside coding."""
    registry = EndpointRegistry(
        endpoints={
            "alpha": Endpoint(name="alpha", model="m/alpha", quality=90, tags=["chat"]),
            "beta": Endpoint(name="beta", model="m/beta", quality=10, tags=["chat"]),
        },
        tiers={
            "coding_quality": Tier(name="coding_quality", strong="alpha", weak="beta"),
            "flat": Tier(name="flat", strong="alpha", weak="beta"),
        },
    )
    registry.areas = {"coding_quality": "coding"}
    registry.area_quality = {
        "alpha": {"coding": 20},
        "beta": {"coding": 95},
    }
    return registry


def test_selector_in_an_area_tier_orders_on_the_area_quality():
    registry = _area_registry()
    selector = Selector(select="tag:chat", order="quality_desc")

    assert resolve_pairing(registry, selector) == "alpha"
    assert resolve_pairing(registry, selector, area="coding") == "beta"


def test_an_endpoint_without_an_area_number_falls_back_to_overall():
    registry = _area_registry()
    # beta has a coding number, alpha does not.
    registry.area_quality = {"beta": {"coding": 95}}
    selector = Selector(select="tag:chat", order="quality_desc")

    # 95 beats alpha's overall 90, so beta still wins; alpha is ranked
    # on its overall number rather than dropped to the unrated bucket.
    ranked = rank_candidates(registry, selector, area="coding")
    assert [name for name, _ in ranked] == ["beta", "alpha"]
    assert dict(ranked)["alpha"].effective_quality == 90
    assert dict(ranked)["beta"].effective_quality == 95


def test_a_tier_with_no_area_orders_on_overall_quality():
    registry = _area_registry()
    selector = Selector(select="tag:chat", order="quality_desc")

    assert resolve_pairing(registry, selector, area=None) == "alpha"


def test_the_explain_output_names_the_area_and_the_source():
    registry = _area_registry()
    selector = Selector(select="tag:chat", order="quality_desc")

    ranked = rank_candidates(registry, selector, area="coding")
    described = [describe_candidate(candidate) for _, candidate in ranked]

    assert any("quality=95 [coding]" in line for line in described)
    overall = rank_candidates(registry, selector)
    assert any(
        "quality=90" in line and "[" not in line
        for line in (describe_candidate(c) for _, c in overall)
    )


def test_registry_pairings_pass_each_tier_its_own_area():
    registry = EndpointRegistry(
        endpoints={
            "alpha": Endpoint(name="alpha", model="m/alpha", quality=90, tags=["chat"]),
            "beta": Endpoint(name="beta", model="m/beta", quality=10, tags=["chat"]),
        },
        tiers={
            "coding_quality": Tier(
                name="coding_quality",
                strong=Selector(select="tag:chat", order="quality_desc"),
                weak=Selector(select="tag:chat", order="quality_asc"),
            ),
            "flat": Tier(
                name="flat",
                strong=Selector(select="tag:chat", order="quality_desc"),
                weak=Selector(select="tag:chat", order="quality_asc"),
            ),
        },
    )
    registry.areas = {"coding_quality": "coding"}
    registry.area_quality = {"alpha": {"coding": 20}, "beta": {"coding": 95}}

    from routellm.pairing import resolve_registry_pairings

    resolve_registry_pairings(registry)

    # Inside the area the measured order wins; outside it the overall
    # one does.
    assert registry.get_tier("coding_quality").strong == "beta"
    assert registry.get_tier("flat").strong == "alpha"


def test_the_recorded_trace_carries_the_area(tmp_path):
    from routellm.quality import FineTuneConfig, QualityManager

    manager = QualityManager(
        fine_tune_config=FineTuneConfig(enabled=True, trace_dir=str(tmp_path / "traces"))
    )
    manager.record_trace(
        "hello",
        "alpha",
        {"choices": [{"message": {"content": "hi"}}]},
        path=[{"tier": "coding_quality", "router": "jev", "win_rate": 0.7}],
        area="coding",
    )

    import json
    import os

    directory = manager.fine_tune_config.trace_dir
    name = [n for n in os.listdir(directory) if n.endswith(".json")][0]
    with open(os.path.join(directory, name)) as handle:
        trace = json.load(handle)

    assert trace["routellm"]["area"] == "coding"
    assert trace["routellm"]["tier"] == "coding_quality"


def test_the_controller_fills_the_area_from_the_registry(tmp_path):
    from unittest.mock import MagicMock, patch

    from litellm.utils import ModelResponse

    from routellm.controller import Controller
    from routellm.quality import FineTuneConfig, QualityManager

    response = ModelResponse(
        **{
            "id": "x",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "created": 0,
            "model": "m",
            "object": "chat.completion",
        }
    )

    registry = EndpointRegistry(
        endpoints={
            "alpha": Endpoint(name="alpha", model="m/alpha"),
            "beta": Endpoint(name="beta", model="m/beta"),
        },
        tiers={"coding_quality": Tier(name="coding_quality", strong="alpha", weak="beta")},
    )
    registry.areas = {"coding_quality": "coding"}

    with patch("routellm.controller.completion", MagicMock(return_value=response)):
        controller = Controller(
            routers=["random"],
            endpoints=registry,
            strong_model="alpha",
            weak_model="beta",
            progress_bar=False,
        )
        controller.quality_manager = QualityManager(
            fine_tune_config=FineTuneConfig(enabled=True, trace_dir=str(tmp_path / "traces"))
        )
        controller.completion(
            model="coding_quality",
            router="random",
            threshold=0.5,
            messages=[{"role": "user", "content": "hello"}],
        )

    import json
    import os

    directory = controller.quality_manager.fine_tune_config.trace_dir
    name = [n for n in os.listdir(directory) if n.endswith(".json")][0]
    with open(os.path.join(directory, name)) as handle:
        trace = json.load(handle)

    assert trace["routellm"]["area"] == "coding"
