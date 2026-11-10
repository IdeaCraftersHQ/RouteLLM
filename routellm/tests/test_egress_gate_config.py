"""Egress gate at load: the config shape and the model-id checks.

A gate selects endpoints by tag and holds each one's model id, and every
id it can send under `also_check`, to an allow and a deny list. A
violation stops startup naming the endpoint and the id. No network.
"""

import pytest

from routellm.controller import Controller
from routellm.egress_gate import EgressGateError, build_egress_gates
from routellm.endpoints import EndpointRegistry
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy import parse_policy
from routellm.tests.test_class_tiers import PAIR, endpoint


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))


POLICY = parse_policy(
    {
        "classes": {
            "general": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["default"]},
            "internal": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["default"]},
        }
    },
    "p.yaml",
)


def gated(model="openai/vendor-a", extra=None, tags=("cloud", "gated")):
    spec = endpoint(list(tags))
    spec["model"] = model
    if extra is not None:
        spec["extra"] = extra
    return spec


def registry(**eps):
    eps = {"loc": endpoint(["local"]), **eps}
    return EndpointRegistry.from_config(
        {
            "endpoints": eps,
            "tiers": {"default": {**PAIR, "strong": "loc", "weak": next(iter(eps))}},
            "strict_registry": True,
        }
    )


def gate(**over):
    return {"name": "gated", "select": "tag:gated", **over}


def load(gates, reg, policy=POLICY, **kw):
    built = build_egress_gates({"egress_gates": gates}, reg)
    built.check(reg, policy, **kw)
    return built


def refused(gates, reg, **kw) -> str:
    with pytest.raises(EgressGateError) as exc:
        load(gates, reg, **kw)
    return str(exc.value)


def test_no_section_builds_nothing():
    assert build_egress_gates({}, registry()) is None
    assert build_egress_gates(None, registry()) is None


def test_an_id_on_the_allow_list_loads():
    load([gate(models={"allow": ["openai/vendor-*"]})], registry(g=gated()))


def test_an_id_off_the_allow_list_is_a_load_error_naming_endpoint_and_id():
    msg = refused([gate(models={"allow": ["openai/other-*"]})], registry(g=gated()))
    assert "gate 'gated'" in msg and "endpoint 'g'" in msg and "'openai/vendor-a'" in msg


def test_an_id_on_the_deny_list_is_a_load_error():
    msg = refused([gate(models={"deny": ["*:free"]})], registry(g=gated("openai/vendor-a:free")))
    assert "endpoint 'g'" in msg and "'openai/vendor-a:free'" in msg and "deny" in msg


def test_deny_wins_over_allow():
    models = {"allow": ["openai/vendor-*"], "deny": ["*:free"]}
    msg = refused([gate(models=models)], registry(g=gated("openai/vendor-a:free")))
    assert "deny" in msg


def test_an_empty_allow_list_allows_nothing():
    msg = refused([gate(models={"allow": []})], registry(g=gated()))
    assert "endpoint 'g'" in msg and "allow" in msg


def test_no_models_section_puts_no_limit_on_ids():
    load([gate(deny_classes=["internal"])], registry(g=gated("openai/anything")))


def test_an_endpoint_the_selector_does_not_match_is_not_checked():
    load(
        [gate(models={"allow": ["openai/vendor-*"]})],
        registry(g=gated(), other=gated("openai/stranger", tags=("cloud",))),
    )


def test_a_denied_id_under_also_check_is_a_load_error():
    extra = {"extra_body": {"models": ["openai/vendor-b", "openai/evil-x"]}}
    msg = refused(
        [gate(models={"allow": ["openai/vendor-*"]}, also_check=["extra_body.models"])],
        registry(g=gated(extra=extra)),
    )
    assert "endpoint 'g'" in msg and "'openai/evil-x'" in msg and "extra_body.models" in msg


def test_a_scalar_under_also_check_is_checked_too():
    extra = {"extra_body": {"route": "openai/evil-x"}}
    msg = refused(
        [gate(models={"allow": ["openai/vendor-*"]}, also_check=["extra_body.route"])],
        registry(g=gated(extra=extra)),
    )
    assert "'openai/evil-x'" in msg


def test_a_path_the_endpoint_does_not_set_is_skipped():
    load(
        [gate(models={"allow": ["openai/vendor-*"]}, also_check=["extra_body.models"])],
        registry(g=gated()),
    )


def test_a_value_under_also_check_that_is_not_ids_is_an_error():
    extra = {"extra_body": {"models": {"a": 1}}}
    msg = refused([gate(also_check=["extra_body.models"])], registry(g=gated(extra=extra)))
    assert "extra_body.models" in msg


def test_ids_under_also_check_are_checked_without_a_list_too():
    # No allow list: only deny applies, and it applies to every id.
    extra = {"extra_body": {"models": ["openai/vendor-b:free"]}}
    msg = refused(
        [gate(models={"deny": ["*:free"]}, also_check=["extra_body.models"])],
        registry(g=gated(extra=extra)),
    )
    assert "'openai/vendor-b:free'" in msg


def test_a_required_param_per_allow_entry_is_verified():
    models = {
        "allow": [
            {"model": "openai/vendor-*", "require": {"extra_body.provider.data_collection": "deny"}}
        ]
    }
    ok = {"extra_body": {"provider": {"data_collection": "deny"}}}
    load([gate(models=models)], registry(g=gated(extra=ok)))
    bad = {"extra_body": {"provider": {"data_collection": "allow"}}}
    msg = refused([gate(models=models)], registry(g=gated(extra=bad)))
    assert "endpoint 'g'" in msg and "extra_body.provider.data_collection" in msg
    msg = refused([gate(models=models)], registry(g=gated()))
    assert "extra_body.provider.data_collection" in msg


def test_a_require_applies_only_to_the_entry_the_id_matched():
    models = {
        "allow": [
            {"model": "openai/vendor-a", "require": {"extra_body.x": "1"}},
            "openai/vendor-b",
        ]
    }
    load([gate(models=models)], registry(g=gated("openai/vendor-b")))


def test_select_takes_negation_and_alternatives():
    models = {"allow": ["openai/vendor-*"]}
    reg = registry(
        g=gated(),
        h=gated("openai/stranger", tags=("cloud", "free")),
        v=gated("openai/stranger", tags=("vpn",)),
    )
    load([gate(select="tag:cloud !tag:free", models=models)], reg)
    msg = refused([gate(select="tag:cloud|vpn", models=models)], reg)
    assert "endpoint 'h'" in msg and "endpoint 'v'" in msg


def test_a_select_that_matches_nothing_is_a_load_error_not_an_open_gate():
    msg = refused([gate(select="tag:typo")], registry(g=gated()))
    assert "matches no endpoint" in msg


@pytest.mark.parametrize(
    "bad",
    [
        {"select": "tag:nope:x"},  # unknown namespace
        {"select": "quality:80"},  # not a tag term
        {"select": ""},
        {"deny_classes": ["ghost"]},  # not a policy class
        {"nonsense": 1},  # unknown key
        {"name": "bad name"},
    ],
)
def test_a_malformed_gate_is_refused(bad):
    with pytest.raises(EgressGateError):
        load([gate(**bad)], registry(g=gated()))


def test_gate_names_are_unique():
    msg = refused([gate(), gate()], registry(g=gated()))
    assert "'gated'" in msg and "twice" in msg


def test_the_section_must_be_a_list_of_mappings():
    with pytest.raises(EgressGateError):
        build_egress_gates({"egress_gates": {"name": "x"}}, registry())
    with pytest.raises(EgressGateError):
        build_egress_gates({"egress_gates": ["x"]}, registry())


def test_deny_classes_need_a_policy():
    msg = refused([gate(deny_classes=["internal"])], registry(g=gated()), policy=None)
    assert "policy" in msg


def test_the_decision_endpoint_may_not_be_gated():
    # The classifier reads the request before any class is known.
    msg = refused([gate(deny_classes=["internal"])], registry(g=gated()), decision_endpoint="g")
    assert "decision" in msg and "'g'" in msg


def test_a_gated_decision_endpoint_is_fine_for_a_gate_with_no_class_condition():
    load([gate(models={"allow": ["openai/vendor-*"]})], registry(g=gated()), decision_endpoint="g")


def test_the_controller_runs_the_checks_at_construction():
    reg = registry(g=gated())
    built = build_egress_gates({"egress_gates": [gate(models={"allow": ["x/*"]})]}, reg)
    with pytest.raises(EgressGateError):
        Controller(
            routers=["random"],
            endpoints=reg,
            policy=POLICY,
            strict_registry=True,
            egress_gates=built,
        )
