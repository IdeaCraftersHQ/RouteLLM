"""Policy file: schema, load rules, deny by default, config wiring."""

import copy

import pytest

from routellm import policy as policy_mod
from routellm.decision import build_decision
from routellm.endpoints import EndpointRegistry
from routellm.policy import (
    PolicyError,
    PolicyRefusal,
    build_policy,
    check_decision_classes,
    load_policy,
    parse_policy,
)
from routellm.tests.decision_fakes import CLASSES

GOOD = {
    "default_class": None,
    "classes": {
        "open": {"treatment": "raw", "reach": ["local", "vpn", "cloud"], "tiers": ["default"]},
        "internal": {
            "treatment": "redact",
            "reach": ["local", "vpn"],
            "jur": ["ca"],
            "tiers": ["private"],
        },
        "secret": {"treatment": "block", "reach": ["local"], "tiers": ["private"]},
        "sealed": {"treatment": "vault"},
    },
    "principals": {"analyst": {"classes": ["open", "internal"]}},
}


def make(**over):
    raw = copy.deepcopy(GOOD)
    raw.update(over)
    return raw


def cls(name="x", **spec):
    return {"classes": {name: spec}}


# --- schema --------------------------------------------------------------


def test_good_policy_loads_with_defaults():
    pol = parse_policy(GOOD, "p.yaml")
    assert pol.names == ["open", "internal", "secret", "sealed"]
    assert pol.classes["open"].cache == "exact"
    assert pol.classes["secret"].cache == "none"
    assert pol.classes["internal"].cache == "none"
    assert pol.default_class is None


def test_explicit_cache_kept_on_block_class():
    raw = cls("s", treatment="block", reach=["local"], tiers=["t"], cache="exact")
    assert parse_policy(raw).classes["s"].cache == "exact"


@pytest.mark.parametrize(
    "raw, needle",
    [
        (cls("a", treatment="shred", reach=["local"], tiers=["t"]), r"class 'a' treatment"),
        (cls("a", treatment="raw", reach=["mars"], tiers=["t"]), r"class 'a' reach: 'mars'"),
        (
            cls("a", treatment="raw", reach=["local"], jur=["CA"], tiers=["t"]),
            r"class 'a' jur: 'CA'",
        ),
        (cls("a", treatment="raw", reach=["local"], jur=["usa"], tiers=["t"]), r"class 'a' jur"),
        (cls("a", treatment="raw", reach=["local"], tiers=["t"], cache="all"), r"class 'a' cache"),
        (cls("a", treatment="raw", reach=["local"], tiers=["t"], color="x"), r"class 'a' color"),
        (cls("a", treatment="raw", tiers=["t"]), r"class 'a' reach: list at least"),
        (cls("a", treatment="raw", reach=["local"]), r"class 'a' tiers: list at least"),
        ({"classes": {}}, r"classes: list at least"),
        ({"classes": {"a b": {"treatment": "vault"}}}, r"classes.a b"),
        (make(typo=1), r"typo"),
        (make(default_class="nope"), r"default_class: 'nope'"),
        (
            make(principals={"p": {"classes": ["nope"]}}),
            r"principals.p.classes: 'nope'",
        ),
        ({"classes": ["a"]}, r"classes"),
    ],
)
def test_invalid_policy_names_file_class_and_field(raw, needle):
    with pytest.raises(PolicyError, match=r"policy p\.yaml: .*" + needle):
        parse_policy(raw, "p.yaml")


def test_non_mapping_policy_rejected():
    with pytest.raises(PolicyError, match="must be a mapping"):
        parse_policy(["x"], "p.yaml")


# --- rules ---------------------------------------------------------------


def test_block_class_allowing_cloud_is_a_load_error():
    raw = cls("secret", treatment="block", reach=["local", "cloud"], tiers=["t"])
    with pytest.raises(PolicyError, match=r"class 'secret' reach: a block class never reaches"):
        parse_policy(raw, "p.yaml")


def test_redact_cloud_loads_because_the_redaction_treatment_exists():
    raw = cls("pii", treatment="redact", reach=["cloud"], tiers=["t"])
    assert parse_policy(raw).classes["pii"].treatment == "redact"


def test_redact_cloud_is_a_load_error_without_the_treatment(monkeypatch):
    monkeypatch.setattr(policy_mod, "REDACTION_AVAILABLE", False)
    raw = cls("pii", treatment="redact", reach=["cloud"], tiers=["t"])
    with pytest.raises(PolicyError, match=r"class 'pii' reach: 'cloud' on a redact"):
        parse_policy(raw, "p.yaml")


@pytest.mark.parametrize(
    "extra", [{"reach": ["local"]}, {"jur": ["ca"]}, {"tiers": ["t"]}, {"cache": "exact"}]
)
def test_vault_class_allows_nothing(extra):
    with pytest.raises(PolicyError, match=r"class 'v'"):
        parse_policy(cls("v", treatment="vault", **extra))


# --- deny by default -----------------------------------------------------


def test_unknown_class_refused():
    with pytest.raises(PolicyRefusal) as exc:
        parse_policy(GOOD).resolve("mystery")
    assert exc.value.code == "unknown_class"
    assert "mystery" in str(exc.value)


def test_missing_class_refused_when_default_is_null():
    for declared in (None, ""):
        with pytest.raises(PolicyRefusal) as exc:
            parse_policy(GOOD).resolve(declared)
        assert exc.value.code == "missing_class"


def test_missing_class_uses_default_class():
    name, spec = parse_policy(make(default_class="open")).resolve(None)
    assert name == "open" and spec.treatment == "raw"


def test_vault_class_refused_without_a_model_call():
    pol = parse_policy(GOOD)
    assert pol.classes["sealed"].refuses_all
    with pytest.raises(PolicyRefusal) as exc:
        pol.resolve("sealed")
    assert exc.value.code == "vault"


def test_known_class_resolves():
    name, spec = parse_policy(GOOD).resolve("secret")
    assert name == "secret" and spec.tiers == ["private"]


def test_principal_gets_only_listed_classes():
    pol = parse_policy(GOOD)
    assert pol.principal_allows("analyst", "open")
    assert not pol.principal_allows("analyst", "secret")
    assert not pol.principal_allows("stranger", "open")
    assert not pol.principal_allows(None, "open")


# --- file load -----------------------------------------------------------


def test_load_policy_reads_yaml(tmp_path):
    f = tmp_path / "policy.yaml"
    f.write_text(
        "classes:\n  open:\n    treatment: raw\n    reach: [cloud]\n    tiers: [default]\n"
    )
    pol = load_policy(f)
    assert pol.source == str(f) and pol.names == ["open"]


def test_load_policy_missing_file_names_it(tmp_path):
    with pytest.raises(PolicyError, match=r"policy .*nope\.yaml: cannot read"):
        load_policy(tmp_path / "nope.yaml")


def test_load_policy_bad_yaml_names_file(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("classes: [unclosed\n")
    with pytest.raises(PolicyError, match=r"p\.yaml: not valid YAML"):
        load_policy(f)


def test_load_error_names_file_class_and_field(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("classes:\n  secret:\n    treatment: block\n    reach: [cloud]\n    tiers: [t]\n")
    with pytest.raises(PolicyError, match=r"p\.yaml: class 'secret' reach"):
        load_policy(f)


# --- config wiring -------------------------------------------------------


def test_no_policy_key_means_no_policy_and_config_untouched():
    cfg = {"endpoints": {}}
    assert build_policy(cfg) is None
    assert "governance" not in cfg and "strict_registry" not in cfg


def test_policy_implies_governance_and_strict_registry(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("classes:\n  open:\n    treatment: raw\n    reach: [cloud]\n    tiers: [d]\n")
    cfg = {"policy": str(f), "governance": False, "strict_registry": False}
    assert build_policy(cfg).names == ["open"]
    assert cfg["governance"] is True and cfg["strict_registry"] is True
    reg = EndpointRegistry.from_config(
        {**cfg, "endpoints": {"a": {"model": "m", "tags": ["cloud", "jur:ca"]}}}
    )
    assert reg.governance and reg.strict_registry


def test_relative_policy_path_resolves_against_setting_file(tmp_path):
    (tmp_path / "p.yaml").write_text(
        "classes:\n  open:\n    treatment: raw\n    reach: [cloud]\n    tiers: [d]\n"
    )
    seen = []

    def resolve(key, value):
        seen.append((key, value))
        return tmp_path / value

    assert build_policy({"policy": "p.yaml"}, resolve).names == ["open"]
    assert seen == [("policy", "p.yaml")]


@pytest.mark.parametrize("bad", [True, 3, ""])
def test_policy_key_must_be_a_path(bad):
    with pytest.raises(PolicyError, match="`policy:`"):
        build_policy({"policy": bad})


# --- agreement with the decision classifier -------------------------------


def _decision(classes):
    reg = EndpointRegistry.from_config(
        {
            "endpoints": {
                "decider": {
                    "model": "ollama_chat/tiny",
                    "api_base": "http://127.0.0.1:11500",
                    "tags": ["local"],
                }
            }
        }
    )
    return build_decision(
        {"decision": {"endpoint": "decider", "data_classes": classes, "intents": {"chat": "x"}}},
        reg,
    )


def _policy_for(names):
    return parse_policy(
        {"classes": {n: {"treatment": "raw", "reach": ["local"], "tiers": ["t"]} for n in names}},
        "p.yaml",
    )


def test_decision_classes_must_all_be_in_policy():
    decision = _decision(CLASSES)
    check_decision_classes(_policy_for([*CLASSES, "extra"]), decision.config.data_classes)
    with pytest.raises(PolicyError, match=r"decision\.data_classes 'private'"):
        check_decision_classes(_policy_for(["general", "internal"]), decision.config.data_classes)


def test_policy_key_is_not_handed_to_routers(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["routellm.openai_server"])
    from routellm.openai_server import build_router_config

    assert build_router_config({"policy": "p.yaml", "endpoints": {}}) is None
