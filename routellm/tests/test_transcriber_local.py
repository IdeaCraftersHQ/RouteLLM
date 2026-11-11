"""A transcriber runs before a request's class is known: redact and block classes need it local."""

import pytest

from routellm.class_tiers import check_transcriber
from routellm.endpoints import EndpointRegistry
from routellm.policy import PolicyError, parse_policy


def registry(transcriber=None, tier_transcriber=None):
    tier = {"router": "random", "threshold": 0.5, "strong": "a", "weak": "a"}
    if tier_transcriber is not None:
        tier["transcriber"] = tier_transcriber
    cfg = {
        "endpoints": {
            "a": {
                "model": "openai/a",
                "api_base": "http://127.0.0.1:1/v1",
                "api_key_env": "K",
                "tags": ["cloud"],
            }
        },
        "tiers": {"default": tier},
    }
    if transcriber is not None:
        cfg["transcriber"] = transcriber
    return EndpointRegistry.from_config(cfg)


def policy(treatment):
    spec = {"treatment": treatment, "reach": ["local"], "tiers": ["default"]}
    if treatment == "redact":
        spec["reach"] = ["local", "cloud"]
    if treatment == "vault":
        spec = {"treatment": "vault"}
    return parse_policy({"classes": {"c": spec}, "default_class": "c"}, "p")


CLOUD = {"mode": "transcribe", "api_base": "https://stt.example.com/v1", "model": "w"}
LOCAL = {**CLOUD, "api_base": "http://127.0.0.1:9/v1", "tags": ["local"]}


@pytest.mark.parametrize("treatment", ["redact", "block"])
def test_a_non_local_transcriber_is_refused_when_a_class_redacts_or_blocks(treatment):
    with pytest.raises(PolicyError, match=r"transcriber.*\bc\b|\bc\b.*transcriber"):
        check_transcriber(policy(treatment), registry(CLOUD))


def test_the_message_names_the_transcriber_scope():
    with pytest.raises(PolicyError, match="'global'"):
        check_transcriber(policy("redact"), registry(CLOUD))
    with pytest.raises(PolicyError, match="'default'"):
        check_transcriber(policy("redact"), registry(None, CLOUD))


@pytest.mark.parametrize("treatment", ["redact", "block"])
def test_a_local_transcriber_passes(treatment):
    check_transcriber(policy(treatment), registry(LOCAL))


def test_a_passthrough_or_absent_transcriber_passes():
    check_transcriber(policy("redact"), registry())
    check_transcriber(policy("redact"), registry({"mode": "passthrough"}))


def test_a_policy_with_only_raw_or_vault_classes_does_not_care():
    check_transcriber(policy("raw"), registry(CLOUD))
    check_transcriber(policy("vault"), registry(CLOUD))


def test_the_controller_refuses_to_start_with_a_non_local_transcriber(tmp_path):
    from routellm.controller import Controller
    from routellm.redact.placeholders import Redactor
    from routellm.redact.tokens import TokenStore
    from routellm.redact.transform import RedactTransform

    redactor = RedactTransform(
        Redactor(TokenStore(str(tmp_path / "t.db")), analyze=lambda t, lang: [])
    )

    def start(transcriber):
        return Controller(
            routers=["random"],
            endpoints=registry(transcriber),
            policy=policy("redact"),
            transforms=[redactor],
        )

    with pytest.raises(PolicyError, match="transcriber"):
        start(CLOUD)
    start(LOCAL)
