"""Declarable classes: a caller may declare them, the model may never choose them."""

import asyncio
from types import SimpleNamespace

import pytest

from routellm.decision import Decision
from routellm.decision.llm import LLMProvider, answer_schema, build_prompt
from routellm.policy import PolicyError, parse_policy
from routellm.policy_routing import PolicyRoutingError, build_policy_routing
from routellm.tests.decision_fakes import FakeTransport, answer, make_classifier, make_config
from routellm.tests.policy_routing_world import A, class_policy, routing_section
from routellm.tests.test_policy_routing_config import registry

CLASSES = {
    "general": "nothing sensitive",
    "trainable": "owner accepts training use",
    "internal": "company-internal material",
    "private": "personal or secret data",
}
MSG = [{"role": "user", "content": "hello there"}]


def run(coro):
    return asyncio.run(coro)


def clf(reply=None, **over):
    return make_classifier(
        FakeTransport(reply), data_classes=CLASSES, declarable=["trainable"], **over
    )


# -- what the model is offered and may answer ------------------------------------------


def test_enum_and_prompt_exclude_a_declarable_class():
    cfg = make_config(data_classes=CLASSES, declarable=["trainable"])
    enum = answer_schema(cfg)["properties"]["data_class"]["enum"]
    assert enum == ["general", "internal", "private"]
    system = build_prompt(cfg, "x")[0]["content"]
    assert "trainable" not in system and "training use" not in system
    assert "- internal:" in system


def test_without_declarable_every_class_is_offered():
    cfg = make_config(data_classes=CLASSES)
    assert "trainable" in answer_schema(cfg)["properties"]["data_class"]["enum"]


def test_parse_rejects_a_declarable_label():
    provider = LLMProvider(make_config(data_classes=CLASSES, declarable=["trainable"]), None)
    from routellm.decision import DecisionError

    with pytest.raises(DecisionError) as info:
        provider.parse(answer("trainable"))
    assert info.value.kind == "invalid_answer"


def test_model_naming_a_declarable_class_takes_the_error_path():
    c, _ = clf(answer("trainable"))
    d = run(c.classify(MSG))
    assert (d.source, d.error, d.data_class) == ("error", "invalid_answer", "private")


def test_model_naming_it_on_a_declared_request_is_still_an_error():
    c, _ = clf(answer("trainable"))
    d = run(c.classify(MSG, declared_class="trainable"))
    assert (d.source, d.error) == ("error", "invalid_answer")


def test_a_provider_that_ignores_the_schema_is_caught_by_the_classifier():
    class Rogue:
        async def decide(self, view):
            return Decision("trainable", "chat", None, 0.99, "model")

    c, _ = clf()
    c._provider = Rogue()
    d = run(c.classify(MSG))
    assert (d.source, d.error) == ("error", "invalid_answer")


def test_the_model_is_never_trusted_even_under_refuse():
    from routellm.decision import DecisionRefused

    c, _ = clf(answer("trainable"), on_error="refuse")
    with pytest.raises(DecisionRefused):
        run(c.classify(MSG))


# -- declaring one -------------------------------------------------------------------------


def test_a_declared_declarable_class_is_accepted_and_not_lowered():
    c, transport = clf(answer("general"))
    d = run(c.classify(MSG, declared_class="trainable"))
    assert (d.data_class, d.source, d.error) == ("trainable", "declared", None)
    assert len(transport.calls) == 1  # the classifier still runs


def test_the_model_can_raise_a_declared_declarable_class():
    c, _ = clf(answer("private"))
    d = run(c.classify(MSG, declared_class="trainable"))
    assert (d.data_class, d.source) == ("private", "model")


def test_the_model_raising_to_a_middle_class_works_too():
    c, _ = clf(answer("internal"))
    assert run(c.classify(MSG, declared_class="trainable")).data_class == "internal"


def test_a_class_outside_data_classes_is_still_unknown():
    c, _ = clf()
    d = run(c.classify(MSG, declared_class="nope"))
    assert d.error == "unknown_declared_class"


def test_an_undeclared_request_never_resolves_to_it():
    c, _ = clf(answer("general"))
    assert run(c.classify(MSG)).data_class == "general"


# -- sticky ------------------------------------------------------------------------------------


def test_a_declared_declarable_class_is_not_carried_into_a_later_undeclared_turn():
    c, transport = clf(answer("general"), sticky={"enabled": True})
    first = run(c.classify(MSG, declared_class="trainable", conversation_id="c1"))
    assert first.data_class == "trainable"
    second = run(c.classify(MSG, conversation_id="c1"))
    assert (second.data_class, second.source) == ("general", "model")
    assert len(transport.calls) == 2


def test_a_stricter_class_stays_sticky_past_a_declarable_turn():
    c, _ = clf(answer("internal"), sticky={"enabled": True})
    run(c.classify(MSG, conversation_id="c2"))
    c._provider._transport.reply = answer("general")
    d = run(c.classify(MSG, declared_class="trainable", conversation_id="c2"))
    assert d.data_class == "internal"
    d = run(c.classify(MSG, conversation_id="c2"))
    assert d.data_class == "internal"


# -- startup ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"declarable": ["ghost"]}, "'ghost' is not in data_classes"),
        ({"declarable": ["private"]}, "fail-closed"),
        ({"declarable": ["trainable", "trainable"]}, "twice"),
        (
            {
                "recognizers": [
                    {"name": "r", "type": "markers", "class": "trainable", "markers": ["x"]}
                ]
            },
            "declarable",
        ),
    ],
)
def test_bad_declarable_config_is_refused(over, needle):
    with pytest.raises(ValueError, match=needle):
        make_config(data_classes=CLASSES, **{"declarable": ["trainable"], **over})


def _policy(default_class):
    names = ["general", "trainable", "internal", "private"]
    return parse_policy(
        {
            "default_class": default_class,
            "classes": {n: {"treatment": "raw", "reach": ["local"], "tiers": ["t"]} for n in names},
        },
        "p.yaml",
    )


def test_a_default_class_naming_a_declarable_class_is_refused():
    from routellm.policy import check_decision_classes

    check_decision_classes(_policy("general"), list(CLASSES), ["trainable"])
    with pytest.raises(PolicyError, match="default_class 'trainable'"):
        check_decision_classes(_policy("trainable"), list(CLASSES), ["trainable"])


def test_a_routing_policy_floor_naming_a_declarable_class_is_refused():
    pol = class_policy(A)
    router = build_policy_routing({"policy_routing": routing_section(A)}, pol)
    cfg = SimpleNamespace(
        data_classes={"general": "", "private": ""},
        intents={"hard": ""},
        declarable=("general",),
    )
    with pytest.raises(PolicyRoutingError, match="declarable"):
        router.check(registry(), pol, cfg)
    ok = SimpleNamespace(data_classes=cfg.data_classes, intents=cfg.intents, declarable=())
    router.check(registry(), pol, ok)
