"""Decision step: provider answers, recognizers, skip rule, failure, sticky."""

import asyncio
import time

import pytest

from routellm.decision import DecisionRefused
from routellm.decision.recognizers import (
    MarkerRecognizer,
    RegexRecognizer,
    SpanRecognizer,
)
from routellm.decision.sticky import StickyStore
from routellm.tests.decision_fakes import FakeTransport, answer, make_classifier

MSG = [{"role": "user", "content": "hello there"}]


def run(coro):
    return asyncio.run(coro)


# -- typed answer ---------------------------------------------------------


def test_typed_answer_parsed():
    clf, transport = make_classifier(
        FakeTransport(answer("internal", "code", 0.8, area="net")), areas={"net": "networking"}
    )
    d = run(clf.classify(MSG))
    assert (d.data_class, d.intent, d.area, d.confidence, d.source) == (
        "internal",
        "code",
        "net",
        0.8,
        "model",
    )
    # the request carried a schema whose enums are the configured labels
    schema = transport.calls[0]["format"]["json_schema"]["schema"]
    assert schema["properties"]["data_class"]["enum"] == ["general", "internal", "private"]


@pytest.mark.parametrize(
    "reply",
    [
        answer("secret"),  # data class outside the set
        answer("general", "poetry"),  # intent outside the set
        "not json",
        "[1]",
        '{"data_class": "general", "intent": "chat"}',  # no confidence
        answer(confidence=1.5),
    ],
)
def test_invalid_answer_takes_on_error_path(reply):
    clf, _ = make_classifier(FakeTransport(reply))
    d = run(clf.classify(MSG))
    assert (d.source, d.error, d.data_class) == ("error", "invalid_answer", "private")


def test_unknown_area_label_is_invalid():
    clf, _ = make_classifier(FakeTransport(answer(area="mars")), areas={"net": "networking"})
    assert run(clf.classify(MSG)).error == "invalid_answer"


# -- fail closed -----------------------------------------------------------


def test_timeout_is_most_restrictive_and_recorded():
    clf, _ = make_classifier(FakeTransport(delay=1.0), timeout_ms=50)
    d = run(clf.classify(MSG))
    assert (d.data_class, d.source, d.error) == ("private", "error", "timeout")


def test_transport_error_is_most_restrictive_and_recorded():
    clf, _ = make_classifier(FakeTransport(ConnectionError("down")))
    d = run(clf.classify(MSG))
    assert (d.data_class, d.error) == ("private", "transport")
    assert "down" in d.detail["message"]


def test_low_confidence_is_error_path():
    clf, _ = make_classifier(FakeTransport(answer("general", "chat", 0.2)))
    d = run(clf.classify(MSG))
    assert (d.data_class, d.error) == ("private", "low_confidence")


def test_on_error_refuse_raises():
    clf, _ = make_classifier(FakeTransport(ConnectionError("down")), on_error="refuse")
    with pytest.raises(DecisionRefused) as info:
        run(clf.classify(MSG))
    assert info.value.error.kind == "transport"


def test_unknown_declared_class_is_error_path():
    clf, transport = make_classifier()
    d = run(clf.classify(MSG, declared_class="bogus"))
    assert (d.error, d.data_class) == ("unknown_declared_class", "private")
    assert transport.calls == []


# -- recognizers and skip rule ---------------------------------------------


def test_recognizer_hit_makes_zero_calls():
    rec = RegexRecognizer("key", "private", [r"sk-[a-z0-9]{8,}"])
    clf, transport = make_classifier(recognizers=[rec])
    d = run(clf.classify([{"role": "user", "content": "use sk-abcdef123456 here"}]))
    assert transport.calls == []
    assert (d.data_class, d.source, d.detail["recognizer"]) == ("private", "recognizer", "key")


def test_recognizer_scans_beyond_the_model_window():
    rec = MarkerRecognizer("m", "private", ["CONFIDENTIAL"])
    clf, transport = make_classifier(recognizers=[rec])
    msgs = [
        {"role": "user", "content": "CONFIDENTIAL plan"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and now?"},
    ]
    assert run(clf.classify(msgs)).source == "recognizer"
    assert transport.calls == []


def test_recognizers_are_ordered_first_hit_wins():
    a = MarkerRecognizer("a", "internal", ["alpha"])
    b = MarkerRecognizer("b", "private", ["alpha"])
    clf, _ = make_classifier(recognizers=[a, b])
    d = run(clf.classify([{"role": "user", "content": "alpha"}]))
    assert d.detail["recognizer"] == "a"


def test_no_recognizer_hit_asks_the_model():
    rec = MarkerRecognizer("m", "private", ["zzz"])
    clf, transport = make_classifier(recognizers=[rec])
    assert run(clf.classify(MSG)).source == "model"
    assert len(transport.calls) == 1


def test_declared_most_restrictive_makes_zero_calls():
    clf, transport = make_classifier()
    d = run(clf.classify(MSG, declared_class="private"))
    assert transport.calls == []
    assert (d.data_class, d.source) == ("private", "declared")


def test_declared_lower_class_is_a_floor_for_the_model():
    clf, transport = make_classifier(FakeTransport(answer("general", "code")))
    d = run(clf.classify(MSG, declared_class="internal"))
    assert len(transport.calls) == 1
    assert (d.data_class, d.source, d.intent) == ("internal", "declared", "code")


def test_blocking_recognizer_runs_off_the_loop():
    class Slow:
        def analyze(self, text, language):
            time.sleep(0.2)
            return [{"entity_type": "EMAIL", "score": 0.9}]

    rec = SpanRecognizer("pii", "private", Slow(), entity_types=["EMAIL"])
    clf, transport = make_classifier(recognizers=[rec])

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        d = await clf.classify(MSG)
        task.cancel()
        return d, ticks

    d, ticks = run(scenario())
    assert (d.source, transport.calls) == ("recognizer", [])
    assert ticks >= 5


def test_span_recognizer_filters_type_and_score():
    class A:
        def analyze(self, text, language):
            return [{"entity_type": "PERSON", "score": 0.2}, {"entity_type": "X", "score": 0.99}]

    assert SpanRecognizer("p", "private", A(), ["PERSON"], 0.5).recognize("t") is None
    assert SpanRecognizer("p", "private", A(), [], 0.5).recognize("t").detail["entity_type"] == "X"


# -- event loop ------------------------------------------------------------


def test_slow_decision_does_not_block_unrelated_work():
    clf, _ = make_classifier(FakeTransport(delay=0.3), timeout_ms=2000)

    async def scenario():
        started = time.perf_counter()
        slow = asyncio.create_task(clf.classify(MSG))
        await asyncio.sleep(0)
        await asyncio.sleep(0.02)  # unrelated request
        unrelated = time.perf_counter() - started
        await slow
        return unrelated

    assert run(scenario()) < 0.15


# -- window ----------------------------------------------------------------


def test_input_window_last_user_and_truncation():
    clf, transport = make_classifier(
        input={"messages": "last_user", "max_chars": 10, "keep": "tail"}
    )
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "0123456789ABCDEF"},
    ]
    run(clf.classify(msgs))
    sent = transport.calls[0]["messages"][-1]["content"]
    assert "6789ABCDEF" in sent and "first" not in sent and "012345" not in sent


def test_string_messages_accepted():
    clf, transport = make_classifier()
    run(clf.classify("just a string"))
    assert "just a string" in transport.calls[0]["messages"][-1]["content"]


# -- sticky ----------------------------------------------------------------


def test_sticky_turn_two_stays_restrictive():
    replies = iter([answer("private", "chat"), answer("general", "chat")])
    clf, transport = make_classifier(
        FakeTransport(lambda m: next(replies)), sticky={"enabled": True}
    )
    first = [{"role": "user", "content": "my secret"}]
    second = [
        *first,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "weather?"},
    ]
    assert run(clf.classify(first)).data_class == "private"
    d = run(clf.classify(second))
    assert d.data_class == "private" and d.source == "sticky"
    # most restrictive seen -> no second model call
    assert len(transport.calls) == 1


def test_sticky_floor_lifts_a_lower_model_answer():
    replies = iter([answer("internal"), answer("general", "code")])
    clf, _transport = make_classifier(
        FakeTransport(lambda m: next(replies)), sticky={"enabled": True}
    )
    first = [{"role": "user", "content": "plan"}]
    run(clf.classify(first))
    d = run(clf.classify([*first, {"role": "user", "content": "more"}]))
    assert (d.data_class, d.source, d.intent) == ("internal", "sticky", "code")


def test_sticky_is_per_conversation_and_expires():
    now = [0.0]
    store = StickyStore(ttl_s=10, clock=lambda: now[0])
    replies = iter([answer("private"), answer("general"), answer("general")])
    from routellm.decision import DecisionClassifier
    from routellm.decision.llm import LLMProvider
    from routellm.tests.decision_fakes import make_config

    cfg = make_config()
    tr = FakeTransport(lambda m: next(replies))
    clf = DecisionClassifier(cfg, provider=LLMProvider(cfg, tr), sticky=store)
    run(clf.classify([{"role": "user", "content": "A"}], conversation_id="c1"))
    other = run(clf.classify([{"role": "user", "content": "B"}], conversation_id="c2"))
    assert other.data_class == "general"
    assert run(clf.classify(MSG, conversation_id="c1")).source == "sticky"
    now[0] = 11
    assert run(clf.classify(MSG, conversation_id="c1")).data_class == "general"


def test_error_class_is_not_made_sticky():
    replies = iter([ConnectionError("down"), answer("general")])
    clf, _ = make_classifier(FakeTransport(lambda m: next(replies)), sticky={"enabled": True})
    assert run(clf.classify(MSG)).error == "transport"
    assert run(clf.classify(MSG)).data_class == "general"


def test_sticky_off_by_default():
    replies = iter([answer("private"), answer("general")])
    clf, _ = make_classifier(FakeTransport(lambda m: next(replies)))
    run(clf.classify(MSG))
    assert run(clf.classify(MSG)).data_class == "general"
