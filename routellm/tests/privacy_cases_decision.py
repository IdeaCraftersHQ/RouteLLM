"""Decision-step privacy cases: a restrictive class reaches local leaves only.

A request gets its class from a declaration, a deterministic recognizer or
the classifier model (the loopback `decider` fake). Whichever source names
a restrictive class, the request is served by a local leaf on every entry
tier it may enter, and a tier the class may not enter refuses it (403)
with zero hits. The classifier failing resolves to the strictest class
(`on_error: most_restrictive`) or to a 400 (`refuse`); a conversation that
was once restrictive stays so; the model can raise a declared class and
never lower it. The controls use a benign request the model labels
`general`: it reaches the cloud, so the zeros mean something.

`obs["decider"]` counts the model calls; `obs["class"]` and `obs["source"]`
read the decision the response carries.
"""

import json

from routellm.tests.privacy_decision_world import INTERNAL_MARKER, PRIVATE_MARKER
from routellm.tests.privacy_fixture import OPEN_PROJECT, ROUTE_PROJECT_HEADER, case

#: Tiers a restrictive request may enter: direct, selector-written, the
#: default one, the audio one, and the legacy `router-` form.
DIRECT_TIERS = ["private", "private_selector", "default", "private_audio", "router-random-0.5"]
ENTRY = "route_entry"
OPEN = {ROUTE_PROJECT_HEADER: OPEN_PROJECT}
BENIGN = [{"role": "user", "content": "what is the capital of france"}]


def _say(w, data_class, confidence=0.9):
    w.fakes["decider"].reply = json.dumps({"data_class": data_class, "confidence": confidence})


def _user(text):
    return [{"role": "user", "content": text}]


def _obs(w, r):
    out = {"status": r.status_code, "decider": len(w.fakes["decider"].hits)}
    if r.status_code != 200:
        return {**out, "message": r.json().get("message", "")}
    decision = r.json()["routellm"]["path"][0]["decision"]
    return {
        **out,
        "class": decision["data_class"],
        "source": decision["source"],
        "error": decision.get("error"),
    }


# -- every source, every entry tier ------------------------------------------------


def _by_declaration(w, tier, extra=None):
    _say(w, "general")
    return _obs(w, w.post(tier, BENIGN, data_class="private", headers=extra))


def _by_recognizer(w, tier, extra=None):
    _say(w, "general")
    return _obs(w, w.post(tier, _user(f"see {PRIVATE_MARKER} here"), headers=extra))


def _by_classifier(w, tier, extra=None):
    _say(w, "private")
    return _obs(w, w.post(tier, BENIGN, headers=extra))


SOURCES = {
    "declaration": _by_declaration,
    "recognizer": _by_recognizer,
    "classifier": _by_classifier,
}


def _register(source, tier, fn, status, total, **flags):
    name = f"decision_{source}_on_{tier}_{'is_served_locally' if status == 200 else 'is_403'}"
    case(name, status, total=total, decision=True, **flags)(lambda w, fn=fn, tier=tier: fn(w, tier))


for _source, _fn in SOURCES.items():
    for _tier in DIRECT_TIERS:
        _register(_source, _tier, _fn, 200, 1)
    # a tier the class may not enter: refused, nothing called
    _register(_source, "general", _fn, 403, 0)


@case("decision_a_private_marker_in_an_earlier_turn_counts", 200, total=1, decision=True)
def earlier_turn(w):
    # The recognizers read every message, not only the window the model sees.
    _say(w, "general")
    turns = [
        {"role": "user", "content": f"note {PRIVATE_MARKER}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and now something harmless"},
    ]
    return _obs(w, w.post("private", turns))


@case("decision_a_recognizer_hit_settles_the_class_without_the_model", 200, total=1, decision=True)
def recognizer_skips_model(w):
    # `internal` is below the strictest class, so only the recognizer
    # hit (not the strictest-class shortcut) can skip the model.
    _say(w, "general")
    return _obs(w, w.post("private", _user(f"see {INTERNAL_MARKER}")))


# -- the controls: a benign request the model labels general reaches the cloud --------


@case(
    "decision_control_a_general_label_reaches_the_cloud", 200, reaches_off_host=True, decision=True
)
def control(w):
    _say(w, "general")
    return _obs(w, w.post("general", BENIGN))


# -- the classifier down ------------------------------------------------------------------


def _down_500(w):
    w.fakes["decider"].mode = "500"


def _down_timeout(w):
    w.fakes["decider"].delay = 1.5


def _down_not_json(w):
    w.fakes["decider"].reply = "no label here"


def _down_bad_label(w):
    w.fakes["decider"].reply = json.dumps({"data_class": "bogus", "confidence": 0.9})


def _down_unsure(w):
    _say(w, "general", confidence=0.1)


DOWN = {
    "returns_5xx": (_down_500, "transport"),
    "times_out": (_down_timeout, "timeout"),
    "answers_with_no_json": (_down_not_json, "invalid_answer"),
    "answers_with_an_unknown_label": (_down_bad_label, "invalid_answer"),
    "is_unsure": (_down_unsure, "low_confidence"),
}


def _down_case(label, fail, kind):
    @case(f"decision_classifier_{label}_falls_to_the_strictest_class", 200, total=1, decision=True)
    def served_local(w):
        fail(w)
        return {**_obs(w, w.post("private", BENIGN)), "kind": kind}

    @case(f"decision_classifier_{label}_never_opens_the_general_tier", 403, total=0, decision=True)
    def refused(w):
        fail(w)
        return _obs(w, w.post("general", BENIGN))

    @case(f"decision_classifier_{label}_under_refuse_is_400", 400, total=0, decision=True)
    def refuse(w):
        config = w.server.CONTROLLER.decision.config
        config.on_error = "refuse"
        try:
            fail(w)
            return _obs(w, w.post("private", BENIGN))
        finally:
            config.on_error = "most_restrictive"


for _label, (_fail, _kind) in DOWN.items():
    _down_case(_label, _fail, _kind)


@case("decision_classifier_down_does_not_lower_a_declared_class", 200, total=1, decision=True)
def down_declared(w):
    _down_500(w)
    return _obs(w, w.post("private", BENIGN, data_class="internal"))


# -- the sticky conversation -----------------------------------------------------------


def _follow_up(first):
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "and what is two plus two"},
    ]


def _two_turns(w, tier, first_label, headers=None):
    first = f"opening of a conversation about {first_label}"
    _say(w, first_label)
    one = _obs(w, w.post(tier, _user(first), headers=headers))
    _say(w, "general")
    two = _obs(w, w.post(tier, _follow_up(first), headers=headers))
    return {"status": max(one["status"], two["status"]), "first": one, "second": two}


@case("decision_a_restrictive_turn_keeps_later_turns_restrictive", 200, total=2, decision=True)
def sticky_private(w):
    return _two_turns(w, "private", "private")


@case("decision_a_mid_class_turn_keeps_later_turns_at_that_class", 200, total=2, decision=True)
def sticky_internal(w):
    return _two_turns(w, "private", "internal")


@case(
    "decision_a_benign_first_turn_does_not_make_the_conversation_restrictive",
    200,
    reaches_off_host=True,
    decision=True,
)
def sticky_control(w):
    first = "opening about the weather"
    _say(w, "general")
    one = _obs(w, w.post("general", _user(first)))
    two = _obs(w, w.post("general", _follow_up(first)))
    return {"status": max(one["status"], two["status"]), "first": one, "second": two}


# -- the classifier raises a declared class, never lowers it -----------------------------


@case("decision_the_classifier_cannot_lower_a_declared_class", 200, total=1, decision=True)
def not_lowered(w):
    _say(w, "general")
    return _obs(w, w.post("private", BENIGN, data_class="internal"))


@case("decision_the_classifier_can_raise_a_declared_class", 200, total=1, decision=True)
def raised(w):
    _say(w, "private")
    return _obs(w, w.post("private", BENIGN, data_class="general"))


@case("decision_a_raised_class_cannot_enter_the_tier_it_was_declared_for", 403, decision=True)
def raised_refused(w):
    _say(w, "private")
    return _obs(w, w.post("general", BENIGN, data_class="general"))


# -- policy routing: the entry tier is replaced by the rules' pick ------------------------


def _route(name, fn, status=200, total=1, **kw):
    case(f"decision_route_{name}", status, total=total, decision=True, route=True, **kw)(fn)


for _source, _fn in SOURCES.items():
    _route(
        f"{_source}_on_the_entry_tier_is_served_locally",
        lambda w, fn=_fn: fn(w, ENTRY, OPEN),
    )


def _route_down(w):
    _down_500(w)
    return _obs(w, w.post(ENTRY, BENIGN, headers=OPEN))


_route("classifier_down_on_the_entry_tier_is_served_locally", _route_down)


def _route_refuse(w):
    config = w.server.CONTROLLER.decision.config
    config.on_error = "refuse"
    try:
        _down_500(w)
        return _obs(w, w.post(ENTRY, BENIGN, headers=OPEN))
    finally:
        config.on_error = "most_restrictive"


_route("classifier_down_under_refuse_is_400", _route_refuse, status=400, total=0)


def _route_sticky(w):
    return _two_turns(w, ENTRY, "private", OPEN)


_route("a_restrictive_turn_keeps_later_turns_restrictive", _route_sticky, total=2)


def _route_control(w):
    _say(w, "general")
    return _obs(w, w.post(ENTRY, BENIGN, headers=OPEN))


_route(
    "control_a_general_label_reaches_the_cloud", _route_control, total=None, reaches_off_host=True
)
