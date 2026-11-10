"""Egress-gate privacy cases: a gated leaf takes zero hits for a denied class.

These run in the gate world (see `privacy_fixture.run_cases`, `gate=True`):
the class-policy world plus an `egress_gates:` entry on the `gated` tag
that denies the `leaky` class, and an `intents:` detector whose endpoint
is the gated leaf. `leaky` may reach the cloud by policy, so only the gate
stands between it and the gated fake. A refused leaf takes zero hits on
every path a request can take to it: primary, forced side, sibling
fallback (streamed too), candidate pool, and the intent detector. The
controls send the `general` class, which the gate allows: the fakes can
be hit, so the zeros mean something.
"""

from routellm.tests.privacy_fixture import TOOLS, case


def _status(r):
    return {"status": r.status_code}


def _fresh_detector(w):
    """Forget detected intents: a cached prompt would hide whether the detector was called."""
    for middleware in w.server.CONTROLLER.middleware:
        middleware.intent_cache.clear()


def _reason(r):
    return {"status": r.status_code, "message": r.json().get("message", "")}


@case("gate_primary_is_never_called_the_ungated_side_serves", 200, {"local_small": 1}, gate=True)
def primary(w):
    return _status(w.post("gate_primary", data_class="leaky"))


@case("gate_forced_side_onto_the_gated_leaf_is_403", 403, gate=True)
def forced(w):
    # `local_audio` takes no tools, so the gated leaf is the only side
    # that could serve: the gate refuses it, nothing else is tried.
    return _reason(w.post("gate_forced", tools=TOOLS, data_class="leaky"))


@case("gate_sibling_is_never_called", 403, {"local_main": 1}, gate=True)
def sibling(w):
    w.fail("local_main")
    return _reason(w.post("gate_sibling", data_class="leaky"))


@case("gate_sibling_streamed_is_never_called", 403, {"local_main": 1}, gate=True)
def sibling_stream(w):
    w.fail("local_main", mode="die_first")
    status, _events = w.stream("gate_sibling", data_class="leaky")
    return {"status": status}


@case("gate_pool_members_are_skipped_the_next_allowed_one_serves", 200, total=3, gate=True)
def pool(w):
    # Ranked by quality: local_small, local_audio, gated_main, local_main.
    w.fail("local_small", "local_audio")
    return _status(w.post("gate_pool", data_class="leaky"))


@case("gate_detector_gets_no_restricted_prompt", 200, {"local_main": 1}, gate=True)
def detector(w):
    # The detector's endpoint is the gated leaf: it is not asked at all,
    # so the gated fake sees no prompt even though the tier has it as a side.
    _fresh_detector(w)
    return _status(w.post("gate_sibling", data_class="leaky"))


# -- controls: the class the gate allows -------------------------------------


@case("gate_control_allowed_class_reaches_the_gated_primary", 200, reaches_off_host=True, gate=True)
def control_primary(w):
    return _status(w.post("gate_primary", data_class="general"))


@case(
    "gate_control_allowed_class_falls_back_to_the_gated_sibling",
    200,
    {"local_main": 1},
    reaches_off_host=True,
    gate=True,
)
def control_sibling(w):
    w.fail("local_main")
    return _status(w.post("gate_sibling", data_class="general"))


@case(
    "gate_control_allowed_class_is_classified_by_the_gated_detector",
    200,
    {"local_main": 1},
    reaches_off_host=True,
    gate=True,
)
def control_detector(w):
    _fresh_detector(w)
    return _status(w.post("gate_sibling", data_class="general"))
