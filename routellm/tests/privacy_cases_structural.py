"""Structural privacy cases: the private tier reaches local leaves only.

Each case is a function over `World` (see `privacy_fixture`) and is
checked by `check`: the registered status, the registered hits on local
leaves, and zero hits on every off-host fake. The `general_*` cases are
the control: they must reach the off-host fakes.
"""

from routellm.tests.privacy_fixture import AUDIO, IMAGE, TOOLS, case


def _status(r):
    return {"status": r.status_code}


# -- the private class, every request shape ---------------------------------


@case("text", 200, {"local_main": 1})
def text(w):
    return _status(w.post("private"))


@case("image", 200, {"local_main": 1})
def image(w):
    return _status(w.post("private", IMAGE))


@case("tools", 200, {"local_main": 1})
def tools(w):
    return _status(w.post("private", tools=TOOLS))


@case("stream", 200, {"local_main": 1})
def stream(w):
    status, events = w.stream("private")
    return {"status": status, "done": events[-1] == "[DONE]"}


@case("stream_tools", 200, {"local_main": 1})
def stream_tools(w):
    status, events = w.stream("private", tools=TOOLS)
    return {"status": status, "done": events[-1] == "[DONE]"}


@case("audio_without_a_hearing_leaf_is_refused", 400)
def audio_refused(w):
    return _status(w.post("private", AUDIO))


@case("audio_goes_to_the_local_leaf_that_hears", 200, {"local_audio": 1})
def audio_served(w):
    return _status(w.post("private_audio", AUDIO))


# -- local engines failing --------------------------------------------------


@case("primary_down_small_request_stays_local", 200, {"local_main": 1, "local_small": 1})
def primary_down(w):
    w.fail("local_main")
    return _status(w.post("private"))


@case("primary_down_agent_sized_request_is_an_error", 502, {"local_main": 1})
def primary_down_large(w):
    # Prompt plus max_tokens beyond the sibling's window: no local side is left.
    w.fail("local_main")
    return _status(w.post("private", max_tokens=9000))


@case("every_local_engine_down", 502, {"local_main": 1, "local_small": 1})
def all_down(w):
    w.fail()
    return _status(w.post("private"))


@case("image_with_every_engine_down", 502, {"local_main": 1})
def image_all_down(w):
    w.fail()
    return _status(w.post("private", IMAGE))


@case("tools_with_every_engine_down", 502, {"local_main": 1, "local_small": 1})
def tools_all_down(w):
    w.fail()
    return _status(w.post("private", tools=TOOLS))


@case("stream_primary_drops_before_first_chunk", 200, {"local_main": 1, "local_small": 1})
def stream_primary_drops(w):
    w.fail("local_main", mode="die_first")
    status, events = w.stream("private")
    return {"status": status, "done": events[-1] == "[DONE]"}


@case("stream_with_every_engine_down", 502, {"local_main": 1, "local_small": 1})
def stream_all_down(w):
    w.fail(mode="die_first")
    status, events = w.stream("private")
    return {"status": status, "events": len(events)}


@case("breaker_open_on_every_local_leaf", 502, {"local_main": 1, "local_small": 1})
def breaker_open(w):
    # Needs `cb_fail_max=1` and a long wait (see the test): one failing
    # request opens both breakers, and the next is refused without a call.
    w.fail()
    first = w.post("private")
    open_ = {
        name: w.server.CONTROLLER.resilience.get_cb(name).state.value
        for name in ("local_main", "local_small")
    }
    second = w.post("private")
    return {"status": second.status_code, "first": first.status_code, "breakers": open_}


# -- refusals before any call -----------------------------------------------


@case("oversized_prompt_is_a_400_with_no_call", 400)
def oversized(w):
    return _status(w.post("private", max_tokens=300000))


# -- addressing forms -------------------------------------------------------


@case("legacy_router_form_lands_on_default", 200, {"local_main": 1})
def legacy(w):
    return _status(w.post("router-random-0.0"))


@case("legacy_router_form_still_local_when_the_weak_side_wins", 200, total=1)
def legacy_weak(w):
    return _status(w.post("router-random-1.0"))


@case("router_tier_form", 200, {"local_main": 1})
def router_tier(w):
    return _status(w.post("router-private"))


@case("bare_default_tier", 200, {"local_main": 1})
def bare_default(w):
    return _status(w.post("default"))


# -- control: the general class does reach off the host ---------------------


@case("general_reaches_cloud", 200, reaches_off_host=True)
def general(w):
    return _status(w.post("general"))


@case("general_falls_back_to_gated", 200, reaches_off_host=True)
def general_gated(w):
    w.fail("cloud_main")
    r = w.post("general")
    return {"status": r.status_code, "gated": len(w.fakes["gated_main"].hits)}
