"""Selector-form privacy cases: `tag:local !tag:cloud` on a tier side.

`private_selector` is `private` written as selectors. Resolved at
startup it must land on local leaves only: the cloud leaf is the
highest-quality one, so a selector that let it through would win it.
"""

from routellm.tests.privacy_fixture import IMAGE, TOOLS, case


def _status(r):
    return {"status": r.status_code}


@case("selector_text", 200, {"local_main": 1})
def selector_text(w):
    return _status(w.post("private_selector"))


@case("selector_image", 200, {"local_main": 1})
def selector_image(w):
    return _status(w.post("private_selector", IMAGE))


@case("selector_tools", 200, {"local_main": 1})
def selector_tools(w):
    return _status(w.post("private_selector", tools=TOOLS))


@case("selector_primary_down_falls_to_the_local_sibling", 200, {"local_main": 1, "local_small": 1})
def selector_primary_down(w):
    w.fail("local_main")
    return _status(w.post("private_selector"))


@case("selector_every_local_engine_down", 502, {"local_main": 1, "local_small": 1})
def selector_all_down(w):
    w.fail()
    return _status(w.post("private_selector"))


@case("selector_resolves_to_local_leaves_only")
def selector_resolved(w):
    tier = w.server.CONTROLLER.endpoints.get_tier("private_selector")
    return {"status": 200, "sides": [tier.strong, tier.weak]}
