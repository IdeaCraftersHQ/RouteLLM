"""Policy-routing privacy cases: the project attribute can only narrow.

These run in the route world (see `privacy_fixture.route_config`): the
request addresses the entry tier and a `policy_routing:` section picks
the tier. The default policy is fail-closed (floor `private`, private
tier only), so a request whose project is missing, unknown or malformed
never leaves the host, whatever class it declares. A policy cannot widen
a class: a rule that points a private request at the cloud tier is
refused (403) with zero off-host hits. The controls use the open project
and a lower class: the cloud fake can be hit, so the zeros mean something.
"""

from routellm.tests.privacy_fixture import (
    OPEN_PROJECT,
    RESTRICTED_PROJECT,
    ROUTE_CLIENT_HEADER,
    ROUTE_PROJECT_HEADER,
    TAGGED_PROJECT,
    case,
)

ENTRY = "route_entry"


def _project(value):
    return {ROUTE_PROJECT_HEADER: value}


def _status(r):
    return {"status": r.status_code}


def _reason(r):
    return {"status": r.status_code, "message": r.json().get("message", "")}


def _record(r):
    return r.json()["routellm"]["path"][0]["policy_routing"]


@case(
    "route_no_project_header_takes_the_default_policy_and_stays_local",
    200,
    {"local_main": 1},
    route=True,
)
def no_header(w):
    r = w.post(ENTRY, data_class="general")
    return {
        "status": r.status_code,
        "policy": _record(r)["policy"],
        "from": _record(r)["policy_from"],
    }


@case(
    "route_unknown_project_takes_the_default_policy_and_stays_local",
    200,
    {"local_main": 1},
    route=True,
)
def unknown(w):
    r = w.post(ENTRY, data_class="general", headers=_project("nobody"))
    return {
        "status": r.status_code,
        "policy": _record(r)["policy"],
        "from": _record(r)["policy_from"],
    }


@case(
    "route_invalid_project_takes_the_default_policy_and_stays_local",
    200,
    {"local_main": 1},
    route=True,
)
def invalid(w):
    r = w.post(ENTRY, data_class="general", headers=_project("../etc passwd"))
    return {
        "status": r.status_code,
        "policy": _record(r)["policy"],
        "from": _record(r)["policy_from"],
    }


@case("route_a_policy_cannot_widen_a_class", 403, route=True)
def no_widening(w):
    # The client rule comes first and names the cloud tier. Declared
    # private, the request may not enter it: refused, nothing called.
    headers = {**_project(OPEN_PROJECT), ROUTE_CLIENT_HEADER: "ci"}
    return _reason(w.post(ENTRY, data_class="private", headers=headers))


@case(
    "route_a_private_request_under_the_open_project_stays_local", 200, {"local_main": 1}, route=True
)
def private_open(w):
    return _status(w.post(ENTRY, data_class="private", headers=_project(OPEN_PROJECT)))


@case(
    "route_the_default_floor_is_not_lowered_by_a_declared_general_class",
    200,
    {"local_main": 1},
    route=True,
)
def not_lowered(w):
    # Declared general under the default policy: its floor wins.
    return _status(w.post(ENTRY, data_class="general", headers=_project("")))


@case("route_attribute_headers_never_reach_an_upstream", 200, {"local_main": 1}, route=True)
def not_forwarded(w):
    headers = {**_project(OPEN_PROJECT), ROUTE_CLIENT_HEADER: "tool"}
    r = w.post(ENTRY, data_class="private", headers=headers)
    seen = sorted(
        {k.lower() for f in w.fakes.values() for _p, hdrs, _b in f.hits for k in hdrs}
        & {ROUTE_PROJECT_HEADER.lower(), ROUTE_CLIENT_HEADER.lower()}
    )
    return {"status": r.status_code, "forwarded": seen}


# -- control: the open project, a lower class ---------------------------------------


@case(
    "route_control_open_project_general_reaches_the_cloud", 200, reaches_off_host=True, route=True
)
def control(w):
    r = w.post(ENTRY, data_class="general", headers=_project(OPEN_PROJECT))
    return {"status": r.status_code, "policy": _record(r)["policy"]}


@case("route_a_spoofed_project_is_403_and_takes_no_hit", 403, route=True)
def spoofed(w):
    # The caller's policy-file entry does not list this project: naming
    # it in a header is refused, not downgraded, and nothing is called.
    return _reason(w.post(ENTRY, data_class="general", headers=_project(RESTRICTED_PROJECT)))


# -- /v1/compare and aux calls: the floor and the tag lists apply there too ---------------


COMPARED = ["local_main", "cloud_main"]


@case("route_compare_under_a_restrictive_project_floor_takes_no_hit", 403, route=True)
def compare_floor(w):
    # Declared general, but the default policy's floor is private: the
    # cloud target is refused, so nothing is called at all.
    return _reason(w.compare(COMPARED, data_class="general"))


@case("route_compare_with_no_declared_class_is_held_to_the_floor", 403, route=True)
def compare_floor_undeclared(w):
    return _reason(w.compare(COMPARED))


@case("route_compare_under_a_tag_denying_project_takes_no_cloud_hit", 403, route=True)
def compare_deny_tags(w):
    return _reason(w.compare(COMPARED, data_class="general", headers=_project(TAGGED_PROJECT)))


@case("route_compare_a_policy_the_caller_may_not_select_is_403", 403, route=True)
def compare_spoofed(w):
    return _reason(w.compare(COMPARED, data_class="general", headers=_project(RESTRICTED_PROJECT)))


@case(
    "route_compare_control_open_project_reaches_the_cloud",
    200,
    {"local_main": 1},
    route=True,
    reaches_off_host=True,
)
def compare_control(w):
    return _status(w.compare(COMPARED, data_class="general", headers=_project(OPEN_PROJECT)))


@case(
    "route_no_declared_class_takes_the_project_floor_and_stays_local",
    200,
    {"local_main": 1},
    route=True,
)
def floor_supplies_class(w):
    # No header, no body class, default_class null: the default policy's
    # floor (private) is the class, so the request is routed at the floor.
    r = w.post(ENTRY)
    if r.status_code != 200:
        return _reason(r)
    return {
        "status": r.status_code,
        "class": _record(r)["effective_class"],
        "classes": _record(r)["classes"],
    }


@case(
    "route_aux_detector_gets_zero_prompts_under_a_tag_denying_policy",
    200,
    {"local_main": 1},
    route=True,
    aux=True,
)
def aux_detector_tag_denied(w):
    for middleware in w.server.CONTROLLER.middleware:
        middleware.intent_cache.clear()
    return _status(w.post(ENTRY, data_class="general", headers=_project(TAGGED_PROJECT)))


@case(
    "route_aux_control_detector_is_called_under_a_policy_without_the_tag",
    200,
    route=True,
    aux=True,
    reaches_off_host=True,
    total=0,
)
def aux_detector_control(w):
    for middleware in w.server.CONTROLLER.middleware:
        middleware.intent_cache.clear()
    return _status(w.post(ENTRY, data_class="general", headers=_project(OPEN_PROJECT)))


@case(
    "route_a_floor_raised_request_is_never_served_from_the_declared_class_cache",
    200,
    {"local_main": 2},
    route=True,
)
def floor_raised_cache(w):
    # Declared general (cache: exact) under the default policy, whose floor
    # is private (block): the effective class decides the cache, so the
    # second identical request is not served from an entry the first stored.
    cache = w.server.CONTROLLER.cache.config
    cache.enabled = True
    try:
        statuses = [w.post(ENTRY, data_class="general").status_code for _ in range(2)]
    finally:
        cache.enabled = False
    return {"status": max(statuses)}
