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
    ROUTE_CLIENT_HEADER,
    ROUTE_PROJECT_HEADER,
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
