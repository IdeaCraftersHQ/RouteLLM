"""Class-policy privacy cases: a policy deny is a 403 and never a downgrade.

These run in the class-policy world (see `privacy_fixture.run_cases`): a
policy file, an authenticated caller, and tiers no class may enter at
startup. A case admits one to `private` after startup (`World.admit`),
which is the change the startup walk never saw, so the request-time
check is all that stands between the request and an off-host leaf. A
denied leaf takes zero hits, and so does every other leaf the request
could have been moved to.
"""

from routellm.tests.privacy_fixture import TOOLS, case


def _status(r):
    return {"status": r.status_code}


def _reason(r):
    return {"status": r.status_code, "message": r.json().get("message", "")}


@case("policy_private_is_served_locally", 200, {"local_main": 1}, policy=True)
def served(w):
    return _status(w.post("private", data_class="private"))


@case("policy_deny_is_403_and_never_downgrades_to_local", 403, policy=True)
def deny_never_downgrades(w):
    # Both sides of the tier are off-host. Nothing is called, not even
    # the local leaves the class could have used: a deny is a refusal,
    # not a reroute.
    w.admit("private", "leaky_both")
    return _reason(w.post("leaky_both", data_class="private"))


@case("policy_tier_outside_the_class_is_403", 403, policy=True)
def outside_tier(w):
    return _reason(w.post("general", data_class="private"))


@case("policy_denied_side_is_skipped_for_the_allowed_one", 200, {"local_main": 1}, policy=True)
def side_skipped(w):
    w.admit("private", "leaky_forced_local")
    return _status(w.post("leaky_forced_local", data_class="private"))


@case("policy_forced_side_onto_a_cloud_leaf_is_403", 403, policy=True)
def forced_cloud(w):
    # `local_audio` takes no tools, so the cloud leaf is the only side
    # that could serve: the class refuses it.
    w.admit("private", "leaky_forced")
    return _reason(w.post("leaky_forced", tools=TOOLS, data_class="private"))


@case("policy_cloud_sibling_is_never_called", 403, {"local_main": 1}, policy=True)
def cloud_sibling(w):
    w.admit("private", "leaky_sibling")
    w.fail("local_main")
    return _reason(w.post("leaky_sibling", data_class="private"))


@case("policy_cloud_sibling_streamed_is_never_called", 403, {"local_main": 1}, policy=True)
def cloud_sibling_stream(w):
    w.admit("private", "leaky_sibling")
    w.fail("local_main", mode="die_first")
    status, _events = w.stream("leaky_sibling", data_class="private")
    return {"status": status}


@case("policy_denied_pool_members_are_never_called", 403, total=2, policy=True)
def pool_walk(w):
    w.admit("private", "leaky_pool")
    w.fail("local_small", "local_audio")
    return _reason(w.post("leaky_pool", data_class="private"))


# -- controls: the same tiers, a class that may reach the cloud --------------


@case(
    "policy_control_leaky_class_reaches_the_cloud_sibling",
    200,
    {"local_main": 1},
    reaches_off_host=True,
    policy=True,
)
def control_sibling(w):
    w.fail("local_main")
    return _status(w.post("leaky_sibling", data_class="leaky"))


@case("policy_control_general_class_reaches_the_cloud", 200, reaches_off_host=True, policy=True)
def control_general(w):
    return _status(w.post("general", data_class="general"))
