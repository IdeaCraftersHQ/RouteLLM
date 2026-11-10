"""Auxiliary-call privacy cases: calls that read a prompt before any leaf does.

These run in the auxiliary world (see `privacy_fixture.AUX_CONFIG`): the
class-policy world plus an intent detector, an `embedding` endpoint and a
router that embeds, every one pointing at the cloud fake. The `private`
class reaches only local leaves, so each of those calls is a remote call
it must never make: the cloud fake takes zero hits and the request is
still answered by an allowed local leaf. The controls send the `leaky`
class, which may reach the cloud: the fake is hit, so the zeros mean
something.
"""

import tempfile
from pathlib import Path

from routellm.tests.privacy_fixture import case


def _status(r):
    return {"status": r.status_code}


def _fresh_detector(w):
    """Forget detected intents: a cached prompt would hide whether the detector was called."""
    for middleware in w.server.CONTROLLER.middleware:
        middleware.intent_cache.clear()


@case("aux_detector_gets_zero_restrictive_prompts", 200, {"local_main": 1}, aux=True)
def detector(w):
    _fresh_detector(w)
    return _status(w.post("default", data_class="private"))


@case("aux_router_that_embeds_is_skipped_the_weak_side_serves", 200, {"local_small": 1}, aux=True)
def router(w):
    w.admit("private", "private_emb")
    return _status(w.post("private_emb", data_class="private"))


@case("aux_semantic_cache_never_embeds_a_restricted_prompt", 200, {"local_main": 1}, aux=True)
def semantic_cache(w):
    cache = w.server.CONTROLLER.cache
    cache.config.enabled = True
    cache.config.semantic_enabled = True
    cache.config.db_path = str(Path(tempfile.mkdtemp()) / "aux_cache.db")
    cache._init_db()
    return _status(w.post("private", data_class="private"))


# -- controls: the class that may reach the cloud ------------------------------------------


@case(
    "aux_control_detector_is_called_for_a_class_that_may_reach_it",
    200,
    {"local_main": 1},
    reaches_off_host=True,
    aux=True,
)
def control_detector(w):
    _fresh_detector(w)
    return _status(w.post("gate_sibling", data_class="leaky"))


@case(
    "aux_control_router_embeds_for_a_class_that_may_reach_it",
    reaches_off_host=True,
    total=0,
    aux=True,
)
def control_router(w):
    w.admit("leaky", "private_emb")
    return _status(w.post("private_emb", data_class="leaky"))
