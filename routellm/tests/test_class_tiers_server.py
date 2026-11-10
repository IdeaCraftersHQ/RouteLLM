"""Class to allowed tiers through the real app: the 403 shape and the startup refusal."""

import pytest

from routellm.tests.test_request_class import KEY_B, _run

TIERS = {
    "default": {"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"},
    "other": {"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"},
}


@pytest.mark.slow
def test_a_tier_outside_the_class_is_403_in_the_api_error_shape(tmp_path):
    out = _run(
        tmp_path,
        f"""
h = H("{KEY_B}", **{{"X-Data-Class": "open"}})
bad = client.post("/v1/chat/completions", headers=h, json={{**BODY, "model": "other"}})
n0 = hits()
ok = client.post("/v1/chat/completions", headers=h, json=BODY)
emit({{"bad": [bad.status_code, bad.json()], "ok": ok.status_code, "before": n0}})
""",
        config_extra={"tiers": TIERS},
    )
    status, body = out["bad"]
    assert status == 403
    assert body["object"] == "error"
    assert "'open'" in body["message"] and "'other'" in body["message"]
    assert out["before"] == 0
    assert out["ok"] == 200


@pytest.mark.slow
def test_a_leaf_outside_the_class_reach_stops_startup(tmp_path):
    result = _run(
        tmp_path,
        "emit({})",
        policy_extra={
            "classes": {"open": {"treatment": "block", "reach": ["vpn"], "tiers": ["default"]}},
            "principals": {"team-b": {"classes": ["open"]}},
        },
        expect_ok=False,
    )
    assert result.returncode != 0
    err = result.stderr
    assert "class 'open'" in err and "tier 'default'" in err
    assert "leaf 'a'" in err and "'local'" in err
