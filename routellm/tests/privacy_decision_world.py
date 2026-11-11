"""The decision world: the policy (or route) world plus a decision step.

The classifier's model is a loopback fake leaf, `decider`, tagged local
so it passes every class (the step reads a request before its class
exists). Classes run general < internal < private, least to most
restrictive; the route world has no `internal`. Two recognizers exist:
`private_marker` settles `private`, `internal_marker` settles
`internal`, which is below the strictest class and so is the one that
shows a recognizer hit skipping the model.
"""

import json

from routellm.tests.privacy_fixture import SELECTOR, class_policy_files, route_config

PRIVATE_MARKER = "marker-private-7f3"
INTERNAL_MARKER = "marker-internal-7f3"

#: A selector tier must not pick the decider, which is local too.
DECIDER_SELECTOR = f"{SELECTOR} !tag:decider"


def decision_section(route: bool) -> dict:
    classes = {"general": "nothing sensitive"}
    if not route:
        classes["internal"] = "company-internal material"
    classes["private"] = "personal or secret data"
    return {
        "provider": "llm",
        "endpoint": "decider",
        "require_tags": ["local"],
        "data_classes": classes,
        "timeout_ms": 400,
        "min_confidence": 0.5,
        "on_error": "most_restrictive",
        "recognizers": [
            {
                "name": "private_marker",
                "type": "markers",
                "class": "private",
                "markers": [PRIVATE_MARKER],
            },
            *(
                []
                if route
                else [
                    {
                        "name": "internal_marker",
                        "type": "markers",
                        "class": "internal",
                        "markers": [INTERNAL_MARKER],
                    }
                ]
            ),
        ],
        "sticky": {"enabled": True, "ttl_s": 3600},
    }


def decision_config(tmp_path, route: bool = False) -> dict:
    """Return the extra config of the decision world (direct, or with policy routing)."""
    base = class_policy_files(tmp_path, route)
    if route:
        extra = {**base, **route_config()}
    else:
        policy = json.loads((tmp_path / "policy.json").read_text())
        policy["classes"]["internal"] = {
            "treatment": "raw",
            "reach": ["local"],
            "tiers": ["private", "private_audio", "private_selector", "default"],
        }
        policy["principals"]["suite"]["classes"].append("internal")
        # An undeclared request is held to `general` until a source says more.
        policy["default_class"] = "general"
        (tmp_path / "policy.json").write_text(json.dumps(policy))
        extra = base
    tiers = {**extra["tiers"]}
    pair = {"router": "random", "threshold": 0.0}
    tiers["private_selector"] = {
        **pair,
        "strong": {"select": DECIDER_SELECTOR, "order": "quality_desc"},
        "weak": {"select": DECIDER_SELECTOR, "order": "quality_asc"},
    }
    return {**extra, "tiers": tiers, "decider": True, "decision": decision_section(route)}
