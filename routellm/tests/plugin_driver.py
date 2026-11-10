"""Run in a subprocess whose sys.path holds the installed test plugin.

Prints one JSON object: what each plugin scenario did. Kept out of the
parent process so `importlib.metadata` finds the plugin the way a user's
environment would, through its installed dist-info.
"""

import asyncio
import json

from routellm.config import ConfigError
from routellm.decision import DecisionRefused, build_decision
from routellm.endpoints import EndpointRegistry
from routellm.transform import load_transforms

CLASSES = {"general": "nothing sensitive", "private": "must stay local"}
INTENTS = {"chat": "small talk"}


def registry():
    return EndpointRegistry.from_config(
        {
            "endpoints": {
                "decider": {"model": "ollama_chat/tiny", "api_base": "http://127.0.0.1:1"},
                "big": {"model": "gpt-4o"},
                "small": {"model": "ollama_chat/s", "api_base": "http://127.0.0.1:1"},
            },
            "tiers": {
                "default": {"router": "random", "threshold": 0.0, "strong": "big", "weak": "small"}
            },
        }
    )


def section(**over):
    return {
        "endpoint": "decider",
        "data_classes": CLASSES,
        "intents": INTENTS,
        "min_confidence": 0.5,
        "timeout_ms": 500,
        **over,
    }


def error_of(call):
    try:
        call()
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return None


def classify(over, text="hello"):
    classifier = build_decision({"decision": section(**over)}, registry())
    classifier.bind(None)
    decision = asyncio.run(classifier.classify([{"role": "user", "content": text}]))
    return decision.as_dict()


def main():
    out = {}
    out["provider_ok"] = classify({"provider": "rllm_ok"})
    out["provider_bad_labels"] = classify({"provider": "rllm_bad_labels"})
    try:
        classify({"provider": "rllm_bad_labels", "on_error": "refuse"})
        out["provider_bad_labels_refuse"] = None
    except DecisionRefused as exc:
        out["provider_bad_labels_refuse"] = str(exc)
    out["provider_unknown"] = error_of(lambda: classify({"provider": "rllm_nope"}))
    out["provider_boom"] = error_of(lambda: classify({"provider": "rllm_boom"}))

    rec = {
        "name": "wordy",
        "type": "rllm_word",
        "class": "private",
        "options": {"word": "swordfish"},
    }
    out["recognizer_hit"] = classify({"provider": "rllm_ok", "recognizers": [rec]}, "the swordfish")
    out["recognizer_miss"] = classify({"provider": "rllm_ok", "recognizers": [rec]}, "plain")
    unknown = {**rec, "type": "rllm_nope"}
    out["recognizer_unknown"] = error_of(
        lambda: build_decision(
            {"decision": section(provider="rllm_ok", recognizers=[unknown])}, registry()
        )
    )
    boom = {**rec, "type": "rllm_boom"}
    out["recognizer_boom"] = error_of(
        lambda: build_decision(
            {"decision": section(provider="rllm_ok", recognizers=[boom])}, registry()
        )
    )

    built = load_transforms([{"use": "rllm_tag", "options": {"tag": "x"}}])
    out["transform_ok"] = {"type": type(built[0]).__name__, "tag": built[0].tag}
    for key, use in (("transform_unknown", "rllm_nope"), ("transform_boom", "rllm_boom")):
        try:
            load_transforms([use])
            out[key] = None
        except ConfigError as exc:
            out[key] = {"type": "ConfigError", "message": str(exc)}
        except Exception as exc:
            out[key] = {"type": type(exc).__name__, "message": str(exc)}
    print(json.dumps(out))


main()
