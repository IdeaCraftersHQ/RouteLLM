"""Report how a config's free-model endpoints compare with a live roster.

Free model lists churn: ids appear, disappear, and some carry an expiry.
This reads a provider's public models listing (an OpenAI-style
`{"data": [{"id": ..., "expiration_date": ...}]}` document) and diffs
the ids it marks free against the configured endpoints. It reports
only; it never edits the config.

    python -m routellm.roster --url https://example.test/api/v1/models \\
        --prefix openrouter/ [--suffix :free] [--config path]

An endpoint is compared when its `model` starts with `--prefix` and the
remainder ends with `--suffix`. The listing is fetched with a plain GET
and no credentials.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from routellm.config import ConfigError, load_config
from routellm.endpoints import EndpointRegistry


@dataclass
class RosterDiff:
    """The three differences between a config and a roster."""

    new: list[str] = field(default_factory=list)
    gone: list[tuple[str, str]] = field(default_factory=list)
    expiring: list[tuple[str, str]] = field(default_factory=list)


def fetch_roster(url: str, suffix: str, timeout: float = 15.0) -> dict[str, str | None]:
    """Return `{id: expiration_date or None}` for every listed id ending in `suffix`."""
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{url} did not return a {{'data': [...]}} document")
    return {
        entry["id"]: entry.get("expiration_date") or None
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and entry["id"].endswith(suffix)
    }


def diff_roster(
    registry: EndpointRegistry, roster: dict[str, str | None], prefix: str, suffix: str
) -> RosterDiff:
    """Compare the registry's matching endpoints with `roster`."""
    configured: dict[str, str] = {}
    for name in registry.names():
        model = registry.get(name).model
        if model.startswith(prefix) and model[len(prefix) :].endswith(suffix):
            configured[model[len(prefix) :]] = name

    diff = RosterDiff()
    diff.new = sorted(set(roster) - set(configured))
    diff.gone = sorted(
        (name, model_id) for model_id, name in configured.items() if model_id not in roster
    )
    diff.expiring = sorted(
        (model_id, roster[model_id]) for model_id in configured if roster.get(model_id)
    )
    return diff


def render(diff: RosterDiff, listed: int, configured: int) -> str:
    """Render a diff as the plain-text report."""
    lines = [f"Roster: {listed} listed, {configured} configured."]
    lines.append(f"Listed but not in the config ({len(diff.new)}):")
    lines += [f"  {model_id}" for model_id in diff.new] or ["  (none)"]
    lines.append(f"In the config but no longer listed ({len(diff.gone)}):")
    lines += [f"  {name} ({model_id})" for name, model_id in diff.gone] or ["  (none)"]
    lines.append(f"Configured and expiring ({len(diff.expiring)}):")
    lines += [f"  {model_id} expires {when}" for model_id, when in diff.expiring] or ["  (none)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the report; 0 on success, 1 when the config or roster cannot be read."""
    parser = argparse.ArgumentParser(
        prog="python -m routellm.roster",
        description="Diff a config's free-model endpoints against a provider's live roster.",
    )
    parser.add_argument("--url", required=True, help="Public models listing to read.")
    parser.add_argument("--prefix", default="", help="litellm prefix of the configured models.")
    parser.add_argument("--suffix", default=":free", help="Id suffix marking a free model.")
    parser.add_argument("--config", default=None, help="Explicit config file.")
    args = parser.parse_args(argv)

    try:
        config = load_config(explicit=args.config).data
        registry = EndpointRegistry.from_config(config, config_path=args.config)
        roster = fetch_roster(args.url, args.suffix)
    except (ConfigError, ValueError, OSError, urllib.error.URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    diff = diff_roster(registry, roster, args.prefix, args.suffix)
    configured = sum(
        1
        for name in registry.names()
        if registry.get(name).model.startswith(args.prefix)
        and registry.get(name).model[len(args.prefix) :].endswith(args.suffix)
    )
    print(render(diff, len(roster), configured))
    return 0


if __name__ == "__main__":
    sys.exit(main())
