"""Layered config discovery: XDG paths, a project walk-up, and a deep merge.

The server reads its config from a precedence chain rather than from a
single file, the convention every kit-built CLI follows. Lowest
precedence first, later layers overriding earlier ones::

    1. built-in defaults       (no file)
    2. system                  /etc/routellm/config.yaml
    3. user                    $XDG_CONFIG_HOME/routellm/config.yaml
    4. project                 ./.routellm.yaml or ./routellm.yaml, in the
                               CWD or the first ancestor that has one
    5. env                     ROUTELLM_CONFIG=<path>
    6. flag                    --config <path>

Layers 2-4 are optional and silently skipped when absent. The env and
flag layers are explicit: naming a file that does not exist is an error,
because the operator asked for that file by name.

Merging is recursive for mappings, so `endpoints:`, `tiers:`, `intents:`
and `areas:` merge by key across layers — a user file can define ten
endpoints and a project file add one or retag another. Scalars and lists
replace wholesale. A `null` value in a higher layer *deletes* the key,
which is the only way to drop a globally defined endpoint locally::

    # $XDG_CONFIG_HOME/routellm/config.yaml
    endpoints:
      cloud_strong: {model: gpt-4}
      cloud_weak:   {model: gpt-4o-mini}

    # ./.routellm.yaml — this repo may not talk to the cloud
    endpoints:
      cloud_strong: null

Relative paths inside a config file (`prompt_file`, `quality_from`)
resolve against the file that set them, never the CWD; `origins` records
which layer last set each leaf key so `resolve_path` can do that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

TOOL = "routellm"
PROJECT_MARKERS = [".routellm.yaml", "routellm.yaml"]
SYSTEM_PATH = "/etc/routellm/config.yaml"
ENV_VAR = "ROUTELLM_CONFIG"
FLAG = "--config"


class ConfigError(Exception):
    """Raised when a config file on the chain cannot be parsed."""


@dataclass(frozen=True)
class ResolvedPath:
    """One entry in the config resolution chain.

    Attributes
    ----------
    path : Path
        The file that would be read for this layer.
    source : str
        One of "system", "user", "project", "env", "flag".
    exists : bool
        Whether a regular file sits at `path` right now, by stat.
    """

    path: Path
    source: str
    exists: bool


@dataclass
class LoadedConfig:
    """The merged config plus the provenance needed to explain it.

    Attributes
    ----------
    data : dict
        The merged mapping, defaults first and the flag layer last.
    layers : list[ResolvedPath]
        The chain entries that existed and were merged, in merge order.
    origins : dict[str, Path]
        Dotted leaf key (``endpoints.cloud_strong.api_base``) mapped to
        the file of the layer that last set it. Built-in defaults have
        no file and so no origin.
    chain : list[ResolvedPath]
        Every location searched, existing or not, in the same order.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    layers: List[ResolvedPath] = field(default_factory=list)
    origins: Dict[str, Path] = field(default_factory=dict)
    chain: List[ResolvedPath] = field(default_factory=list)

    def resolve_path(self, key: str, value: Any) -> Path:
        """Resolve a path-valued config entry against the file that set it.

        Parameters
        ----------
        key : str
            Dotted key the value came from, as it appears in `origins`.
        value : Any
            The value to resolve; coerced to `Path`.

        Returns
        -------
        Path
            `value` unchanged when it is already absolute or when `key`
            has no recorded origin; otherwise `value` joined onto the
            directory of the file that set it.
        """
        path = Path(value)
        if path.is_absolute():
            return path
        origin = self.origins.get(key)
        if origin is None:
            return path
        return origin.parent / path


def _user_config_dir() -> Path:
    """Return `$XDG_CONFIG_HOME`, falling back to `~/.config`.

    A relative `XDG_CONFIG_HOME` is invalid per the XDG basedir spec and
    is treated as unset.
    """
    raw = os.environ.get("XDG_CONFIG_HOME", "")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
    return Path.home() / ".config"


def _project_file(cwd: Path) -> Optional[Path]:
    """Find the nearest project marker at or above `cwd`.

    Both markers are probed per directory, in `PROJECT_MARKERS` order.
    The walk stops at the filesystem root and at `$HOME`, which is never
    itself read as a project directory — the user layer covers it.
    `$HOME` is resolved before comparing: on macOS it commonly points
    through a symlink, and an unresolved boundary never matches, which
    would read the user's own `$HOME/.routellm.yaml` as a project file.
    """
    home = Path.home().resolve()
    directory = cwd
    while True:
        if directory == home:
            return None
        for marker in PROJECT_MARKERS:
            candidate = directory / marker
            if candidate.is_file():
                return candidate
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent


def config_paths(
    cwd: Optional[os.PathLike | str] = None,
    explicit: Optional[os.PathLike | str] = None,
) -> List[ResolvedPath]:
    """Return the full config chain, lowest precedence first.

    Parameters
    ----------
    cwd : path-like, optional
        Directory the project walk-up starts from. Defaults to the
        process CWD.
    explicit : path-like, optional
        The `--config` flag value. Wins over `ROUTELLM_CONFIG` when both
        are set; both are appended after the project layer.

    Returns
    -------
    list[ResolvedPath]
        System, user, and (when a marker was found) project entries,
        then env and flag entries when those are set. `exists` is set by
        stat, so absent layers are still reported.
    """
    start = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()

    chain = [
        _entry(Path(SYSTEM_PATH), "system"),
        _entry(_user_config_dir() / TOOL / "config.yaml", "user"),
    ]

    project = _project_file(start)
    if project is not None:
        chain.append(_entry(project, "project"))

    env_value = os.environ.get(ENV_VAR)
    if env_value:
        chain.append(_entry(Path(env_value), "env"))
    if explicit:
        chain.append(_entry(Path(explicit), "flag"))

    return chain


def _entry(path: Path, source: str) -> ResolvedPath:
    return ResolvedPath(path, source, path.is_file())


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested mappings.

    Parameters
    ----------
    base : Mapping
        The lower-precedence mapping. Not mutated.
    override : Mapping
        The higher-precedence mapping.

    Returns
    -------
    dict
        A new mapping. Nested mappings merge by key; scalars and lists
        replace wholesale; a key whose value in `override` is None is
        deleted from the result rather than set to None.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
            continue
        current = merged.get(key)
        if isinstance(value, Mapping):
            # Recurse even when no lower layer set this parent, so a
            # null leaf under a brand-new parent is dropped on the way
            # in rather than surviving as an explicit None.
            base_side = current if isinstance(current, Mapping) else {}
            merged[key] = deep_merge(base_side, value)
        else:
            merged[key] = value
    return merged


def _record_origins(
    origins: Dict[str, Path],
    layer: Mapping[str, Any],
    path: Path,
    prefix: str = "",
) -> None:
    """Stamp `path` onto every leaf key `layer` sets, and unstamp deletions."""
    for key, value in layer.items():
        dotted = f"{prefix}{key}"
        if value is None:
            for stale in [k for k in origins if k == dotted or k.startswith(dotted + ".")]:
                del origins[stale]
            continue
        if isinstance(value, Mapping):
            _record_origins(origins, value, path, prefix=f"{dotted}.")
        else:
            origins[dotted] = path


def _read(entry: ResolvedPath) -> Dict[str, Any]:
    """Parse one layer's YAML, naming the file and the layer on failure."""
    try:
        parsed = yaml.safe_load(entry.path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"{entry.path}: invalid YAML in the {entry.source} config layer: {exc}"
        ) from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise ConfigError(
            f"{entry.path}: the {entry.source} config layer must be a mapping, "
            f"got {type(parsed).__name__}"
        )
    return dict(parsed)


def _defaults() -> Dict[str, Any]:
    """Return the built-in defaults layer: today's router configs, nothing else."""
    from routellm.controller import GPT_4_AUGMENTED_CONFIG

    return {name: dict(section) for name, section in GPT_4_AUGMENTED_CONFIG.items()}


def load_config(
    cwd: Optional[os.PathLike | str] = None,
    explicit: Optional[os.PathLike | str] = None,
) -> LoadedConfig:
    """Load and merge every config layer that applies.

    Parameters
    ----------
    cwd : path-like, optional
        Directory the project walk-up starts from. Defaults to the CWD.
    explicit : path-like, optional
        The `--config` flag value.

    Returns
    -------
    LoadedConfig
        The merged data, the layers that existed, per-key origins, and
        the full searched chain.

    Raises
    ------
    FileNotFoundError
        When `ROUTELLM_CONFIG` or `--config` names a file that is not
        there; the message names the variable or the flag.
    ConfigError
        When a layer is not valid YAML or is not a mapping; the message
        names the file and the layer.
    """
    chain = config_paths(cwd=cwd, explicit=explicit)

    data = _defaults()
    origins: Dict[str, Path] = {}
    layers: List[ResolvedPath] = []

    for entry in chain:
        if not entry.exists:
            if entry.source == "env":
                raise FileNotFoundError(f"{ENV_VAR}={entry.path}: no such config file")
            if entry.source == "flag":
                raise FileNotFoundError(f"{FLAG} {entry.path}: no such config file")
            continue
        parsed = _read(entry)
        data = deep_merge(data, parsed)
        _record_origins(origins, parsed, entry.path)
        layers.append(entry)

    return LoadedConfig(data=data, layers=layers, origins=origins, chain=chain)


def explain(loaded: LoadedConfig) -> str:
    """Render the chain and the effective config, annotated with origins.

    Every searched location is listed with `[used]` or `[absent]`, then a
    blank line, then the merged config as YAML with each top-level key
    followed by a comment naming the file that last set it.

    Parameters
    ----------
    loaded : LoadedConfig
        The result of `load_config`.

    Returns
    -------
    str
        The rendered explanation, newline-terminated.
    """
    lines = _chain_lines(loaded.chain)
    lines.append("")

    dumped = yaml.safe_dump(loaded.data, sort_keys=False)
    for line in dumped.splitlines():
        origin = _top_level_origin(loaded, line)
        lines.append(f"{line}  # from {origin}" if origin else line)

    return "\n".join(lines) + "\n"


def _top_level_origin(loaded: LoadedConfig, line: str) -> Optional[Path]:
    """Return the origin for a dumped top-level key line, or None.

    Only unindented `key:` / `key: value` lines carry a comment; a
    mapping-valued key takes the origin of the first leaf under it, so
    every top-level key names a file.
    """
    if not line or line[0].isspace() or ":" not in line:
        return None
    key = line.split(":", 1)[0]
    if key in loaded.origins:
        return loaded.origins[key]
    prefix = key + "."
    for dotted, path in loaded.origins.items():
        if dotted.startswith(prefix):
            return path
    return None


# ---------------------------------------------------------------------------
# Inspection CLI
# ---------------------------------------------------------------------------


def winning_path(chain: List[ResolvedPath]) -> Optional[ResolvedPath]:
    """Return the highest-precedence entry that exists, or None.

    Parameters
    ----------
    chain : list[ResolvedPath]
        The chain as `config_paths` returns it, lowest precedence first.

    Returns
    -------
    ResolvedPath or None
        The last entry whose file is there; None when none of them is.
    """
    for entry in reversed(chain):
        if entry.exists:
            return entry
    return None


#: Stands in for the project layer when the walk-up found no marker.
#: `config_paths` omits the entry entirely in that case — there is no
#: path to name — but a chain listing that simply skipped the layer
#: would read as though it were never searched.
NO_PROJECT_LINE = "[absent] project (no .routellm.yaml or routellm.yaml found)"


def _chain_lines(chain: List[ResolvedPath]) -> List[str]:
    """Render one `[used]`/`[absent]` line per searched location.

    A chain carrying no project entry gets `NO_PROJECT_LINE` in its
    place, so every surface that prints the chain reports the same six
    layers whether or not a marker was found.
    """
    lines = [
        f"{'[used]  ' if entry.exists else '[absent]'} {entry.source:<7} {entry.path}"
        for entry in chain
    ]
    if not any(entry.source == "project" for entry in chain):
        # After user, before the explicit layers: its place in the
        # precedence order, not the end of the list.
        after_user = sum(1 for entry in chain if entry.source in ("system", "user"))
        lines.insert(after_user, NO_PROJECT_LINE)
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    """Inspect the config chain: `path`, `paths`, or `show`.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        0 on success; 1 when `path` finds nothing on the chain or when a
        named file is missing or unparseable.
    """
    import argparse
    import json
    import sys

    # `--config` is declared on a shared parent so it reads the same
    # before or after the subcommand; argparse otherwise rejects it in
    # the trailing position, which is where an operator naturally types
    # it.
    # SUPPRESS, not `default=None` or `set_defaults(config=None)`: the
    # parent parser and every subparser each own a copy of this action
    # (one per `parents=[common]`), and argparse applies the subparser's
    # default *after* the parent's, so a real default here would always
    # clobber a leading `--config` with None. SUPPRESS means the action
    # sets nothing when absent, so whichever level the flag was actually
    # given at is the one that lands on the namespace.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Explicit config file; the highest-precedence layer.",
    )

    parser = argparse.ArgumentParser(
        prog="python -m routellm.config",
        description="Show where the RouteLLM config comes from.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "path",
        parents=[common],
        help="Print the highest-precedence config file that exists.",
    )
    for name, helptext in (
        ("paths", "Print every searched location, lowest precedence first."),
        ("show", "Print the effective merged config."),
    ):
        sub = subparsers.add_parser(name, parents=[common], help=helptext)
        sub.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args(argv)

    try:
        if args.command == "paths":
            chain = config_paths(explicit=getattr(args, "config", None))
            if args.format == "json":
                print(
                    json.dumps(
                        [
                            {
                                "source": entry.source,
                                "path": str(entry.path),
                                "exists": entry.exists,
                            }
                            for entry in chain
                        ],
                        indent=2,
                    )
                )
            else:
                print("\n".join(_chain_lines(chain)))
            return 0

        if args.command == "path":
            explicit = getattr(args, "config", None)
            # config_paths()+winning_path() never stat the explicit
            # layer for validity, only for precedence: a missing
            # --config/ROUTELLM_CONFIG file just falls through to
            # whatever exists lower on the chain. load_config() already
            # raises FileNotFoundError with the right message for that
            # case (caught below); call it here for the side effect of
            # that check before ever consulting the chain for a winner.
            load_config(explicit=explicit)
            winner = winning_path(config_paths(explicit=explicit))
            if winner is None:
                print(
                    "no routellm config file on the chain; run `paths` to see where it looked",
                    file=sys.stderr,
                )
                return 1
            print(winner.path)
            return 0

        loaded = load_config(explicit=getattr(args, "config", None))
        if args.format == "json":
            print(json.dumps(loaded.data, indent=2, default=str))
        else:
            print(explain(loaded), end="")
        return 0
    except (FileNotFoundError, ConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
