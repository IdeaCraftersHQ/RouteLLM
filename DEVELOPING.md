# Developing RouteLLM

## Recommended: devcontainer

Open this repo in a devcontainer (VS Code: `Reopen in Container`, or
GitHub Codespaces). Preinstalled:

- `uv` — Python package/venv manager
- `python 3.12` — matches `requires-python = ">=3.10"`
- `pre-commit` — git hook runner
- `gh` — GitHub CLI
- `make` — task runner

After opening, a `.venv` is created and the `dev` and `serve` extras are
installed automatically. Run the test command printed at the end of
`postCreateCommand`:

```sh
PYTHONPATH=. python -m pytest routellm/tests -q
```

## Local setup without the container

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,serve]"
```

`PYTHONPATH` must include the repo root (the package is not installed in
`src/` layout):

```sh
PYTHONPATH=. python -m pytest routellm/tests -q
```

The server (`routellm/openai_server.py` et al.) listens on port 6060.

## Your local instance config

The server discovers its config; it does not read one from the repo.
Put your own instance config in the user layer:

```sh
mkdir -p ~/.config/routellm
cp config.example.yaml ~/.config/routellm/config.yaml
```

Then `python -m routellm.openai_server` needs no `--config` at all.
`config.example.yaml` and `examples/multitier.yaml` are samples to copy
from, never configuration.

To override a few keys for this checkout only, write `./.routellm.yaml`
— it is gitignored, merges over the user layer, and `null` deletes a key
the user layer set. Check what is in effect with:

```sh
python -m routellm.config paths   # every searched location, used or absent
python -m routellm.config show    # the merged result, each key's origin
```

## Extensions

Extensions live under `extensions/<name>/` with their own `pyproject.toml`.
Install one into the active environment with:

```sh
pip install -e extensions/<name>
```

## Follow-ups

The playbook's target toolchain is `mise.toml` + a `Makefile` with
`build`/`test`/`lint`/`clean`/`check` targets; neither exists in this repo
yet.
