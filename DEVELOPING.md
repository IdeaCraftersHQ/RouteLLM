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
