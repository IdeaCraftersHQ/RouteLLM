# US-0107: Config discovery

**Status:** shipped
**Persona:** operator
**Slug:** RLM-CONFIG-DISCOVERY

## User goal

As an operator, I want RouteLLM to find its configuration the way every
other tool on my machine does — a global file under XDG, a project-local
override next to the code, an explicit file for a one-off run — so I do
not have to pass `--config` on every command, and so the real instance
config can live outside the repo.

## Precedence chain

Lowest first; every layer that exists is merged onto the ones below it.

| # | Layer | Location | Missing file |
|---|-------|----------|--------------|
| 1 | defaults | built in, no file | — |
| 2 | system | `/etc/routellm/config.yaml` | skipped |
| 3 | user | `$XDG_CONFIG_HOME/routellm/config.yaml` (`~/.config` when unset) | skipped |
| 4 | project | `./.routellm.yaml` or `./routellm.yaml`, in the CWD or the nearest ancestor below `$HOME` | skipped |
| 5 | env | `ROUTELLM_CONFIG=<path>` | **error** |
| 6 | flag | `--config <path>` | **error** |

Layers 2-4 are discovered, so their absence is silence. Layers 5 and 6
were named by the operator, so a file that is not there is an error that
names the variable or the flag.

## Merge rules

- Mappings merge recursively, so `endpoints:`, `tiers:`, `intents:` and
  `areas:` merge by key across layers: a user file can define ten
  endpoints and a project file add one or retag another.
- Scalars and lists replace wholesale. A project's `tags:` list replaces
  the user's for that endpoint; it never concatenates.
- A key whose value is `null` in a higher layer **deletes** it. That is
  the only way to drop a globally defined endpoint locally:

  ```yaml
  # ~/.config/routellm/config.yaml
  endpoints:
    cloud_strong: {model: gpt-4o}
    local_fast:   {model: ollama_chat/qwen3:8b}

  # ./.routellm.yaml — this repo may not talk to the cloud
  endpoints:
    cloud_strong: null
  ```

- A relative path inside a config file resolves against **that file**,
  not the CWD, so a user-layer `prompt_file: prompts/router.yaml` means
  `~/.config/routellm/prompts/router.yaml` wherever the server is run.

## Acceptance criteria

1. The server started with no `--config` routes against the tiers of the
   discovered config.
2. `--config` still wins: it is merged last, over everything discovered.
3. `python -m routellm.config path` prints the highest-precedence file
   that exists, and exits 1 when none does.
4. `python -m routellm.config paths` lists every searched location,
   lowest first, each marked `[used]` or `[absent]`.
5. `python -m routellm.config show` prints the effective merged config
   with a comment per top-level key naming the file that set it.
6. `paths` and `show` both take `--format json`.
7. `python -m routellm.pairing` with no `--config` explains the
   discovered config.
8. A project-local `.routellm.yaml` is gitignored, so it is never
   committed by accident.

## E2E Tests

- `routellm/tests/test_openai_server_models.py::test_server_discovers_the_user_config_without_a_flag`
- `routellm/tests/test_openai_server_models.py::test_flag_still_overrides_discovery`
- `routellm/tests/test_config_cli.py::test_path_prints_the_winner`
- `routellm/tests/test_config_cli.py::test_paths_marks_used_and_absent`
- `routellm/tests/test_config_cli.py::test_show_carries_origin_comments`
- `routellm/tests/test_config_cli.py::test_show_json_is_the_merged_dict`
- `routellm/tests/test_config_cli.py::test_no_config_anywhere_path_exits_1`
- `routellm/tests/test_config_cli.py::test_flag_wins_over_the_discovered_chain`
- `routellm/tests/test_config_cli.py::test_env_var_joins_the_chain`
- `routellm/tests/test_pairing.py::test_explain_without_a_flag_reads_the_discovered_config`
- `routellm/tests/test_config_discovery.py::test_chain_order_lowest_first`
- `routellm/tests/test_config_discovery.py::test_null_deletes_a_key_from_a_lower_layer`
- `routellm/tests/test_config_discovery.py::test_relative_prompt_file_resolves_against_its_own_file`

## Non-goals

- Hot reload.
- `-c key=value` inline overrides; the null-delete rule covers the
  common need for a nested config.
- Per-request config.
