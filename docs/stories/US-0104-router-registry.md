# US-0104: Router registry and entry-point contract

**Status:** shipped
**Persona:** API consumer
**Slug:** RLM-NEW-ROUTER-REGISTRY

## User goal

As a developer, I want to add a router to RouteLLM without editing
core source, so a router I maintain in my own package can register
itself and show up in `--routers` for the server, calibration, and
evals.

## Context

Routers were registered by direct mutation of a `ROUTER_CLS` dict
in `routers.py`, which meant any new router — in-tree or external —
required a core-file edit. `routellm/routers/registry.py` replaces
that with `register_router`/`get_router_class`/`router_names`, plus
`discover_routers`, which reads the `routellm.routers` entry-point
group and registers what installed packages publish there. A
plugin that fails to import is skipped with a warning and recorded
in `discovery_failures`, so a broken extension never breaks import
or the CLI, and its name still resolves to a useful error.

## Acceptance criteria

- [x] Extension router discoverable via entry point with no core
      edit
- [x] `--routers` choices include it
- [x] Broken plugin does not break the CLI and is reported in the
      unknown-name error
- [x] Existing `ROUTER_CLS` readers unchanged

## Implementation notes

- Module: `routellm/routers/registry.py` — `register_router`,
  `get_router_class`, `router_names`, `name_for`,
  `discover_routers`, `discovery_failures`, `reset_registry`,
  `ROUTER_CLS`
- `routellm/routers/routers.py` registers the five built-ins
  (`random`, `mf`, `causal_llm`, `bert`, `sw_ranking`) and runs
  `discover_routers()` at import

## E2E tests

- `routellm/tests/test_router_registry.py::test_register_get_names_round_trip`
- `routellm/tests/test_router_registry.py::test_router_names_sorted`
- `routellm/tests/test_router_registry.py::test_duplicate_name_raises`
- `routellm/tests/test_router_registry.py::test_replace_overrides`
- `routellm/tests/test_router_registry.py::test_empty_name_raises`
- `routellm/tests/test_router_registry.py::test_non_class_raises`
- `routellm/tests/test_router_registry.py::test_decorator_form_registers`
- `routellm/tests/test_router_registry.py::test_name_for_unregistered_raises`
- `routellm/tests/test_router_registry.py::test_unknown_name_error_lists_registered`
- `routellm/tests/test_router_registry.py::test_discover_registers_good_and_records_broken`
- `routellm/tests/test_router_registry.py::test_discover_twice_is_idempotent`
- `routellm/tests/test_router_registry.py::test_unknown_name_error_lists_discovery_failure`
- `routellm/tests/test_router_registry.py::test_router_cls_is_backing_dict`
- `routellm/tests/test_router_registry.py::test_str_of_late_registered_instance`

## Related

- US-0001 — strong/weak routing via cost threshold (readers of
  `ROUTER_CLS` this story keeps unchanged)
- US-0103 — TypeSafe Jev routing (`jev` registers through this
  same registry)
