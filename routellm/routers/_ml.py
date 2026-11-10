"""Guard for the routers that run a model.

torch, transformers, datasets and huggingface_hub are the `ml` extra.
Importing `routellm.routers.routers` must not need them, so each router
that does calls `require_ml` first and gets one clear error naming the
extra, instead of a bare `ModuleNotFoundError` from deep inside.
"""

import importlib

ML_INSTALL_HINT = "pip install 'routellm[ml]'"


class MLExtraMissing(ImportError):
    """A router needs the optional `ml` extra, which is not installed."""


def require_ml(router: str, *modules: str) -> None:
    """Import `modules` or raise `MLExtraMissing` naming the extra.

    Parameters
    ----------
    router : str
        CLI name of the router asking, for the message.
    *modules : str
        Top-level modules the router imports when it is built.

    Raises
    ------
    MLExtraMissing
        If any of `modules` cannot be imported.
    """
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise MLExtraMissing(
                f"The {router!r} router needs {name!r}, which is part of the optional "
                f"ML stack. Install it with: {ML_INSTALL_HINT}"
            ) from exc
