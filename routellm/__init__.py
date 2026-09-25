"""RouteLLM: serve and evaluate LLM routers.

Importing this package must not touch the network. `import litellm`
fetches a model cost map over HTTPS at import time, so anything that
reaches litellm -- the server, the CLIs, a library consumer -- pays a
request and a timeout before it does any work, and fails outright where
there is no network.

litellm ships the same map inside the package and its own CLIs default
to it. Nothing here reads pricing: litellm is used for `completion`,
`token_counter`, `get_llm_provider` and the client session, none of
which consult the cost map.

`setdefault`, not assignment: an operator who wants the freshly fetched
map sets the variable themselves and this does not override them. The
bundled copy carries fewer of the newest models, which matters only to
a caller that prices requests.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
