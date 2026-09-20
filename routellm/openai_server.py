"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import logging
import os
import time
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import uvicorn
import yaml
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from routellm.controller import Controller, RoutingError
from routellm.endpoints import Endpoint, EndpointRegistry, Tier
from routellm.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONTROLLER = None

#: `created` for every listed model: the moment this process came up.
#: The listing is derived from the config, so it never changes while the
#: process lives.
SERVER_START = int(time.time())

#: Endpoint names the implicit `default` tier is built under. Hyphens are
#: reserved for the model-name grammar, so these carry none.
IMPLICIT_STRONG = "implicit_strong"
IMPLICIT_WEAK = "implicit_weak"

#: The pair this server has always routed `router-<r>-<thr>` against.
#: Used only when the caller names no pair and the config defines no
#: `default` tier, so the legacy flat form keeps answering. It is never
#: registered as an endpoint and never advertised on `/v1/models`.
LEGACY_STRONG = "gpt-4-1106-preview"
LEGACY_WEAK = "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1"


def legacy_pair() -> tuple[str, str]:
    """Return the historic flat pair, warning that it was assumed.

    Reached only when neither model flag was given and the config
    defines no `default` tier. The warning goes out at WARNING so it is
    visible without `--verbose`: routing against a pair the caller never
    named is a fact they should see.

    Returns
    -------
    tuple[str, str]
        The `(strong, weak)` model names to hand the controller.
    """
    logging.warning(
        "No --strong-model/--weak-model and no 'default' tier in --config; "
        "routing 'router-<router>-<threshold>' against the historic pair "
        "%s / %s. Pass both flags, or define a 'default' tier, to choose "
        "the pair yourself.",
        LEGACY_STRONG,
        LEGACY_WEAK,
    )
    return LEGACY_STRONG, LEGACY_WEAK


def build_registry(file_config: Optional[dict]) -> EndpointRegistry:
    """Build the endpoint registry the server routes against.

    `--config` is the single source for endpoints and tiers. When that
    config carries no `default` tier and both `--strong-model` and
    `--weak-model` are given, the two are wrapped as endpoints and
    joined into an implicit `default` tier running `--routers[0]` at
    `--default-threshold`, so a flat command line still answers a
    request addressed to `default`.

    A config `default` tier wins, and the flags are then left to the
    flat pair the controller keeps. Neither flag and no config
    `default` derives nothing: no tier is invented and none is
    advertised, and the legacy flat form routes against the historic
    pair instead.

    Parameters
    ----------
    file_config : dict, optional
        The loaded YAML config, or None when `--config` was not given.

    Returns
    -------
    EndpointRegistry
        Registry holding the configured endpoints and tiers, plus the
        implicit `default` tier when one was derived.
    """
    registry = EndpointRegistry.from_config(file_config or {})

    if registry.has_tier("default"):
        logging.info("default tier: from --config")
        if args.strong_model or args.weak_model:
            logging.info(
                "--strong-model/--weak-model ignored for the 'default' tier: "
                "the config defines one. They remain the flat pair."
            )
        return registry

    # Argparse has already rejected exactly one of the two, so either
    # both are set or neither is.
    if not args.strong_model:
        return registry

    endpoints = {name: registry.get(name) for name in registry.names()}

    # A flag may name a configured endpoint, which already carries its
    # own base URL and credential; only a raw model name needs wrapping.
    sides = {}
    for side, flag, wrapper in (
        ("strong", args.strong_model, IMPLICIT_STRONG),
        ("weak", args.weak_model, IMPLICIT_WEAK),
    ):
        if flag in endpoints:
            sides[side] = flag
            continue
        endpoints[wrapper] = Endpoint(name=wrapper, model=flag)
        sides[side] = wrapper

    tiers = registry.tiers
    tiers["default"] = Tier(
        name="default",
        router=args.routers[0] if args.routers else None,
        threshold=args.default_threshold,
        strong=sides["strong"],
        weak=sides["weak"],
    )

    logging.info(
        "default tier: implicit, %s/%s at %s over %s",
        args.strong_model,
        args.weak_model,
        args.default_threshold,
        args.routers[0] if args.routers else "<none>",
    )
    return EndpointRegistry(endpoints, tiers)


@asynccontextmanager
async def lifespan(app):
    global CONTROLLER

    gateway = None
    if args.payment_provider == "x402":
        from routellm.payment.x402 import X402Adapter
        key = os.environ.get(args.wallet_key_env or "ROUTELLM_WALLET_KEY", "")
        if key:
            gateway = X402Adapter(private_key=key)

    # `endpoints:` and `tiers:` belong to the registry; the rest of the
    # file stays router config, so they are popped out before the
    # handoff. A file holding nothing else leaves None, which keeps the
    # router defaults.
    file_config = yaml.safe_load(open(args.config, "r")) if args.config else None
    endpoints = build_registry(file_config)
    router_config = dict(file_config or {})
    router_config.pop("endpoints", None)
    router_config.pop("tiers", None)
    router_config = router_config or None

    # Neither flag and no tier to route into leaves the legacy flat form
    # with nothing to pair; fall back to the pair this server has always
    # used rather than refusing requests it used to answer.
    strong_model, weak_model = args.strong_model, args.weak_model
    if not strong_model and not endpoints.has_tier("default"):
        strong_model, weak_model = legacy_pair()

    CONTROLLER = Controller(
        routers=args.routers,
        config=router_config,
        strong_model=strong_model,
        weak_model=weak_model,
        endpoints=endpoints,
        api_base=args.base_url,
        api_key=args.api_key,
        progress_bar=True,
        payment_gateway=gateway,
        default_router=args.routers[0] if args.routers else None,
        default_threshold=args.default_threshold,
    )
    yield
    CONTROLLER = None


app = fastapi.FastAPI(lifespan=lifespan)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = (
        None  # { "type": "json_object" } for json mode
    )
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


async def stream_response(response) -> AsyncGenerator:
    async for chunk in response:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Route one chat completion and return it in OpenAI's shape.

    The model field carries the routing parameters in any of four
    forms: `<tier>:router-<r>-<thr>`, `router-<r>-<thr>`,
    `router-<tier>`, or a bare `<tier>`. A tiered request walks the
    tier tree, one router call per level.

    A non-streamed response carries the decision path under a top-level
    `routellm` key, which OpenAI clients ignore. A streamed response
    carries it in the server logs only, since its chunks follow the SSE
    shape the client parses.
    """
    logging.info(f"Received request: {request}")
    try:
        res = await CONTROLLER.acompletion(
            **request.model_dump(exclude_none=True),
        )
    except RoutingError as e:
        return JSONResponse(
            ErrorResponse(message=str(e)).model_dump(),
            status_code=400,
        )

    logging.info(CONTROLLER.model_counts)

    hidden = getattr(res, "_hidden_params", None)
    path = hidden.get("routellm_path") if isinstance(hidden, dict) else None
    logging.info(f"Routing path: {path}")

    if request.stream:
        return StreamingResponse(
            content=stream_response(res), media_type="text/event-stream"
        )

    body = res.model_dump()
    if path is not None:
        body["routellm"] = {"path": path}
    return JSONResponse(content=body)


@app.get("/v1/models")
async def list_models():
    """List every model name this server answers to.

    Two families of id: one per configured tier, addressable bare or
    as `router-<tier>`, and one `router-<name>-<default_threshold>` per
    loaded router. The legacy flat form carries a threshold, and only a
    concrete one is a usable id, so each router is listed at the
    server's `--default-threshold`; any other threshold still routes.

    Returns
    -------
    JSONResponse
        An OpenAI model list, `data` sorted by id.
    """
    ids = set(CONTROLLER.endpoints.tier_names())
    ids.update(
        f"router-{name}-{CONTROLLER.default_threshold}"
        for name in CONTROLLER.routers
    )

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": SERVER_START,
                    "owned_by": "routellm",
                }
                for model_id in sorted(ids)
            ],
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "online"})


parser = argparse.ArgumentParser(
    description="An OpenAI-compatible API server for LLM routing."
)
parser.add_argument(
    "--verbose",
    action="store_true",
)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--port", type=int, default=6060)
parser.add_argument(
    "--routers",
    nargs="+",
    type=str,
    default=["random"],
    choices=list(ROUTER_CLS.keys()),
)
parser.add_argument(
    "--base-url",
    help="The base URL used for all LLM requests",
    type=str,
    default=None,
)
parser.add_argument(
    "--api-key",
    help="The API key used for all LLM requests",
    type=str,
    default=None,
)
parser.add_argument(
    "--strong-model",
    help=(
        "Endpoint name from the config, or a raw model name. Pass with "
        "--weak-model to derive a 'default' tier when the config defines none."
    ),
    type=str,
    default=None,
)
parser.add_argument(
    "--weak-model",
    help="Endpoint name from the config, or a raw model name. See --strong-model.",
    type=str,
    default=None,
)
parser.add_argument(
    "--default-threshold",
    help="Threshold for a tier level that names none and whose request carries none",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--payment-provider",
    default=None,
    choices=["x402"],
    help="Enable payment gateway (e.g. x402)",
)
parser.add_argument(
    "--wallet-key-env",
    default="ROUTELLM_WALLET_KEY",
    help="Env var holding wallet private key",
)
args = parser.parse_args()

# A flat pair needs both sides. One alone would silently pair with the
# historic default for the other, which is never what the caller meant.
if bool(args.strong_model) != bool(args.weak_model):
    parser.error(
        "--strong-model and --weak-model must be given together: one alone "
        "cannot form a pair. Pass both, or neither and define a 'default' "
        "tier in --config."
    )

if args.verbose:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Launching server with routers:", args.routers)
    uvicorn.run(
        "routellm.openai_server:app",
        port=args.port,
        host="0.0.0.0",
        workers=args.workers,
    )
