# Response Middleware Refactor + x402 Separate Package

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task.

**Goal:** Replace the ad-hoc `PaymentGateway` in `routellm/payment/` with a general
`ResponseMiddleware` protocol in core, then move the x402 implementation into a
separate installable package `routellm-x402`.

**Architecture:** A new `ResponseMiddleware` Protocol in `routellm/controller.py`
intercepts every LLM response (or exception) and can retry with modified headers.
`Controller` holds a list of `response_middleware: list[ResponseMiddleware]` and
calls them in order. The x402 adapter becomes `X402ResponseMiddleware` living in a
sibling package `extensions/routellm_x402/` with its own `pyproject.toml` — heavy
EVM deps stay out of core. The existing `PaymentGateway` ABC, `payment/` module,
and payment-specific `Controller` params/logic are deleted from core.

**Tech Stack:** Python ≥3.10, x402[evm]>=2.0.0 (in the extension package only),
pytest, pytest-asyncio.

---

## Task List

- [ ] T-01: Add `ResponseMiddleware` protocol to core + wire into `Controller`
- [ ] T-02: Delete `routellm/payment/` and clean up `Controller` / `openai_server`
- [ ] T-03: Create `extensions/routellm_x402/` package scaffold
- [ ] T-04: Implement `X402ResponseMiddleware` in the extension package
- [ ] T-05: Tests for `X402ResponseMiddleware`
- [ ] T-06: Wire extension into `openai_server` via `--response-middleware` CLI flag

---

### Task T-01: Add `ResponseMiddleware` protocol + wire into Controller

**Files:**
- Modify: `routellm/controller.py`
- Test: `routellm/tests/test_response_middleware.py`

**Step 1: Write failing test**

```python
# routellm/tests/test_response_middleware.py
import pytest
from unittest.mock import AsyncMock

from routellm.controller import Controller, ResponseMiddleware


class PassthroughMiddleware:
    """ResponseMiddleware that does nothing — lets response pass through."""
    async def handle(self, exc: Exception, retry) -> object:
        raise exc  # re-raise, no payment


@pytest.mark.asyncio
async def test_response_middleware_called_on_exception():
    """Controller should call response_middleware.handle when request raises."""
    handled = []

    class RecordingMiddleware:
        async def handle(self, exc: Exception, retry):
            handled.append(exc)
            raise exc

    from unittest.mock import patch

    error = Exception("upstream error")
    error.status_code = 500

    async def bad_request(extra_headers):
        raise error

    with patch("routellm.controller.acompletion", side_effect=error):
        controller = Controller(
            routers=["random"],
            strong_model="gpt-4",
            weak_model="gpt-3.5-turbo",
            response_middleware=[RecordingMiddleware()],
        )
        with pytest.raises(Exception, match="upstream error"):
            await controller._call_with_middleware(bad_request)

    assert len(handled) == 1
```

**Step 2: Run — expect ImportError / AttributeError**

```
pytest routellm/tests/test_response_middleware.py -v
```

**Step 3: Add `ResponseMiddleware` protocol and wire into `Controller`**

In `routellm/controller.py`, add after the existing `Middleware` Protocol:

```python
class ResponseMiddleware(Protocol):
    """Protocol for middleware that intercepts LLM responses or errors.

    Implement this to add retry logic, payment, caching, etc.
    handle() receives the exception that was raised (if any) and a retry
    callable. Call retry(extra_headers={...}) to retry with modified headers,
    or re-raise exc to propagate the error.
    """
    async def handle(self, exc: Exception, retry) -> object:
        """
        Args:
            exc: The exception raised by the upstream LLM call.
            retry: Async callable retry(extra_headers: dict) -> response.
        Returns:
            The (possibly retried) response.
        Raises:
            exc (or another exception) if not handled.
        """
        ...
```

Add `response_middleware: list[ResponseMiddleware] | None = None` to
`Controller.__init__` alongside existing `middleware` param:

```python
self.response_middleware = response_middleware or []
```

Add `_call_with_middleware` replacing `_request_with_payment`:

```python
async def _call_with_middleware(self, make_request, extra_headers: dict | None = None):
    """Call make_request; on exception, run response_middleware chain."""
    try:
        return await make_request(extra_headers=extra_headers or {})
    except Exception as exc:
        last_exc = exc
        for mw in self.response_middleware:
            try:
                return await mw.handle(exc, lambda h: make_request(extra_headers=h))
            except Exception as e:
                last_exc = e
        raise last_exc
```

Update `acompletion` to call `_call_with_middleware` instead of
`_request_with_payment` (same call site, just rename).

**Step 4: Run tests — PASS**

```
pytest routellm/tests/test_response_middleware.py -v
```

**Step 5: Commit**

```
git commit -m "feat(controller): ResponseMiddleware protocol for response interception"
```

---

### Task T-02: Delete `routellm/payment/` and clean `Controller` / `openai_server`

**Files:**
- Delete: `routellm/payment/` (entire directory)
- Delete: `routellm/tests/payment/` (entire directory)
- Delete: `routellm/tests/test_controller_payment.py`
- Delete: `routellm/tests/test_x402_integration.py`
- Modify: `routellm/controller.py`
- Modify: `routellm/openai_server.py`
- Modify: `pyproject.toml`

**Step 1: Remove payment directory and tests**

```bash
git rm -r routellm/payment/
git rm -r routellm/tests/payment/
git rm routellm/tests/test_controller_payment.py
git rm routellm/tests/test_x402_integration.py
```

**Step 2: Clean `routellm/controller.py`**

Remove:
- `from routellm.payment.gateway import PaymentGateway`
- `from routellm.payment.types import PaymentChallenge`
- `payment_gateway: Optional[PaymentGateway] = None` param from `__init__`
- `self.payment_gateway = payment_gateway`
- `_parse_402_challenge` static method
- `_request_with_payment` method (replaced by `_call_with_middleware` from T-01)

**Step 3: Clean `routellm/openai_server.py`**

Remove from `lifespan`:
```python
gateway = None
if args.payment_provider == "x402":
    from routellm.payment.x402 import X402Adapter
    key = os.environ.get(args.wallet_key_env or "ROUTELLM_WALLET_KEY", "")
    if key:
        gateway = X402Adapter(private_key=key)
```
Remove `payment_gateway=gateway` from `Controller(...)`.

Remove argparse args:
- `--payment-provider`
- `--wallet-key-env`

(These will return in T-06 as `--response-middleware x402`.)

**Step 4: Clean `pyproject.toml`**

Remove `'x402[evm]>=2.0.0'` from `[project.dependencies]`.

**Step 5: Verify nothing broken**

```
pytest routellm/tests/ -v --ignore=routellm/tests/payment 2>&1 | tail -20
```

**Step 6: Commit**

```
git commit -m "refactor(core): remove PaymentGateway; superseded by ResponseMiddleware"
```

---

### Task T-03: Create `extensions/routellm_x402/` package scaffold

**Files:**
- Create: `extensions/routellm_x402/pyproject.toml`
- Create: `extensions/routellm_x402/routellm_x402/__init__.py`
- Create: `extensions/routellm_x402/tests/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p extensions/routellm_x402/routellm_x402
mkdir -p extensions/routellm_x402/tests
```

**Step 2: Create `extensions/routellm_x402/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "routellm-x402"
version = "0.1.0"
description = "x402 payment middleware adapter for RouteLLM"
requires-python = ">=3.10"
dependencies = [
    "routellm>=0.2.0",
    "x402[evm]>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[tool.setuptools.packages.find]
where = ["."]
include = ["routellm_x402*"]
```

**Step 3: Create `extensions/routellm_x402/routellm_x402/__init__.py`**

```python
from .middleware import X402ResponseMiddleware

__all__ = ["X402ResponseMiddleware"]
```

**Step 4: Create empty `extensions/routellm_x402/tests/__init__.py`**

```python
```

**Step 5: Verify package is importable (without x402 installed, just structure)**

```bash
cd extensions/routellm_x402 && python -c "import routellm_x402" 2>&1
```

Expected: `ModuleNotFoundError: No module named 'routellm_x402.middleware'` — that's fine, scaffold done.

**Step 6: Commit**

```
git commit -m "chore: scaffold routellm-x402 extension package"
```

---

### Task T-04: Implement `X402ResponseMiddleware`

**Files:**
- Create: `extensions/routellm_x402/routellm_x402/middleware.py`

**Step 1: Write failing test** (see T-05 — write it first, then implement)

Skip ahead to T-05 Step 1, run to confirm ImportError, then return here.

**Step 2: Implement `extensions/routellm_x402/routellm_x402/middleware.py`**

```python
import os
import time

from routellm.controller import ResponseMiddleware


class X402ResponseMiddleware:
    """ResponseMiddleware that pays x402-gated LLM endpoints automatically.

    On HTTP 402, parses the payment challenge from the response body,
    signs and submits payment via the x402 SDK, then retries the request
    with the X-PAYMENT header set to the transaction hash.

    Requires:
        - x402[evm] installed
        - Private key via constructor or ROUTELLM_WALLET_KEY env var

    Usage:
        from routellm_x402 import X402ResponseMiddleware
        from routellm.controller import Controller

        mw = X402ResponseMiddleware(private_key=os.environ["ROUTELLM_WALLET_KEY"])
        controller = Controller(..., response_middleware=[mw])
    """

    def __init__(
        self,
        private_key: str | None = None,
        networks: list[str] | None = None,
    ):
        self._private_key = private_key or os.environ.get("ROUTELLM_WALLET_KEY", "")
        self._networks = networks or ["base", "ethereum", "polygon"]

    def _build_client(self):
        """Build a configured x402HTTPClient with EVM signer for each network."""
        from eth_account import Account
        from x402.client import x402Client
        from x402.http.x402_http_client import x402HTTPClient
        from x402.mechanisms.evm.exact import ExactEvmScheme

        account = Account.from_key(self._private_key)

        async def signer(message: bytes) -> bytes:
            return account.sign_message(message).signature

        client = x402Client()
        network_map = {
            "base": "eip155:8453",
            "ethereum": "eip155:1",
            "polygon": "eip155:137",
        }
        for net in self._networks:
            caip2 = network_map.get(net)
            if caip2:
                client.register(caip2, ExactEvmScheme(signer=signer))

        return x402HTTPClient(client)

    async def handle(self, exc: Exception, retry) -> object:
        """Pay and retry on HTTP 402; re-raise all other exceptions."""
        if getattr(exc, "status_code", None) != 402:
            raise exc

        body = {}
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.json()
            except Exception:
                pass

        from x402.schemas import PaymentRequired

        http_client = self._build_client()
        payment_required = PaymentRequired(**body)
        payload = await http_client.create_payment_payload(payment_required)
        tx_hash = getattr(payload, "transaction_hash", None) or str(payload)

        return await retry({"X-PAYMENT": tx_hash})
```

**Step 3: Run tests from T-05 — PASS**

```
cd extensions/routellm_x402 && pytest tests/ -v
```

**Step 4: Commit**

```
git commit -m "feat(routellm-x402): X402ResponseMiddleware implementation"
```

---

### Task T-05: Tests for `X402ResponseMiddleware`

**Files:**
- Create: `extensions/routellm_x402/tests/test_middleware.py`

**Step 1: Write failing test**

```python
# extensions/routellm_x402/tests/test_middleware.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_import():
    from routellm_x402 import X402ResponseMiddleware
    assert X402ResponseMiddleware is not None


def test_networks_default():
    from routellm_x402 import X402ResponseMiddleware
    mw = X402ResponseMiddleware(private_key="0x" + "a" * 64)
    assert "base" in mw._networks
    assert "ethereum" in mw._networks


@pytest.mark.asyncio
async def test_handle_reraises_non_402():
    from routellm_x402 import X402ResponseMiddleware
    mw = X402ResponseMiddleware(private_key="0x" + "a" * 64)

    err = Exception("server error")
    err.status_code = 500

    with pytest.raises(Exception, match="server error"):
        await mw.handle(err, retry=AsyncMock())


@pytest.mark.asyncio
async def test_handle_pays_and_retries_on_402():
    from routellm_x402 import X402ResponseMiddleware
    mw = X402ResponseMiddleware(private_key="0x" + "a" * 64)

    err = Exception("Payment Required")
    err.status_code = 402
    err.response = MagicMock()
    err.response.json = MagicMock(return_value={
        "scheme": "x402",
        "network": "base",
        "amount": "0.001",
        "currency": "USDC",
        "resource": "https://llm.example.com/v1/chat",
    })

    mock_payload = MagicMock()
    mock_payload.transaction_hash = "0xpaid"
    mock_http_client = MagicMock()
    mock_http_client.create_payment_payload = AsyncMock(return_value=mock_payload)

    mock_response = MagicMock()
    retry = AsyncMock(return_value=mock_response)

    with patch.object(mw, "_build_client", return_value=mock_http_client):
        with patch("routellm_x402.middleware.PaymentRequired", side_effect=lambda **kw: kw):
            result = await mw.handle(err, retry)

    retry.assert_awaited_once_with({"X-PAYMENT": "0xpaid"})
    assert result is mock_response


@pytest.mark.asyncio
async def test_full_controller_integration():
    """Controller with X402ResponseMiddleware retries on 402."""
    from unittest.mock import patch as upatch
    from routellm.controller import Controller
    from routellm_x402 import X402ResponseMiddleware

    mw = X402ResponseMiddleware(private_key="0x" + "a" * 64)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hello"

    err = Exception("402")
    err.status_code = 402
    err.response = MagicMock()
    err.response.json = MagicMock(return_value={})

    mock_payload = MagicMock()
    mock_payload.transaction_hash = "0xpaid"
    mock_http_client = MagicMock()
    mock_http_client.create_payment_payload = AsyncMock(return_value=mock_payload)

    call_count = 0

    async def fake_request(extra_headers):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise err
        return mock_response

    with upatch.object(mw, "_build_client", return_value=mock_http_client):
        with upatch("routellm_x402.middleware.PaymentRequired", side_effect=lambda **kw: kw):
            controller = Controller(
                routers=["random"],
                strong_model="gpt-4",
                weak_model="gpt-3.5-turbo",
                response_middleware=[mw],
            )
            result = await controller._call_with_middleware(fake_request)

    assert call_count == 2
    assert result is mock_response
```

**Step 2: Run — expect ImportError (middleware.py not yet written)**

```
cd extensions/routellm_x402 && pytest tests/test_middleware.py -v 2>&1 | tail -10
```

**Step 3: Go implement T-04, then return and run:**

```
cd extensions/routellm_x402 && pytest tests/test_middleware.py -v
```

Expected: 5 passed (mock-based; tests requiring live x402 SDK will skip if not installed).

**Step 4: Commit**

```
git commit -m "test(routellm-x402): tests for X402ResponseMiddleware"
```

---

### Task T-06: Wire extension into `openai_server` via `--response-middleware`

**Files:**
- Modify: `routellm/openai_server.py`

**Step 1: Add `--response-middleware` argparse arg**

Replace the removed `--payment-provider` / `--wallet-key-env` with a generic flag:

```python
parser.add_argument(
    "--response-middleware",
    nargs="*",
    default=[],
    help="Response middleware to enable (e.g. x402)",
)
parser.add_argument(
    "--wallet-key-env",
    default="ROUTELLM_WALLET_KEY",
    help="Env var holding wallet private key (used by x402 middleware)",
)
```

**Step 2: Build middleware list in `lifespan`**

```python
response_middleware = []
for mw_name in (args.response_middleware or []):
    if mw_name == "x402":
        try:
            from routellm_x402 import X402ResponseMiddleware
        except ImportError:
            raise RuntimeError(
                "x402 middleware requested but routellm-x402 not installed. "
                "Run: pip install routellm-x402"
            )
        key = os.environ.get(args.wallet_key_env or "ROUTELLM_WALLET_KEY", "")
        if not key:
            raise RuntimeError(
                f"x402 middleware requires a wallet key in env var {args.wallet_key_env}"
            )
        response_middleware.append(X402ResponseMiddleware(private_key=key))
    else:
        raise RuntimeError(f"Unknown response middleware: {mw_name}")

CONTROLLER = Controller(
    ...,
    response_middleware=response_middleware,
)
```

**Step 3: Smoke test**

```bash
python -m routellm.openai_server --routers random --help 2>&1 | grep -E "response-middleware|wallet-key"
```

Expected: both flags appear.

**Step 4: Commit**

```
git commit -m "feat(server): --response-middleware flag replaces --payment-provider"
```
