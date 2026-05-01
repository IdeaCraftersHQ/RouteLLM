# RouteLLM x402 Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task.

**Goal:** Enable RouteLLM to pay x402-gated LLM endpoints on behalf of the caller,
using a `PaymentGateway` interface with a concrete `X402Adapter` wrapping the
official `x402` PyPI package.

**Architecture:** `routellm/payment/gateway.py` defines `PaymentGateway` ABC;
`routellm/payment/x402.py` implements it using `x402` (PyPI); the RouteLLM
`Controller` and `openai_server` accept an optional `PaymentGateway`; when a
downstream LLM returns 402, the gateway pays and the request is retried
transparently.

**Tech Stack:** Python ≥3.10, `x402` PyPI (v2+), `httpx` (async HTTP),
`fastapi` (existing), `pydantic` (existing).

---

## Task List

- [ ] T-01: Add `x402` dependency
- [ ] T-02: `PaymentGateway` ABC + types
- [ ] T-03: `X402Adapter` impl
- [ ] T-04: Integrate into `Controller`
- [ ] T-05: Integrate into `openai_server`
- [ ] T-06: Unit tests
- [ ] T-07: Integration test

---

### Task T-01: Add x402 dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependency**

```toml
[project.dependencies]
x402 = ">=2.0.0"
```

**Step 2: Install**

```
pip install -e ".[dev]"
```

**Step 3: Verify import**

```python
import x402
print(x402.__version__)
```

**Step 4: Commit**

```
git commit -m "chore(deps): add x402 PyPI package"
```

---

### Task T-02: `PaymentGateway` ABC + types

**Files:**
- Create: `routellm/payment/__init__.py`
- Create: `routellm/payment/gateway.py`
- Create: `routellm/payment/types.py`
- Test: `routellm/tests/payment/test_gateway.py`

**Step 1: Write failing test**

```python
from routellm.payment.types import PaymentChallenge, PaymentReceipt

def test_payment_challenge_fields():
    c = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="1.00",
        currency="USDC",
        payload={},
    )
    assert c.scheme == "x402"
    assert c.amount == "1.00"
```

**Step 2: Run — expect ImportError**

```
pytest routellm/tests/payment/test_gateway.py -v
```

**Step 3: Implement `routellm/payment/types.py`**

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PaymentChallenge:
    scheme: str          # "x402"
    network: str         # "base", "ethereum"
    amount: str          # decimal string
    currency: str        # "USDC"
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass
class PaymentReceipt:
    tx_hash: str
    network: str
    amount: str
    currency: str
    paid_at: int         # unix timestamp
    resource: str = ""
```

**Step 4: Implement `routellm/payment/gateway.py`**

```python
from abc import ABC, abstractmethod
from .types import PaymentChallenge, PaymentReceipt

class PaymentGateway(ABC):
    """Transport-agnostic payment interface.
    Implement this to add a new payment provider to RouteLLM.
    """

    @abstractmethod
    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        """Fulfill a 402 payment challenge. Raises PaymentError on failure."""

    @abstractmethod
    async def verify(self, receipt: PaymentReceipt) -> bool:
        """Verify a receipt is valid (for server-side use)."""

    @property
    @abstractmethod
    def networks(self) -> list[str]:
        """Supported network identifiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, e.g. 'x402'."""
```

**Step 5: Create `routellm/payment/__init__.py`**

```python
from .gateway import PaymentGateway
from .types import PaymentChallenge, PaymentReceipt

__all__ = ["PaymentGateway", "PaymentChallenge", "PaymentReceipt"]
```

**Step 6: Run tests — PASS**

```
pytest routellm/tests/payment/test_gateway.py -v
```

**Step 7: Commit**

```
git commit -m "feat(payment): PaymentGateway ABC and types"
```

---

### Task T-03: `X402Adapter` impl

**Files:**
- Create: `routellm/payment/x402.py`
- Test: `routellm/tests/payment/test_x402_adapter.py`

**Note:** `x402` PyPI is async-first. Uses `x402.client.PaymentClient` for paying
and `x402.server.verify_payment` for verifying. Wallet keypair loaded from env var
`ROUTELLM_WALLET_PRIVATE_KEY` (EVM) or delegated to `x402` wallet manager.

**Step 1: Write failing test**

```python
from unittest.mock import AsyncMock, patch
from routellm.payment.x402 import X402Adapter
from routellm.payment.types import PaymentChallenge

@pytest.mark.asyncio
async def test_x402_adapter_name():
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    assert adapter.name == "x402"
    assert "base" in adapter.networks
```

**Step 2: Run — expect ImportError**

```
pytest routellm/tests/payment/test_x402_adapter.py -v
```

**Step 3: Implement `routellm/payment/x402.py`**

```python
import time
from .gateway import PaymentGateway
from .types import PaymentChallenge, PaymentReceipt

class X402Adapter(PaymentGateway):
    """Wraps the x402 PyPI package (coinbase/x402)."""

    def __init__(self, private_key: str,
                 networks: list[str] | None = None):
        # Use exact x402 SDK — see pypi.org/project/x402
        # import x402.client, x402.wallet
        self._private_key = private_key
        self._networks = networks or ["base", "ethereum", "polygon"]

    @property
    def name(self) -> str:
        return "x402"

    @property
    def networks(self) -> list[str]:
        return self._networks

    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        # Delegate to x402.client.PaymentClient.pay(challenge.payload)
        # Map result to PaymentReceipt
        raise NotImplementedError

    async def verify(self, receipt: PaymentReceipt) -> bool:
        # Delegate to x402.server.verify_payment(receipt.tx_hash, ...)
        raise NotImplementedError
```

**Step 4: Complete impl using `x402` SDK**

> Follow exact x402 PyPI docs: `x402.client.PaymentClient` for outbound,
> `x402.server.verify_payment` for inbound. Do not re-implement signing.

**Step 5: Run tests — PASS**

```
pytest routellm/tests/payment/test_x402_adapter.py -v
```

**Step 6: Commit**

```
git commit -m "feat(payment/x402): X402Adapter wrapping x402 PyPI"
```

---

### Task T-04: Integrate into Controller

**Files:**
- Modify: `routellm/controller.py`
- Test: `routellm/tests/test_controller_payment.py`

**Note:** `Controller.__init__` accepts optional `payment_gateway: PaymentGateway`.
`Controller.route()` (or equivalent request path) catches HTTP 402 from downstream
LLM, calls `gateway.pay(challenge)`, retries once with payment header.

**Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_controller_retries_on_402(mock_gateway, mock_402_client):
    controller = Controller(
        routers=["random"],
        payment_gateway=mock_gateway,
    )
    response = await controller.acompletion(...)
    mock_gateway.pay.assert_awaited_once()
    assert response.choices[0].message.content is not None
```

**Step 2: Modify `Controller.__init__`**

```python
def __init__(self, ..., payment_gateway: PaymentGateway | None = None):
    ...
    self.payment_gateway = payment_gateway
```

**Step 3: Add 402 retry logic**

```python
async def _request_with_payment(self, client, **kwargs):
    try:
        return await client.chat.completions.create(**kwargs)
    except HTTPStatusError as e:
        if e.response.status_code == 402 and self.payment_gateway:
            challenge = _parse_challenge(e.response)
            receipt = await self.payment_gateway.pay(challenge)
            kwargs["extra_headers"]["X-Payment"] = receipt.tx_hash
            return await client.chat.completions.create(**kwargs)
        raise
```

**Step 4: Run tests — PASS**

```
pytest routellm/tests/test_controller_payment.py -v
```

**Step 5: Commit**

```
git commit -m "feat(controller): 402 retry with PaymentGateway"
```

---

### Task T-05: Integrate into openai_server

**Files:**
- Modify: `routellm/openai_server.py`

**Note:** Pass `payment_gateway` from CLI arg / env to `Controller` at startup.
Add `--payment-provider` and `--wallet-key-env` CLI flags.

**Step 1: Add to `lifespan`**

```python
gateway = None
if args.payment_provider == "x402":
    key = os.environ.get(args.wallet_key_env or "ROUTELLM_WALLET_KEY", "")
    if key:
        gateway = X402Adapter(private_key=key)

CONTROLLER = Controller(..., payment_gateway=gateway)
```

**Step 2: Add argparse args**

```python
parser.add_argument("--payment-provider", default=None,
    choices=["x402"], help="Enable payment gateway")
parser.add_argument("--wallet-key-env", default="ROUTELLM_WALLET_KEY",
    help="Env var holding wallet private key")
```

**Step 3: Run server smoke test**

```
python -m routellm.openai_server --routers random --help
```

Verify `--payment-provider` appears in help output.

**Step 4: Commit**

```
git commit -m "feat(server): expose payment gateway via CLI flags"
```

---

### Task T-06: Unit tests

**Files:**
- Create: `routellm/tests/payment/test_types.py`
- Create: `routellm/tests/payment/test_x402_adapter.py`

**Run all payment tests:**

```
pytest routellm/tests/payment/ -v
```

**Commit:**

```
git commit -m "test(payment): unit tests for gateway and x402 adapter"
```

---

### Task T-07: Integration test

**Files:**
- Create: `routellm/tests/test_x402_integration.py`

**Note:** Use `x402` test facilitator (no real chain / wallet needed).
Tests that a mocked 402 response from an LLM endpoint triggers pay + retry.

**Step 1: Write test**

```python
# mark: integration
@pytest.mark.asyncio
async def test_full_402_flow():
    # 1. Start local x402 test facilitator
    # 2. Point controller at mock LLM that returns 402 first call
    # 3. Assert second call succeeds with payment header
    # 4. Assert receipt returned
```

**Step 2: Run**

```
pytest routellm/tests/test_x402_integration.py -v -m integration
```

**Step 3: Commit**

```
git commit -m "test(integration): full 402 retry flow in RouteLLM"
```
