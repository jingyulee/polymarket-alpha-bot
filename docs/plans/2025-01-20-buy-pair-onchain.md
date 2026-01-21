# Buy Pair On-Chain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Buy Pair On-Chain" button to portfolio cards that purchases both hedge positions with one click.

**Architecture:** Backend wallet management with password-based encryption (AES-256-GCM). Trading via hybrid approach: split USDC on-chain into YES+NO tokens, then sell unwanted side via CLOB API. Frontend settings page for wallet setup, buy modal for trade execution.

**Tech Stack:** FastAPI, Web3.py, py-clob-client, cryptography (backend); React, TypeScript (frontend)

---

## Phase 1: Backend Wallet Management

### Task 1.1: Create Wallet Encryption Module

**Files:**
- Create: `backend/core/wallet/__init__.py`
- Create: `backend/core/wallet/encryption.py`

**Step 1: Create the wallet package init**

```python
# backend/core/wallet/__init__.py
"""Wallet management for on-chain trading."""

from core.wallet.encryption import encrypt_private_key, decrypt_private_key
from core.wallet.storage import WalletStorage
from core.wallet.manager import WalletManager

__all__ = [
    "encrypt_private_key",
    "decrypt_private_key",
    "WalletStorage",
    "WalletManager",
]
```

**Step 2: Create encryption module**

```python
# backend/core/wallet/encryption.py
"""AES-256-GCM encryption for private keys."""

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive 256-bit key from password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(password.encode())


def encrypt_private_key(private_key: str, password: str) -> tuple[str, str]:
    """
    Encrypt private key with password.

    Returns:
        Tuple of (encrypted_key_b64, salt_b64)
    """
    salt = os.urandom(16)
    key = _derive_key(password, salt)

    aesgcm = AESGCM(key)
    nonce = os.urandom(12)

    ciphertext = aesgcm.encrypt(nonce, private_key.encode(), None)
    encrypted = nonce + ciphertext

    return base64.b64encode(encrypted).decode(), base64.b64encode(salt).decode()


def decrypt_private_key(encrypted_b64: str, salt_b64: str, password: str) -> str:
    """
    Decrypt private key with password.

    Raises:
        ValueError: If password is incorrect
    """
    encrypted = base64.b64decode(encrypted_b64)
    salt = base64.b64decode(salt_b64)
    key = _derive_key(password, salt)

    nonce = encrypted[:12]
    ciphertext = encrypted[12:]

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()
    except Exception:
        raise ValueError("Invalid password")
```

**Step 3: Commit**

```bash
git add backend/core/wallet/__init__.py backend/core/wallet/encryption.py
git commit -m "feat(wallet): add AES-256-GCM encryption module"
```

---

### Task 1.2: Create Wallet Storage Module

**Files:**
- Create: `backend/core/wallet/storage.py`

**Step 1: Create storage module**

```python
# backend/core/wallet/storage.py
"""Encrypted wallet file storage."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from core.wallet.encryption import encrypt_private_key, decrypt_private_key


class WalletData(TypedDict):
    address: str
    encrypted_key: str
    salt: str
    created_at: str


class WalletStorage:
    """Manages encrypted wallet storage on disk."""

    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        """Check if wallet file exists."""
        return self.path.exists()

    def load(self) -> WalletData | None:
        """Load wallet data (without decrypting)."""
        if not self.exists():
            return None
        return json.loads(self.path.read_text())

    def save(self, address: str, private_key: str, password: str) -> WalletData:
        """Encrypt and save wallet."""
        encrypted_key, salt = encrypt_private_key(private_key, password)

        data: WalletData = {
            "address": address,
            "encrypted_key": encrypted_key,
            "salt": salt,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        return data

    def decrypt(self, password: str) -> str:
        """Decrypt and return private key."""
        data = self.load()
        if not data:
            raise ValueError("No wallet found")
        return decrypt_private_key(data["encrypted_key"], data["salt"], password)

    def delete(self) -> bool:
        """Delete wallet file."""
        if self.exists():
            self.path.unlink()
            return True
        return False
```

**Step 2: Commit**

```bash
git add backend/core/wallet/storage.py
git commit -m "feat(wallet): add encrypted storage module"
```

---

### Task 1.3: Create Wallet Manager Module

**Files:**
- Create: `backend/core/wallet/manager.py`
- Create: `backend/core/wallet/contracts.py`

**Step 1: Create contracts module with ABIs and addresses**

```python
# backend/core/wallet/contracts.py
"""Polymarket contract addresses and ABIs."""

# Polygon mainnet contracts
CONTRACTS = {
    "USDC_E": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "CTF": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
    "CTF_EXCHANGE": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NEG_RISK_CTF_EXCHANGE": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "NEG_RISK_ADAPTER": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]

CTF_ABI = [
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_operator", "type": "address"},
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_operator", "type": "address"},
            {"name": "_approved", "type": "bool"},
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "type": "function",
    },
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "partition", "type": "uint256[]"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "splitPosition",
        "outputs": [],
        "type": "function",
    },
    {
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
]
```

**Step 2: Create wallet manager**

```python
# backend/core/wallet/manager.py
"""Wallet management: generate, import, unlock, status."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eth_account import Account
from web3 import Web3
from loguru import logger

from core.wallet.storage import WalletStorage
from core.wallet.contracts import CONTRACTS, ERC20_ABI, CTF_ABI


@dataclass
class WalletBalances:
    pol: float
    usdc_e: float


@dataclass
class WalletStatus:
    exists: bool
    address: Optional[str]
    unlocked: bool
    balances: Optional[WalletBalances]
    approvals_set: bool


class WalletManager:
    """Manages wallet lifecycle and blockchain interactions."""

    def __init__(self, storage: WalletStorage, rpc_url: str):
        self.storage = storage
        self.rpc_url = rpc_url
        self._unlocked_key: Optional[str] = None
        self._address: Optional[str] = None

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked_key is not None

    @property
    def address(self) -> Optional[str]:
        if self._address:
            return self._address
        data = self.storage.load()
        return data["address"] if data else None

    def _get_web3(self) -> Web3:
        return Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 60}))

    def generate(self, password: str) -> str:
        """Generate new wallet, encrypt and save."""
        if self.storage.exists():
            raise ValueError("Wallet already exists")

        account = Account.create()
        self.storage.save(account.address, account.key.hex(), password)
        logger.info(f"Generated new wallet: {account.address}")
        return account.address

    def import_key(self, private_key: str, password: str) -> str:
        """Import existing private key, encrypt and save."""
        if self.storage.exists():
            raise ValueError("Wallet already exists")

        account = Account.from_key(private_key)
        self.storage.save(account.address, private_key, password)
        logger.info(f"Imported wallet: {account.address}")
        return account.address

    def unlock(self, password: str) -> str:
        """Decrypt private key into memory."""
        self._unlocked_key = self.storage.decrypt(password)
        data = self.storage.load()
        self._address = data["address"] if data else None
        logger.info(f"Wallet unlocked: {self._address}")
        return self._address

    def lock(self) -> None:
        """Clear private key from memory."""
        self._unlocked_key = None
        logger.info("Wallet locked")

    def get_balances(self) -> WalletBalances:
        """Get POL and USDC.e balances."""
        address = self.address
        if not address:
            raise ValueError("No wallet configured")

        w3 = self._get_web3()
        checksum = Web3.to_checksum_address(address)

        pol = float(w3.from_wei(w3.eth.get_balance(checksum), "ether"))

        usdc = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["USDC_E"]),
            abi=ERC20_ABI,
        )
        usdc_balance = usdc.functions.balanceOf(checksum).call() / 1e6

        return WalletBalances(pol=pol, usdc_e=usdc_balance)

    def check_approvals(self) -> bool:
        """Check if all Polymarket approvals are set."""
        address = self.address
        if not address:
            return False

        w3 = self._get_web3()
        checksum = Web3.to_checksum_address(address)

        usdc = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["USDC_E"]),
            abi=ERC20_ABI,
        )
        ctf = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["CTF"]),
            abi=CTF_ABI,
        )

        # Check USDC approvals
        for contract in ["CTF", "CTF_EXCHANGE", "NEG_RISK_CTF_EXCHANGE"]:
            allowance = usdc.functions.allowance(checksum, CONTRACTS[contract]).call()
            if allowance == 0:
                return False

        # Check CTF approvals
        for contract in ["CTF_EXCHANGE", "NEG_RISK_CTF_EXCHANGE", "NEG_RISK_ADAPTER"]:
            approved = ctf.functions.isApprovedForAll(checksum, CONTRACTS[contract]).call()
            if not approved:
                return False

        return True

    def get_status(self) -> WalletStatus:
        """Get complete wallet status."""
        exists = self.storage.exists()
        address = self.address

        balances = None
        approvals_set = False

        if exists and address:
            try:
                balances = self.get_balances()
                approvals_set = self.check_approvals()
            except Exception as e:
                logger.warning(f"Failed to fetch wallet status: {e}")

        return WalletStatus(
            exists=exists,
            address=address,
            unlocked=self.is_unlocked,
            balances=balances,
            approvals_set=approvals_set,
        )

    def set_approvals(self) -> list[str]:
        """Set all Polymarket contract approvals. Returns tx hashes."""
        if not self.is_unlocked:
            raise ValueError("Wallet must be unlocked")

        w3 = self._get_web3()
        address = Web3.to_checksum_address(self._address)
        account = w3.eth.account.from_key(self._unlocked_key)

        usdc = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["USDC_E"]),
            abi=ERC20_ABI,
        )
        ctf = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["CTF"]),
            abi=CTF_ABI,
        )

        MAX_UINT256 = 2**256 - 1
        tx_hashes = []

        approvals = [
            (usdc, "approve", CONTRACTS["CTF"], MAX_UINT256),
            (usdc, "approve", CONTRACTS["CTF_EXCHANGE"], MAX_UINT256),
            (usdc, "approve", CONTRACTS["NEG_RISK_CTF_EXCHANGE"], MAX_UINT256),
            (ctf, "setApprovalForAll", CONTRACTS["CTF_EXCHANGE"], True),
            (ctf, "setApprovalForAll", CONTRACTS["NEG_RISK_CTF_EXCHANGE"], True),
            (ctf, "setApprovalForAll", CONTRACTS["NEG_RISK_ADAPTER"], True),
        ]

        for contract, method, spender, value in approvals:
            fn = getattr(contract.functions, method)
            tx = fn(Web3.to_checksum_address(spender), value).build_transaction({
                "from": address,
                "nonce": w3.eth.get_transaction_count(address),
                "gas": 100000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] != 1:
                raise ValueError(f"Approval failed: {tx_hash.hex()}")

            tx_hashes.append(tx_hash.hex())
            logger.info(f"Approval tx: {tx_hash.hex()[:20]}...")

        return tx_hashes

    def get_unlocked_key(self) -> str:
        """Get the unlocked private key (for trading)."""
        if not self._unlocked_key:
            raise ValueError("Wallet not unlocked")
        return self._unlocked_key
```

**Step 3: Commit**

```bash
git add backend/core/wallet/contracts.py backend/core/wallet/manager.py
git commit -m "feat(wallet): add wallet manager with blockchain interactions"
```

---

### Task 1.4: Create Wallet API Router

**Files:**
- Create: `backend/server/routers/wallet.py`
- Modify: `backend/server/main.py`

**Step 1: Create wallet router**

```python
# backend/server/routers/wallet.py
"""Wallet management API endpoints."""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from core.wallet.storage import WalletStorage
from core.wallet.manager import WalletManager


router = APIRouter()

# Singleton wallet manager
_wallet_manager: Optional[WalletManager] = None


def get_wallet_manager() -> WalletManager:
    """Get or create wallet manager singleton."""
    global _wallet_manager
    if _wallet_manager is None:
        wallet_path = Path("data/wallet.enc")
        rpc_url = os.environ.get("CHAINSTACK_NODE", "")
        if not rpc_url:
            raise HTTPException(status_code=500, detail="CHAINSTACK_NODE not configured")

        storage = WalletStorage(wallet_path)
        _wallet_manager = WalletManager(storage, rpc_url)
    return _wallet_manager


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class PasswordRequest(BaseModel):
    password: str


class ImportRequest(BaseModel):
    private_key: str
    password: str


class WalletStatusResponse(BaseModel):
    exists: bool
    address: Optional[str]
    unlocked: bool
    balances: Optional[dict]
    approvals_set: bool


class GenerateResponse(BaseModel):
    address: str
    message: str


class UnlockResponse(BaseModel):
    unlocked: bool
    address: str
    balances: dict


class ApprovalResponse(BaseModel):
    success: bool
    tx_hashes: list[str]


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/status", response_model=WalletStatusResponse)
async def get_status():
    """Get wallet status including balances and approval state."""
    manager = get_wallet_manager()
    status = manager.get_status()

    return WalletStatusResponse(
        exists=status.exists,
        address=status.address,
        unlocked=status.unlocked,
        balances={
            "pol": status.balances.pol,
            "usdc_e": status.balances.usdc_e,
        } if status.balances else None,
        approvals_set=status.approvals_set,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_wallet(req: PasswordRequest):
    """Generate a new wallet encrypted with password."""
    manager = get_wallet_manager()

    try:
        address = manager.generate(req.password)
        return GenerateResponse(
            address=address,
            message="Wallet created. Fund with POL and USDC.e, then set approvals.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import", response_model=GenerateResponse)
async def import_wallet(req: ImportRequest):
    """Import existing private key encrypted with password."""
    manager = get_wallet_manager()

    try:
        address = manager.import_key(req.private_key, req.password)
        return GenerateResponse(
            address=address,
            message="Wallet imported. Check balances and set approvals if needed.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/unlock", response_model=UnlockResponse)
async def unlock_wallet(req: PasswordRequest):
    """Unlock wallet for trading (decrypt key into memory)."""
    manager = get_wallet_manager()

    try:
        address = manager.unlock(req.password)
        balances = manager.get_balances()
        return UnlockResponse(
            unlocked=True,
            address=address,
            balances={"pol": balances.pol, "usdc_e": balances.usdc_e},
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail="Invalid password")


@router.post("/lock")
async def lock_wallet():
    """Lock wallet (clear key from memory)."""
    manager = get_wallet_manager()
    manager.lock()
    return {"locked": True}


@router.post("/approve-contracts", response_model=ApprovalResponse)
async def approve_contracts():
    """Set all Polymarket contract approvals (requires unlocked wallet)."""
    manager = get_wallet_manager()

    if not manager.is_unlocked:
        raise HTTPException(status_code=401, detail="Unlock wallet first")

    try:
        tx_hashes = manager.set_approvals()
        return ApprovalResponse(success=True, tx_hashes=tx_hashes)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Register router in main.py**

Add to `backend/server/main.py` after the existing router imports (around line 14):

```python
from server.routers import data, pipeline, prices, wallet
```

And add after the existing `include_router` calls (around line 53):

```python
app.include_router(wallet.router, prefix="/wallet", tags=["wallet"])
```

**Step 3: Add cryptography dependency**

```bash
cd backend && uv add cryptography
```

**Step 4: Commit**

```bash
git add backend/server/routers/wallet.py backend/server/main.py backend/pyproject.toml backend/uv.lock
git commit -m "feat(api): add wallet management endpoints"
```

---

## Phase 2: Backend Trading Logic

### Task 2.1: Create Trading Executor Module

**Files:**
- Create: `backend/core/trading/__init__.py`
- Create: `backend/core/trading/executor.py`

**Step 1: Create trading package init**

```python
# backend/core/trading/__init__.py
"""Trading execution for on-chain operations."""

from core.trading.executor import TradingExecutor

__all__ = ["TradingExecutor"]
```

**Step 2: Create trading executor**

```python
# backend/core/trading/executor.py
"""Execute on-chain trades: split + CLOB sell."""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from web3 import Web3
from loguru import logger

from core.wallet.contracts import CONTRACTS, ERC20_ABI, CTF_ABI
from core.wallet.manager import WalletManager


@dataclass
class MarketInfo:
    market_id: str
    question: str
    condition_id: str
    yes_token_id: str
    no_token_id: Optional[str]
    yes_price: float
    no_price: float


@dataclass
class TradeResult:
    success: bool
    market_id: str
    position: str
    amount: float
    split_tx: Optional[str]
    clob_order_id: Optional[str]
    clob_filled: bool
    error: Optional[str] = None


@dataclass
class BuyPairResult:
    success: bool
    pair_id: str
    target: TradeResult
    cover: TradeResult
    total_spent: float
    final_balances: dict


class TradingExecutor:
    """Executes on-chain trades via split + CLOB sell."""

    def __init__(self, wallet_manager: WalletManager):
        self.wallet = wallet_manager
        self.rpc_url = wallet_manager.rpc_url

    def _get_web3(self) -> Web3:
        return Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 60}))

    async def get_market_info(self, market_id: str) -> MarketInfo:
        """Fetch market info from Polymarket API."""
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
            data = resp.json()

        clob_tokens = json.loads(data.get("clobTokenIds", "[]"))
        prices = json.loads(data.get("outcomePrices", "[0.5, 0.5]"))

        return MarketInfo(
            market_id=market_id,
            question=data.get("question", ""),
            condition_id=data.get("conditionId", ""),
            yes_token_id=clob_tokens[0] if clob_tokens else "",
            no_token_id=clob_tokens[1] if len(clob_tokens) > 1 else None,
            yes_price=float(prices[0]) if prices else 0.5,
            no_price=float(prices[1]) if len(prices) > 1 else 0.5,
        )

    def _get_clob_client(self):
        """Initialize CLOB client with optional proxy support."""
        try:
            from py_clob_client.client import ClobClient
            import py_clob_client.http_helpers.helpers as clob_helpers
        except ImportError:
            logger.error("py-clob-client not installed")
            return None

        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy:
            logger.info(f"Using proxy: {proxy[:30]}...")
            clob_helpers._http_client = httpx.Client(http2=True, proxy=proxy, timeout=30.0)

        private_key = self.wallet.get_unlocked_key()
        address = self.wallet.address

        try:
            client = ClobClient(
                "https://clob.polymarket.com",
                key=private_key,
                chain_id=137,
                signature_type=0,
                funder=address,
            )
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            return client
        except Exception as e:
            logger.error(f"CLOB API error: {e}")
            return None

    def _split_position(
        self,
        condition_id: str,
        amount_usd: float,
    ) -> str:
        """Split USDC into YES + NO tokens. Returns tx hash."""
        w3 = self._get_web3()
        address = Web3.to_checksum_address(self.wallet.address)
        account = w3.eth.account.from_key(self.wallet.get_unlocked_key())

        ctf = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["CTF"]),
            abi=CTF_ABI,
        )

        amount_wei = int(amount_usd * 1e6)
        condition_bytes = bytes.fromhex(
            condition_id[2:] if condition_id.startswith("0x") else condition_id
        )

        tx = ctf.functions.splitPosition(
            Web3.to_checksum_address(CONTRACTS["USDC_E"]),
            bytes(32),  # parentCollectionId
            condition_bytes,
            [1, 2],  # partition for YES, NO
            amount_wei,
        ).build_transaction({
            "from": address,
            "nonce": w3.eth.get_transaction_count(address),
            "gas": 300000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })

        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(f"Split TX: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt["status"] != 1:
            raise ValueError(f"Split failed: {tx_hash.hex()}")

        return tx_hash.hex()

    def _sell_via_clob(
        self,
        token_id: str,
        amount: float,
        price: float,
    ) -> tuple[Optional[str], bool]:
        """Sell tokens via CLOB. Returns (order_id, filled)."""
        client = self._get_clob_client()
        if not client:
            return None, False

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL

            sell_price = round(price * 0.98, 2)  # 2% below market

            order = client.create_order(
                OrderArgs(
                    token_id=token_id,
                    price=sell_price,
                    size=amount,
                    side=SELL,
                )
            )
            result = client.post_order(order, OrderType.GTC)
            order_id = result.get("orderID", str(result)[:40])
            logger.info(f"CLOB order placed: {order_id}")
            return order_id, True
        except Exception as e:
            logger.error(f"CLOB sell error: {e}")
            return None, False

    async def buy_single_position(
        self,
        market_id: str,
        position: str,  # "YES" or "NO"
        amount: float,
        skip_clob_sell: bool = False,
    ) -> TradeResult:
        """Buy a single position on a market."""
        position = position.upper()
        if position not in ["YES", "NO"]:
            return TradeResult(
                success=False,
                market_id=market_id,
                position=position,
                amount=amount,
                split_tx=None,
                clob_order_id=None,
                clob_filled=False,
                error="Position must be YES or NO",
            )

        # Get market info
        market = await self.get_market_info(market_id)

        # Determine unwanted side
        unwanted_token = market.no_token_id if position == "YES" else market.yes_token_id
        unwanted_price = market.no_price if position == "YES" else market.yes_price

        # Split position
        try:
            split_tx = self._split_position(market.condition_id, amount)
        except Exception as e:
            return TradeResult(
                success=False,
                market_id=market_id,
                position=position,
                amount=amount,
                split_tx=None,
                clob_order_id=None,
                clob_filled=False,
                error=f"Split failed: {e}",
            )

        time.sleep(2)  # Wait for chain confirmation

        # Sell unwanted side
        clob_order_id = None
        clob_filled = False

        if not skip_clob_sell and unwanted_token:
            clob_order_id, clob_filled = self._sell_via_clob(
                unwanted_token,
                amount,
                unwanted_price,
            )

        return TradeResult(
            success=True,
            market_id=market_id,
            position=position,
            amount=amount,
            split_tx=split_tx,
            clob_order_id=clob_order_id,
            clob_filled=clob_filled,
        )

    async def buy_pair(
        self,
        pair_id: str,
        target_market_id: str,
        target_position: str,
        cover_market_id: str,
        cover_position: str,
        amount_per_position: float,
        skip_clob_sell: bool = False,
    ) -> BuyPairResult:
        """Buy both positions in a portfolio pair."""

        # Check wallet status
        if not self.wallet.is_unlocked:
            raise ValueError("Wallet not unlocked")

        balances = self.wallet.get_balances()
        required = amount_per_position * 2

        if balances.usdc_e < required:
            raise ValueError(
                f"Insufficient USDC.e: need {required:.2f}, have {balances.usdc_e:.2f}"
            )

        # Buy target position
        logger.info(f"Buying target: {target_position} on {target_market_id}")
        target_result = await self.buy_single_position(
            target_market_id,
            target_position,
            amount_per_position,
            skip_clob_sell,
        )

        # Buy cover position
        logger.info(f"Buying cover: {cover_position} on {cover_market_id}")
        cover_result = await self.buy_single_position(
            cover_market_id,
            cover_position,
            amount_per_position,
            skip_clob_sell,
        )

        # Get final balances
        final_balances = self.wallet.get_balances()

        return BuyPairResult(
            success=target_result.success and cover_result.success,
            pair_id=pair_id,
            target=target_result,
            cover=cover_result,
            total_spent=amount_per_position * 2,
            final_balances={
                "pol": final_balances.pol,
                "usdc_e": final_balances.usdc_e,
            },
        )
```

**Step 3: Commit**

```bash
git add backend/core/trading/__init__.py backend/core/trading/executor.py
git commit -m "feat(trading): add on-chain trading executor"
```

---

### Task 2.2: Create Trading API Router

**Files:**
- Create: `backend/server/routers/trading.py`
- Modify: `backend/server/main.py`

**Step 1: Create trading router**

```python
# backend/server/routers/trading.py
"""Trading API endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from server.routers.wallet import get_wallet_manager
from core.trading.executor import TradingExecutor


router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class BuyPairRequest(BaseModel):
    pair_id: str
    target_market_id: str
    target_position: str
    cover_market_id: str
    cover_position: str
    amount_per_position: float
    skip_clob_sell: bool = False


class TradeResultModel(BaseModel):
    success: bool
    market_id: str
    position: str
    amount: float
    split_tx: Optional[str]
    clob_order_id: Optional[str]
    clob_filled: bool
    error: Optional[str] = None


class BuyPairResponse(BaseModel):
    success: bool
    pair_id: str
    target: TradeResultModel
    cover: TradeResultModel
    total_spent: float
    final_balances: dict


class EstimateResponse(BaseModel):
    pair_id: str
    total_cost: float
    target_market: dict
    cover_market: dict
    wallet_balance: float
    sufficient_balance: bool


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/buy-pair", response_model=BuyPairResponse)
async def buy_pair(req: BuyPairRequest):
    """Execute a pair purchase (target + cover positions)."""
    wallet_manager = get_wallet_manager()

    if not wallet_manager.is_unlocked:
        raise HTTPException(status_code=401, detail="Unlock wallet first")

    executor = TradingExecutor(wallet_manager)

    try:
        result = await executor.buy_pair(
            pair_id=req.pair_id,
            target_market_id=req.target_market_id,
            target_position=req.target_position,
            cover_market_id=req.cover_market_id,
            cover_position=req.cover_position,
            amount_per_position=req.amount_per_position,
            skip_clob_sell=req.skip_clob_sell,
        )

        return BuyPairResponse(
            success=result.success,
            pair_id=result.pair_id,
            target=TradeResultModel(
                success=result.target.success,
                market_id=result.target.market_id,
                position=result.target.position,
                amount=result.target.amount,
                split_tx=result.target.split_tx,
                clob_order_id=result.target.clob_order_id,
                clob_filled=result.target.clob_filled,
                error=result.target.error,
            ),
            cover=TradeResultModel(
                success=result.cover.success,
                market_id=result.cover.market_id,
                position=result.cover.position,
                amount=result.cover.amount,
                split_tx=result.cover.split_tx,
                clob_order_id=result.cover.clob_order_id,
                clob_filled=result.cover.clob_filled,
                error=result.cover.error,
            ),
            total_spent=result.total_spent,
            final_balances=result.final_balances,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Buy pair failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buy-pair/estimate", response_model=EstimateResponse)
async def estimate_buy_pair(req: BuyPairRequest):
    """Estimate costs for a pair purchase without executing."""
    wallet_manager = get_wallet_manager()
    executor = TradingExecutor(wallet_manager)

    try:
        target_market = await executor.get_market_info(req.target_market_id)
        cover_market = await executor.get_market_info(req.cover_market_id)

        total_cost = req.amount_per_position * 2
        balances = wallet_manager.get_balances()

        return EstimateResponse(
            pair_id=req.pair_id,
            total_cost=total_cost,
            target_market={
                "question": target_market.question[:60],
                "position": req.target_position,
                "price": target_market.yes_price if req.target_position == "YES" else target_market.no_price,
            },
            cover_market={
                "question": cover_market.question[:60],
                "position": req.cover_position,
                "price": cover_market.yes_price if req.cover_position == "YES" else cover_market.no_price,
            },
            wallet_balance=balances.usdc_e,
            sufficient_balance=balances.usdc_e >= total_cost,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Register router in main.py**

Add to imports in `backend/server/main.py`:

```python
from server.routers import data, pipeline, prices, wallet, trading
```

And add after wallet router:

```python
app.include_router(trading.router, prefix="/trading", tags=["trading"])
```

**Step 3: Commit**

```bash
git add backend/server/routers/trading.py backend/server/main.py
git commit -m "feat(api): add trading endpoints for buy-pair"
```

---

## Phase 3: Frontend Settings Page

### Task 3.1: Create Wallet Hook

**Files:**
- Create: `frontend/hooks/useWallet.ts`

**Step 1: Create the hook**

```typescript
// frontend/hooks/useWallet.ts
'use client'

import { useState, useEffect, useCallback } from 'react'
import { getApiBaseUrl } from '@/config/api-config'

interface WalletBalances {
  pol: number
  usdc_e: number
}

interface WalletStatus {
  exists: boolean
  address: string | null
  unlocked: boolean
  balances: WalletBalances | null
  approvals_set: boolean
}

interface UseWalletReturn {
  status: WalletStatus | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  generate: (password: string) => Promise<string>
  importKey: (privateKey: string, password: string) => Promise<string>
  unlock: (password: string) => Promise<void>
  lock: () => Promise<void>
  approveContracts: () => Promise<void>
}

export function useWallet(): UseWalletReturn {
  const [status, setStatus] = useState<WalletStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const apiBase = getApiBaseUrl()

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/wallet/status`)
      if (!res.ok) throw new Error('Failed to fetch wallet status')
      const data = await res.json()
      setStatus(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [apiBase])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [refresh])

  const generate = useCallback(async (password: string): Promise<string> => {
    const res = await fetch(`${apiBase}/wallet/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to generate wallet')
    }
    const data = await res.json()
    await refresh()
    return data.address
  }, [apiBase, refresh])

  const importKey = useCallback(async (privateKey: string, password: string): Promise<string> => {
    const res = await fetch(`${apiBase}/wallet/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ private_key: privateKey, password }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to import wallet')
    }
    const data = await res.json()
    await refresh()
    return data.address
  }, [apiBase, refresh])

  const unlock = useCallback(async (password: string): Promise<void> => {
    const res = await fetch(`${apiBase}/wallet/unlock`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Invalid password')
    }
    await refresh()
  }, [apiBase, refresh])

  const lock = useCallback(async (): Promise<void> => {
    await fetch(`${apiBase}/wallet/lock`, { method: 'POST' })
    await refresh()
  }, [apiBase, refresh])

  const approveContracts = useCallback(async (): Promise<void> => {
    const res = await fetch(`${apiBase}/wallet/approve-contracts`, {
      method: 'POST',
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to approve contracts')
    }
    await refresh()
  }, [apiBase, refresh])

  return {
    status,
    loading,
    error,
    refresh,
    generate,
    importKey,
    unlock,
    lock,
    approveContracts,
  }
}
```

**Step 2: Commit**

```bash
git add frontend/hooks/useWallet.ts
git commit -m "feat(frontend): add useWallet hook"
```

---

### Task 3.2: Create Settings Page

**Files:**
- Create: `frontend/app/settings/page.tsx`

**Step 1: Create the settings page**

```typescript
// frontend/app/settings/page.tsx
'use client'

import { useState } from 'react'
import { useWallet } from '@/hooks/useWallet'

export default function SettingsPage() {
  const { status, loading, error, generate, importKey, unlock, lock, approveContracts, refresh } = useWallet()

  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [privateKey, setPrivateKey] = useState('')
  const [mode, setMode] = useState<'generate' | 'import' | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const handleGenerate = async () => {
    if (password !== confirmPassword) {
      setActionError('Passwords do not match')
      return
    }
    if (password.length < 8) {
      setActionError('Password must be at least 8 characters')
      return
    }

    setActionLoading(true)
    setActionError(null)
    try {
      await generate(password)
      setPassword('')
      setConfirmPassword('')
      setMode(null)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to generate wallet')
    } finally {
      setActionLoading(false)
    }
  }

  const handleImport = async () => {
    if (password.length < 8) {
      setActionError('Password must be at least 8 characters')
      return
    }
    if (!privateKey.startsWith('0x') || privateKey.length !== 66) {
      setActionError('Invalid private key format')
      return
    }

    setActionLoading(true)
    setActionError(null)
    try {
      await importKey(privateKey, password)
      setPassword('')
      setPrivateKey('')
      setMode(null)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to import wallet')
    } finally {
      setActionLoading(false)
    }
  }

  const handleUnlock = async () => {
    setActionLoading(true)
    setActionError(null)
    try {
      await unlock(password)
      setPassword('')
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Invalid password')
    } finally {
      setActionLoading(false)
    }
  }

  const handleApprove = async () => {
    setActionLoading(true)
    setActionError(null)
    try {
      await approveContracts()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to approve contracts')
    } finally {
      setActionLoading(false)
    }
  }

  const copyAddress = () => {
    if (status?.address) {
      navigator.clipboard.writeText(status.address)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-void p-6">
        <div className="max-w-xl mx-auto">
          <h1 className="text-2xl font-bold text-text-primary mb-6">Settings</h1>
          <div className="text-text-muted">Loading...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-void p-6">
      <div className="max-w-xl mx-auto">
        <h1 className="text-2xl font-bold text-text-primary mb-6">Settings</h1>

        {/* Wallet Section */}
        <div className="bg-surface border border-border rounded-xl p-5">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Wallet</h2>

          {error && (
            <div className="mb-4 p-3 bg-rose/10 border border-rose/25 rounded-lg text-rose text-sm">
              {error}
            </div>
          )}

          {actionError && (
            <div className="mb-4 p-3 bg-rose/10 border border-rose/25 rounded-lg text-rose text-sm">
              {actionError}
            </div>
          )}

          {!status?.exists ? (
            // No wallet - show setup options
            <div className="space-y-4">
              {!mode && (
                <div className="flex gap-3">
                  <button
                    onClick={() => setMode('generate')}
                    className="flex-1 py-2.5 px-4 bg-emerald/10 hover:bg-emerald/15 border border-emerald/25 rounded-lg text-emerald text-sm font-medium transition-colors"
                  >
                    Generate New Wallet
                  </button>
                  <button
                    onClick={() => setMode('import')}
                    className="flex-1 py-2.5 px-4 bg-cyan/10 hover:bg-cyan/15 border border-cyan/25 rounded-lg text-cyan text-sm font-medium transition-colors"
                  >
                    Import Existing
                  </button>
                </div>
              )}

              {mode === 'generate' && (
                <div className="space-y-3">
                  <input
                    type="password"
                    placeholder="Password (min 8 characters)"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <input
                    type="password"
                    placeholder="Confirm password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => { setMode(null); setPassword(''); setConfirmPassword(''); setActionError(null); }}
                      className="px-4 py-2 text-text-muted hover:text-text-primary text-sm"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleGenerate}
                      disabled={actionLoading}
                      className="flex-1 py-2 px-4 bg-emerald hover:bg-emerald/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {actionLoading ? 'Creating...' : 'Create Wallet'}
                    </button>
                  </div>
                </div>
              )}

              {mode === 'import' && (
                <div className="space-y-3">
                  <input
                    type="password"
                    placeholder="Private key (0x...)"
                    value={privateKey}
                    onChange={(e) => setPrivateKey(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm font-mono focus:outline-none focus:border-cyan"
                  />
                  <input
                    type="password"
                    placeholder="Encryption password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => { setMode(null); setPassword(''); setPrivateKey(''); setActionError(null); }}
                      className="px-4 py-2 text-text-muted hover:text-text-primary text-sm"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleImport}
                      disabled={actionLoading}
                      className="flex-1 py-2 px-4 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {actionLoading ? 'Importing...' : 'Import Wallet'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            // Wallet exists - show status
            <div className="space-y-4">
              {/* Status indicator */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${status.unlocked ? 'bg-emerald' : 'bg-amber'}`} />
                  <span className="text-sm text-text-muted">
                    {status.unlocked ? 'Unlocked' : 'Locked'}
                  </span>
                </div>
                {status.unlocked && (
                  <button
                    onClick={lock}
                    className="text-sm text-text-muted hover:text-rose transition-colors"
                  >
                    Lock Wallet
                  </button>
                )}
              </div>

              {/* Address */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">Address:</span>
                <code className="text-sm text-text-primary font-mono">
                  {status.address?.slice(0, 10)}...{status.address?.slice(-8)}
                </code>
                <button
                  onClick={copyAddress}
                  className="text-text-muted hover:text-cyan transition-colors"
                >
                  {copied ? '✓' : '📋'}
                </button>
              </div>

              {/* Balances */}
              {status.balances && (
                <div className="bg-surface-elevated rounded-lg p-3 space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">POL:</span>
                    <span className="text-text-primary font-mono">{status.balances.pol.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">USDC.e:</span>
                    <span className="text-text-primary font-mono">${status.balances.usdc_e.toFixed(2)}</span>
                  </div>
                </div>
              )}

              {/* Approvals */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">Contract Approvals:</span>
                <span className={`text-sm ${status.approvals_set ? 'text-emerald' : 'text-amber'}`}>
                  {status.approvals_set ? '✓ Set' : '⚠ Not Set'}
                </span>
              </div>

              {/* Unlock form */}
              {!status.unlocked && (
                <div className="flex gap-2 pt-2">
                  <input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleUnlock()}
                    className="flex-1 px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <button
                    onClick={handleUnlock}
                    disabled={actionLoading}
                    className="px-4 py-2 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50"
                  >
                    {actionLoading ? '...' : 'Unlock'}
                  </button>
                </div>
              )}

              {/* Approve button */}
              {status.unlocked && !status.approvals_set && (
                <button
                  onClick={handleApprove}
                  disabled={actionLoading}
                  className="w-full py-2.5 px-4 bg-amber/10 hover:bg-amber/15 border border-amber/25 rounded-lg text-amber text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading ? 'Approving...' : 'Set Contract Approvals'}
                </button>
              )}
            </div>
          )}
        </div>

        {/* Refresh button */}
        <div className="mt-4 text-center">
          <button
            onClick={refresh}
            className="text-sm text-text-muted hover:text-text-primary transition-colors"
          >
            ↻ Refresh
          </button>
        </div>
      </div>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/app/settings/page.tsx
git commit -m "feat(frontend): add settings page with wallet management"
```

---

### Task 3.3: Add Settings Link to Navigation

**Files:**
- Modify: `frontend/app/layout.tsx`

**Step 1: Read current layout**

Check the current navigation structure.

**Step 2: Add settings link**

Add a settings link to the navigation. The exact location depends on the current layout structure.

**Step 3: Commit**

```bash
git add frontend/app/layout.tsx
git commit -m "feat(frontend): add settings link to navigation"
```

---

## Phase 4: Frontend Buy Flow

### Task 4.1: Create Buy Pair Modal Component

**Files:**
- Create: `frontend/components/trading/BuyPairModal.tsx`

**Step 1: Create the modal component**

```typescript
// frontend/components/trading/BuyPairModal.tsx
'use client'

import { useState } from 'react'
import { useWallet } from '@/hooks/useWallet'
import { getApiBaseUrl } from '@/config/api-config'
import type { Portfolio } from '@/types/portfolio'

interface BuyPairModalProps {
  portfolio: Portfolio
  onClose: () => void
}

type Step = 'input' | 'confirming' | 'executing' | 'success' | 'error'

interface TradeResult {
  success: boolean
  target: { split_tx?: string; clob_order_id?: string; error?: string }
  cover: { split_tx?: string; clob_order_id?: string; error?: string }
  total_spent: number
  final_balances: { pol: number; usdc_e: number }
}

export function BuyPairModal({ portfolio: p, onClose }: BuyPairModalProps) {
  const { status } = useWallet()
  const [amount, setAmount] = useState('10')
  const [step, setStep] = useState<Step>('input')
  const [result, setResult] = useState<TradeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [executionStep, setExecutionStep] = useState('')

  const apiBase = getApiBaseUrl()
  const amountNum = parseFloat(amount) || 0
  const totalCost = amountNum * 2
  const hasSufficientBalance = (status?.balances?.usdc_e || 0) >= totalCost

  const handleBuy = async () => {
    if (!status?.unlocked) {
      setError('Please unlock your wallet first')
      return
    }
    if (!hasSufficientBalance) {
      setError('Insufficient USDC.e balance')
      return
    }

    setStep('executing')
    setError(null)
    setExecutionStep('Splitting target position...')

    try {
      const res = await fetch(`${apiBase}/trading/buy-pair`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pair_id: p.pair_id,
          target_market_id: p.target_market_id,
          target_position: p.target_position,
          cover_market_id: p.cover_market_id,
          cover_position: p.cover_position,
          amount_per_position: amountNum,
          skip_clob_sell: false,
        }),
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Trade failed')
      }

      setResult(data)
      setStep(data.success ? 'success' : 'error')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Trade failed')
      setStep('error')
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-void/80 backdrop-blur-sm" />

      <div
        className="relative w-full max-w-md bg-surface border border-border rounded-xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="h-0.5 bg-cyan" />
        <div className="px-5 py-4 border-b border-border flex items-center justify-between">
          <h2 className="text-lg font-semibold text-text-primary">Buy Pair On-Chain</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          {step === 'input' && (
            <>
              {/* Positions */}
              <div className="space-y-2">
                <div className="bg-surface-elevated rounded-lg p-3">
                  <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">Target</div>
                  <p className="text-sm text-text-primary">{p.target_question.slice(0, 50)}...</p>
                  <span className={`text-xs font-mono ${p.target_position === 'YES' ? 'text-emerald' : 'text-rose'}`}>
                    {p.target_position} @ ${p.target_price.toFixed(2)}
                  </span>
                </div>
                <div className="bg-surface-elevated rounded-lg p-3">
                  <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">Cover</div>
                  <p className="text-sm text-text-primary">{p.cover_question.slice(0, 50)}...</p>
                  <span className={`text-xs font-mono ${p.cover_position === 'YES' ? 'text-emerald' : 'text-rose'}`}>
                    {p.cover_position} @ ${p.cover_price.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Amount input */}
              <div>
                <label className="text-sm text-text-muted block mb-1">Amount per position</label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted">$</span>
                  <input
                    type="number"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    min="1"
                    step="1"
                    className="w-full pl-7 pr-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary text-sm font-mono focus:outline-none focus:border-cyan"
                  />
                </div>
              </div>

              {/* Summary */}
              <div className="bg-surface-elevated rounded-lg p-3 space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">2 positions × ${amountNum.toFixed(2)}</span>
                  <span className="text-text-primary font-mono">${totalCost.toFixed(2)} total</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Your balance</span>
                  <span className={`font-mono ${hasSufficientBalance ? 'text-emerald' : 'text-rose'}`}>
                    ${(status?.balances?.usdc_e || 0).toFixed(2)} USDC.e
                  </span>
                </div>
              </div>

              {error && (
                <div className="p-3 bg-rose/10 border border-rose/25 rounded-lg text-rose text-sm">
                  {error}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={onClose}
                  className="flex-1 py-2.5 px-4 border border-border rounded-lg text-text-muted hover:text-text-primary text-sm transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleBuy}
                  disabled={!status?.unlocked || !hasSufficientBalance || amountNum <= 0}
                  className="flex-1 py-2.5 px-4 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {!status?.unlocked ? 'Unlock Wallet First' : 'Confirm Purchase'}
                </button>
              </div>
            </>
          )}

          {step === 'executing' && (
            <div className="py-8 text-center">
              <div className="animate-spin w-8 h-8 border-2 border-cyan border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-text-primary">{executionStep}</p>
              <p className="text-text-muted text-sm mt-2">This may take a minute...</p>
            </div>
          )}

          {step === 'success' && result && (
            <div className="py-4 space-y-4">
              <div className="text-center">
                <div className="w-12 h-12 bg-emerald/20 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-emerald" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">Purchase Complete</h3>
              </div>

              <div className="bg-surface-elevated rounded-lg p-3 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">Total spent</span>
                  <span className="text-text-primary font-mono">${result.total_spent.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">New balance</span>
                  <span className="text-text-primary font-mono">${result.final_balances.usdc_e.toFixed(2)}</span>
                </div>
              </div>

              {result.target.split_tx && (
                <div className="text-xs text-text-muted">
                  Target TX: <code className="text-cyan">{result.target.split_tx.slice(0, 20)}...</code>
                </div>
              )}
              {result.cover.split_tx && (
                <div className="text-xs text-text-muted">
                  Cover TX: <code className="text-cyan">{result.cover.split_tx.slice(0, 20)}...</code>
                </div>
              )}

              <button
                onClick={onClose}
                className="w-full py-2.5 px-4 bg-surface-elevated hover:bg-surface border border-border rounded-lg text-text-primary text-sm transition-colors"
              >
                Close
              </button>
            </div>
          )}

          {step === 'error' && (
            <div className="py-4 space-y-4">
              <div className="text-center">
                <div className="w-12 h-12 bg-rose/20 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-rose" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">Trade Failed</h3>
                <p className="text-rose text-sm mt-2">{error}</p>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={onClose}
                  className="flex-1 py-2.5 px-4 border border-border rounded-lg text-text-muted text-sm"
                >
                  Close
                </button>
                <button
                  onClick={() => { setStep('input'); setError(null); }}
                  className="flex-1 py-2.5 px-4 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/components/trading/BuyPairModal.tsx
git commit -m "feat(frontend): add BuyPairModal component"
```

---

### Task 4.2: Add Buy Button to Portfolio Modal

**Files:**
- Modify: `frontend/components/PortfolioModal.tsx`

**Step 1: Import and add state for buy modal**

Add at the top of the file:
```typescript
import { useState } from 'react'
import { BuyPairModal } from '@/components/trading/BuyPairModal'
```

**Step 2: Add state and button**

Inside the component, add state:
```typescript
const [showBuyModal, setShowBuyModal] = useState(false)
```

**Step 3: Add Buy button next to Open Markets button**

Replace the single button with a button group:

```typescript
{/* Action Buttons */}
<div className="flex gap-2">
  {(p.target_group_slug || p.cover_group_slug) && (
    <button
      onClick={() => {
        if (p.target_group_slug) {
          window.open(`https://polymarket.com/event/${p.target_group_slug}`, '_blank')
        }
        if (p.cover_group_slug) {
          window.open(`https://polymarket.com/event/${p.cover_group_slug}`, '_blank')
        }
      }}
      className="flex-1 py-2.5 px-4 bg-cyan/10 hover:bg-cyan/15 border border-cyan/25 hover:border-cyan/40 rounded-lg text-cyan text-sm font-medium transition-all flex items-center justify-center gap-2"
    >
      <span>Open Markets</span>
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
      </svg>
    </button>
  )}
  <button
    onClick={() => setShowBuyModal(true)}
    className="flex-1 py-2.5 px-4 bg-emerald/10 hover:bg-emerald/15 border border-emerald/25 hover:border-emerald/40 rounded-lg text-emerald text-sm font-medium transition-all flex items-center justify-center gap-2"
  >
    <span>Buy Pair On-Chain</span>
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
    </svg>
  </button>
</div>

{/* Buy Modal */}
{showBuyModal && (
  <BuyPairModal portfolio={p} onClose={() => setShowBuyModal(false)} />
)}
```

**Step 4: Commit**

```bash
git add frontend/components/PortfolioModal.tsx
git commit -m "feat(frontend): add Buy Pair button to portfolio modal"
```

---

## Final Steps

### Task 5.1: Update Routers Init

**Files:**
- Modify: `backend/server/routers/__init__.py`

**Step 1: Add new routers to init**

```python
# backend/server/routers/__init__.py
from server.routers import data, pipeline, prices, portfolio_prices, wallet, trading
```

**Step 2: Commit**

```bash
git add backend/server/routers/__init__.py
git commit -m "chore: export wallet and trading routers"
```

---

### Task 5.2: Test the Complete Flow

**Step 1: Start the backend**

```bash
cd backend && uv run uvicorn server.main:app --reload
```

**Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```

**Step 3: Manual testing checklist**

1. Navigate to `/settings`
2. Generate or import a wallet
3. Unlock the wallet with password
4. Set contract approvals (requires funded wallet)
5. Navigate to terminal/portfolio view
6. Click on a portfolio row to open modal
7. Click "Buy Pair On-Chain"
8. Enter amount and confirm
9. Verify transaction executes

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete buy pair on-chain feature"
```

---

## Summary

This plan implements the "Buy Pair On-Chain" feature in 4 phases:

1. **Backend Wallet Management** - Encrypted storage, generate/import/unlock APIs
2. **Backend Trading Logic** - Split + CLOB sell execution
3. **Frontend Settings Page** - Wallet setup and management UI
4. **Frontend Buy Flow** - Modal and button integration

Total new files: ~12
Modified files: ~4
Estimated tasks: ~15 commits
