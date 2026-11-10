"""Caching layer for LLM completions with exact and semantic matching.

Supports both synchronous and asynchronous cache operations with optional
semantic similarity matching using embeddings.
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for completion caching.

    Attributes
    ----------
    enabled : bool
        Whether caching is enabled. Default: False.
    db_path : str
        Path to SQLite cache database. Default: ".routellm_cache.db".
    ttl_seconds : int
        Time-to-live for cache entries in seconds. Default: 86400 (24h).
    semantic_enabled : bool
        Whether to enable semantic matching via embeddings. Default: False.
    semantic_threshold : float
        Cosine similarity threshold for semantic matching in [0, 1].
        Default: 0.95.
    embedding_model : str
        Model name for generating embeddings. Default: "text-embedding-3-small".
    """

    enabled: bool = False
    db_path: str = ".routellm_cache.db"
    ttl_seconds: int = 86400  # 24 hours
    semantic_enabled: bool = False
    semantic_threshold: float = 0.95
    embedding_model: str = "text-embedding-3-small"


def cache_namespace(principal: str | None, data_class: str | None) -> str:
    """Digest naming whose entries a request may read and write.

    An entry is visible only inside the namespace it was stored under:
    one caller, one data class. The digest, not the names, reaches the
    cache database, so a shared or inspectable store does not list
    callers or classes. Components are encoded unambiguously and `None`
    differs from an empty name.

    Parameters
    ----------
    principal : str, optional
        The authenticated caller's name; None when auth is off.
    data_class : str, optional
        The request's effective data class; None when none applies.

    Returns
    -------
    str
        SHA256 hex digest; "" for no principal and no class, the namespace
        of an unauthenticated, unclassified setup.
    """
    if principal is None and data_class is None:
        return ""
    canonical = json.dumps([principal, data_class], separators=(",", ":"))
    return hashlib.sha256(b"routellm-cache-ns\0" + canonical.encode()).hexdigest()


def conversation_key(messages: Any) -> str:
    """Digest of a whole conversation, for use in a cache key.

    Covers every message: role, text, content parts (images included,
    by value), tool calls and tool results. Two conversations that merely
    end in the same message get different digests.

    Parameters
    ----------
    messages : Any
        The request's `messages`, a list of dicts or a bare string.

    Returns
    -------
    str
        SHA256 hex digest of the canonical JSON form.
    """
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _blank_text(content: Any) -> Any:
    """`content` with the text of its text parts emptied, structure kept."""
    if isinstance(content, str):
        return ""
    if isinstance(content, list):
        return [
            {**part, "text": ""} if isinstance(part, dict) and part.get("type") == "text" else part
            for part in content
        ]
    return content


def context_key(messages: Any) -> str:
    """Digest of everything in a conversation except its last turn's text.

    The semantic cache compares the final user turn by embedding, so that
    is the one thing allowed to differ between a request and the entry it
    reuses. Everything else must match exactly: earlier messages, the
    system prompt, and the last turn's role, tool calls and non-text
    parts (images included, by value).

    Parameters
    ----------
    messages : Any
        The request's `messages`, a list of dicts or a bare string.

    Returns
    -------
    str
        SHA256 hex digest of the canonical JSON form.
    """
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            last = {**last, "content": _blank_text(last.get("content"))}
        scope: Any = [*messages[:-1], last]
    else:
        scope = _blank_text(messages)
    return conversation_key(scope)


class Cache:
    """Persistent cache for completions, supporting exact and semantic matching.

    Stores prompt-response pairs with optional embeddings for semantic
    similarity lookup. Implements both sync and async interfaces.
    """

    def __init__(self, config: CacheConfig = None):
        """Initialize cache with configuration.

        Parameters
        ----------
        config : CacheConfig, optional
            Cache configuration. If None, uses defaults.
        """
        self.config = config or CacheConfig()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema.

        A disabled cache touches nothing on disk. The database is created
        owner-only (0600), and an existing one is brought to 0600: it holds
        prompts and completions in plaintext.
        """
        if not self.config.enabled:
            return
        parent = os.path.dirname(self.config.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.close(os.open(self.config.db_path, os.O_CREAT | os.O_RDWR, 0o600))
        # The mode above only applies to a file this call creates; one
        # that already exists (older version, other umask) is tightened.
        os.chmod(self.config.db_path, 0o600)
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS completion_cache (
                key TEXT PRIMARY KEY,
                prompt TEXT,
                model TEXT,
                response TEXT,
                embedding BLOB,
                created_at INTEGER,
                scope TEXT,
                ns TEXT
            )
        """)
        # A database made before `scope` existed gains the column; its old
        # rows keep a NULL scope, which no semantic lookup matches, so
        # they still serve exact hits and never a fuzzy one.
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(completion_cache)")}
        if "scope" not in columns:
            cursor.execute("ALTER TABLE completion_cache ADD COLUMN scope TEXT")
        # Likewise `ns`: a row without one belongs to no namespace and
        # is never served, exact or semantic.
        if "ns" not in columns:
            cursor.execute("ALTER TABLE completion_cache ADD COLUMN ns TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON completion_cache(model)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_scope ON completion_cache(model, scope)"
        )
        conn.commit()
        conn.close()

    def _get_key(
        self, prompt: str, model: str, params: dict[str, Any], namespace: str | None = None
    ) -> str:
        """Generate cache key from prompt, model, parameters and namespace.

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters (temperature, top_p, etc.).
        namespace : str, optional
            Digest from `cache_namespace`; "" when the caller has none.

        Returns
        -------
        str
            SHA256 hash as cache key.
        """
        data = {"prompt": prompt, "model": model, "params": params, "ns": namespace or ""}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _get_scope(params: dict[str, Any]) -> str:
        """Digest of what a semantic match must share with its entry.

        Callers put a digest of the whole conversation under
        `_conversation` (the exact key) and a digest of everything but
        the last turn's text under `_context`. The scope is every param
        except the first, so only the final turn's wording may differ.
        A caller that supplies no `_context` has no safe fuzzy scope, so
        `_conversation` stays in and semantic matching degrades to exact.

        Parameters
        ----------
        params : dict
            Generation parameters, as passed to `get` and `put`.

        Returns
        -------
        str
            SHA256 hex digest.
        """
        scoped = {k: v for k, v in params.items() if k != "_conversation"}
        if "_context" not in params and "_conversation" in params:
            scoped["_conversation"] = params["_conversation"]
        return hashlib.sha256(json.dumps(scoped, sort_keys=True, default=str).encode()).hexdigest()

    def get(
        self,
        prompt: str,
        model: str,
        params: dict[str, Any],
        namespace: str | None = None,
        semantic: bool = True,
    ) -> dict[str, Any] | None:
        """Retrieve cached completion via exact or semantic match.

        Attempts exact match first, then falls back to semantic matching
        if enabled. Returns None if no match found or cache disabled.

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters.
        namespace : str, optional
            Digest from `cache_namespace`. Only entries stored under the
            same one match, exactly or semantically.
        semantic : bool
            False restricts the lookup to exact matches whatever the
            config says; no embedding is requested.

        Returns
        -------
        dict or None
            Cached response dict, or None if cache miss.
        """
        if not self.config.enabled:
            return None

        namespace = namespace or ""
        key = self._get_key(prompt, model, params, namespace)
        logger.debug(f"Cache: Checking for model={model} key={key}")
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        # 1. Try exact match
        cursor.execute("SELECT response, created_at FROM completion_cache WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row:
            response_str, created_at = row
            if time.time() - created_at <= self.config.ttl_seconds:
                logger.info(f"Cache hit (exact) for {model}")
                conn.close()
                return json.loads(response_str)
            else:
                logger.debug("Cache entry expired (exact)")

        # 2. Try semantic match if enabled
        if self.config.semantic_enabled and semantic:
            logger.debug("Cache: Attempting semantic match")
            prompt_emb = self._get_embedding(prompt)
            cursor.execute(
                "SELECT response, embedding, created_at FROM completion_cache "
                "WHERE model = ? AND scope = ? AND ns = ? AND embedding IS NOT NULL",
                (model, self._get_scope(params), namespace),
            )
            rows = cursor.fetchall()
            for r, emb_blob, c_at in rows:
                if time.time() - c_at > self.config.ttl_seconds:
                    continue

                emb = np.frombuffer(emb_blob, dtype=np.float32)
                # Cosine similarity
                norm_p = np.linalg.norm(prompt_emb)
                norm_e = np.linalg.norm(emb)
                if norm_p == 0 or norm_e == 0:
                    similarity = 0
                else:
                    similarity = np.dot(prompt_emb, emb) / (norm_p * norm_e)

                logger.debug(f"Cache: Semantic similarity candidate score={similarity:.4f}")
                if similarity >= self.config.semantic_threshold:
                    logger.info(f"Cache hit (semantic, score={similarity:.4f}) for {model}")
                    conn.close()
                    return json.loads(r)
        else:
            logger.debug("Cache: Semantic match disabled")

        conn.close()
        return None

    async def aget(
        self,
        prompt: str,
        model: str,
        params: dict[str, Any],
        namespace: str | None = None,
        semantic: bool = True,
    ) -> dict[str, Any] | None:
        """Async wrapper for get().

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters.
        namespace : str, optional
            See `get`.
        semantic : bool
            See `get`.

        Returns
        -------
        dict or None
            Cached response dict, or None if cache miss.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get, prompt, model, params, namespace, semantic
        )

    def put(
        self,
        prompt: str,
        model: str,
        params: dict[str, Any],
        response: dict[str, Any],
        namespace: str | None = None,
        semantic: bool = True,
    ):
        """Store completion in cache.

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters.
        response : dict
            Model response dict (typically ModelResponse.model_dump()).
        namespace : str, optional
            Digest from `cache_namespace`; the entry is visible only there.
        semantic : bool
            False stores no embedding, so the entry never serves a fuzzy hit.
        """
        if not self.config.enabled:
            return

        namespace = namespace or ""
        key = self._get_key(prompt, model, params, namespace)
        logger.debug(f"Cache: Putting entry for model={model} key={key}")
        embedding = None
        if self.config.semantic_enabled and semantic:
            try:
                embedding = self._get_embedding(prompt)
                logger.debug(f"Cache: Generated embedding for model={model}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for cache: {e!s}")

        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO completion_cache "
            "(key, prompt, model, response, embedding, created_at, scope, ns) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                prompt,
                model,
                json.dumps(response),
                embedding.tobytes() if embedding is not None else None,
                int(time.time()),
                self._get_scope(params),
                namespace,
            ),
        )
        conn.commit()
        conn.close()

    async def aput(
        self,
        prompt: str,
        model: str,
        params: dict[str, Any],
        response: dict[str, Any],
        namespace: str | None = None,
        semantic: bool = True,
    ):
        """Async wrapper for put().

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters.
        response : dict
            Model response dict.
        namespace : str, optional
            See `put`.
        semantic : bool
            See `put`.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.put, prompt, model, params, response, namespace, semantic
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text via LiteLLM.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector (float32).
        """
        from litellm import embedding

        res = embedding(model=self.config.embedding_model, input=[text])
        return np.array(res.data[0]["embedding"], dtype=np.float32)
