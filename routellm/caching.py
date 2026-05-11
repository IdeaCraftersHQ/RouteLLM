"""Caching layer for LLM completions with exact and semantic matching.

Supports both synchronous and asynchronous cache operations with optional
semantic similarity matching using embeddings.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for completion caching.

    Attributes
    ----------
    enabled : bool
        Whether caching is enabled. Default: True.
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
    enabled: bool = True
    db_path: str = ".routellm_cache.db"
    ttl_seconds: int = 86400  # 24 hours
    semantic_enabled: bool = False
    semantic_threshold: float = 0.95
    embedding_model: str = "text-embedding-3-small"


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
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS completion_cache (
                key TEXT PRIMARY KEY,
                prompt TEXT,
                model TEXT,
                response TEXT,
                embedding BLOB,
                created_at INTEGER
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON completion_cache(model)")
        conn.commit()
        conn.close()

    def _get_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prompt, model, and parameters.

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters (temperature, top_p, etc.).

        Returns
        -------
        str
            SHA256 hash as cache key.
        """
        data = {
            "prompt": prompt,
            "model": model,
            "params": params
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

        Returns
        -------
        dict or None
            Cached response dict, or None if cache miss.
        """
        if not self.config.enabled:
            return None

        key = self._get_key(prompt, model, params)
        logger.debug(f"Cache: Checking for model={model} prompt='{prompt}' key={key}")
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        # 1. Try exact match
        cursor.execute(
            "SELECT response, created_at FROM completion_cache WHERE key = ?",
            (key,)
        )
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
        if self.config.semantic_enabled:
            logger.debug(f"Cache: Attempting semantic match for '{prompt}'")
            prompt_emb = self._get_embedding(prompt)
            cursor.execute(
                "SELECT response, embedding, created_at FROM completion_cache WHERE model = ? AND embedding IS NOT NULL",
                (model,)
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

    async def aget(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async wrapper for get().

        Parameters
        ----------
        prompt : str
            User prompt.
        model : str
            Model name.
        params : dict
            Generation parameters.

        Returns
        -------
        dict or None
            Cached response dict, or None if cache miss.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get, prompt, model, params
        )

    def put(self, prompt: str, model: str, params: Dict[str, Any], response: Dict[str, Any]):
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
        """
        if not self.config.enabled:
            return

        key = self._get_key(prompt, model, params)
        logger.debug(f"Cache: Putting entry for model={model} key={key}")
        embedding = None
        if self.config.semantic_enabled:
            try:
                embedding = self._get_embedding(prompt)
                logger.debug(f"Cache: Generated embedding for model={model}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for cache: {str(e)}")

        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO completion_cache (key, prompt, model, response, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                prompt,
                model,
                json.dumps(response),
                embedding.tobytes() if embedding is not None else None,
                int(time.time())
            )
        )
        conn.commit()
        conn.close()

    async def aput(self, prompt: str, model: str, params: Dict[str, Any], response: Dict[str, Any]):
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
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.put, prompt, model, params, response
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
