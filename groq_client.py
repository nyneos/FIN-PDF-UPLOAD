"""
Groq API Client with retry, timeout, structured logging.
"""

import time
import requests
from logging_config import configure_logging
from config import config

logger = configure_logging()

from config import config

# Use GROQ_URL from config so it can be overridden by env
GROQ_URL = config.GROQ_URL


import time
import requests
from logging_config import configure_logging
from config import config
from typing import List, Optional

logger = configure_logging()

# Use GROQ_URL from config so it can be overridden by env
GROQ_URL = config.GROQ_URL


def _load_api_keys() -> List[str]:
    """Return a list of API keys from either GROQ_API_KEYS or GROQ_API_KEY."""
    keys_env = getattr(config, "GROQ_API_KEYS", None) or None
    if not keys_env:
        # try reading from environment variable directly as a fallback
        import os

        keys_env = os.getenv("GROQ_API_KEYS")

    keys: List[str] = []
    if keys_env:
        # comma or whitespace separated
        for part in str(keys_env).split(','):
            k = part.strip()
            if k:
                keys.append(k)

    # fallback to single key if provided
    if not keys and config.GROQ_API_KEY:
        keys = [config.GROQ_API_KEY]

    return keys


def groq_llm(messages, model: Optional[str] = None):
    """Call Groq with simple round-robin key failover.

    This will try each key in order and retry per key with backoff. Keys must
    be provided via `GROQ_API_KEYS` (comma-separated) or `GROQ_API_KEY`.
    """
    keys = _load_api_keys()
    if not keys:
        raise RuntimeError("No GROQ API keys configured. Set GROQ_API_KEY or GROQ_API_KEYS in env.")

    model = model or config.GROQ_MODEL

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    # Try each key in round-robin fashion with retries per key
    last_error = None
    for key in keys:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, config.AI_RETRY_COUNT + 1):
            try:
                logger.info(f"[Groq] Attempt key_prefix={key[:8]} attempt={attempt} model={model}")
                start = time.time()
                res = requests.post(
                    GROQ_URL,
                    json=payload,
                    headers=headers,
                    timeout=config.AI_TIMEOUT_SECONDS,
                )
                latency = round(time.time() - start, 2)

                if res.status_code == 200:
                    logger.info(f"[Groq] Success in {latency}s with key_prefix={key[:8]}")
                    data = res.json()
                    return data["choices"][0]["message"]["content"]

                # If we hit rate limit or large payload errors, log and try retry/backoff
                logger.warning(f"[Groq] Error {res.status_code} (key_prefix={key[:8]}): {res.text}")
                last_error = RuntimeError(f"HTTP {res.status_code}: {res.text}")

            except Exception as e:
                logger.error(f"[Groq] Exception key_prefix={key[:8]} attempt {attempt}: {e}")
                last_error = e

            # backoff before retrying the same key
            time.sleep(attempt * 1.2)

        # if this key exhausted attempts, move to next key
        logger.info(f"[Groq] Moving to next key after exhausting attempts for key_prefix={key[:8]}")

    # all keys exhausted
    raise RuntimeError(f"Groq API failed after trying all keys: {last_error}")
