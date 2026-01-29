"""
Groq API Client with retry, timeout, structured logging.
"""

import time
import requests
from logging_config import configure_logging
from config import config
from utils.token_tracker import record_tokens
from utils.token_budget import default_budget as token_budget, TokenBudget
import os

# Per-key token budgets (lazy-init)
_per_key_budgets = {}

def _get_key_budget(key: str) -> TokenBudget:
    """Return a TokenBudget instance for the given API key (cached)."""
    if not key:
        return token_budget
    if key in _per_key_budgets:
        return _per_key_budgets[key]

    # Per-key defaults: prefer explicit env vars, otherwise fallback to global config
    try:
        key_tpm = int(os.getenv("GROQ_KEY_TPM", "0")) or int(getattr(config, "GROQ_TPM_LIMIT", 0) or 0)
    except Exception:
        key_tpm = int(getattr(config, "GROQ_TPM_LIMIT", 0) or 0)
    try:
        key_daily = int(os.getenv("GROQ_KEY_DAILY", "0")) or 0
    except Exception:
        key_daily = 0

    bud = TokenBudget(tpm_limit=key_tpm, daily_limit=key_daily)
    _per_key_budgets[key] = bud
    logger.debug(f"[TOKEN_BUDGET] Created per-key budget for {key[:8]} tpm={key_tpm} daily={key_daily}")
    return bud

logger = configure_logging()

# Use GROQ_URL from config so it can be overridden by env
GROQ_URL = config.GROQ_URL

from typing import List, Optional

# Configure module-level token budget from config
try:
    token_budget.tpm_limit = int(getattr(config, "GROQ_TPM_LIMIT", 0) or 0)
    # daily limit is not configured in config currently; keep default 0 (disabled)
except Exception:
    logger.debug("[TOKEN_BUDGET] Failed to set tpm_limit from config")


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


def _parse_retry_after(response_text: str) -> float:
    """Extract retry-after seconds from Groq 429 error message."""
    try:
        import json
        data = json.loads(response_text)
        msg = data.get("error", {}).get("message", "")
        # Parse "Please try again in 1.88s" or "Please try again in 664ms"
        import re
        match = re.search(r"try again in ([\d.]+)(s|ms)", msg, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            unit = match.group(2).lower()
            return val if unit == "s" else val / 1000.0
    except Exception:
        pass
    return 0.0


def groq_llm(messages, model: Optional[str] = None):
    """Call Groq with simple round-robin key failover.

    This will try each key in order and retry per key with backoff. Keys must
    be provided via `GROQ_API_KEYS` (comma-separated) or `GROQ_API_KEY`.
    
    Includes intelligent rate-limit handling:
    - Parses 429 retry-after hints from Groq error messages
    - Implements exponential backoff for rate-limited requests
    - Respects AI_RATE_LIMIT_RETRY_COUNT for 429-specific retries
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

    # Heuristic token estimate (simple chars->tokens): roughly 4 chars per token
    try:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = max(1, total_chars // 4)
    except Exception:
        estimated_tokens = None

    # If configured, prevent sending requests that clearly exceed a configured TPM limit
    if getattr(config, "GROQ_TPM_LIMIT", 0):
        try:
            tpm_limit = int(config.GROQ_TPM_LIMIT)
            if estimated_tokens and estimated_tokens > tpm_limit:
                raise RuntimeError(f"Estimated tokens {estimated_tokens} exceed configured GROQ_TPM_LIMIT={tpm_limit}")
        except Exception as e:
            logger.warning(f"[NYNEOS-AI-0.9.1.1] Pre-flight token check failed or blocked request: {e}")
            raise

    # Try each key in round-robin fashion with retries per key
    last_error = None
    rate_limit_retry_count = getattr(config, "AI_RATE_LIMIT_RETRY_COUNT", 3)
    
    # If we have an estimated token count, prefer keys that currently have capacity
    keys_to_try = []
    if estimated_tokens:
        for k in keys:
            kb = _get_key_budget(k)
            try:
                # treat tpm_limit==0 as unlimited
                if not kb.tpm_limit or (kb.current_tpm() + estimated_tokens) <= kb.tpm_limit:
                    keys_to_try.append(k)
            except Exception:
                # conservative: if we can't query, include the key
                keys_to_try.append(k)

    if not keys_to_try:
        # No key currently has capacity (or we couldn't estimate); fall back to trying all keys
        keys_to_try = keys

    for key in keys_to_try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, config.AI_RETRY_COUNT + 1):
            try:
                logger.info(f"[NYNEOS-AI-0.9.1.1] Attempt key_prefix={key[:8]} attempt={attempt} model={model} est_tokens={estimated_tokens}")

                # Wait for token budget capacity before making the call (if configured)
                try:
                    if estimated_tokens:
                            key_budget = _get_key_budget(key)
                            key_budget.wait_for_budget(estimated_tokens)
                except Exception:
                    logger.debug("[TOKEN_BUDGET] wait_for_budget raised, continuing to request")

                start = time.time()
                res = requests.post(
                    GROQ_URL,
                    json=payload,
                    headers=headers,
                    timeout=config.AI_TIMEOUT_SECONDS,
                )
                latency = round(time.time() - start, 2)

                if res.status_code == 200:
                    logger.info(f"[NYNEOS-AI-0.9.1.1] Success in {latency}s with key_prefix={key[:8]}")
                    data = res.json()
                    
                    # Record token usage for tracking
                    try:
                        usage = data.get("usage", {})
                        total_tokens = usage.get("total_tokens", 0)
                        if total_tokens > 0:
                            record_tokens(total_tokens)
                            # Update token budget accounting
                            try:
                                key_budget.consume(total_tokens)
                                logger.debug(f"[TOKEN_BUDGET] Consumed {total_tokens} tokens (current_tpm={key_budget.current_tpm()})")
                            except Exception:
                                logger.debug("[TOKEN_BUDGET] consume failed")
                            logger.debug(f"[TOKEN_TRACKER] Recorded {total_tokens} tokens from response")
                        else:
                            # Fallback: estimate if usage not in response
                            record_tokens(estimated_tokens)
                            try:
                                key_budget.consume(estimated_tokens)
                            except Exception:
                                pass
                            logger.debug(f"[TOKEN_TRACKER] Recorded estimated {estimated_tokens} tokens (no usage in response)")
                    except Exception as e:
                        logger.warning(f"[TOKEN_TRACKER] Failed to record tokens: {e}")
                    
                    return data["choices"][0]["message"]["content"]

                # Special handling for 429 rate-limit errors with retry-after hint
                if res.status_code == 429:
                    retry_after = _parse_retry_after(res.text)
                    logger.warning(f"[NYNEOS-AI-0.9.1.1] Rate limit 429 (key_prefix={key[:8]}): {res.text}")
                    
                    # Retry 429s with exponential backoff + parsed retry-after hint
                    for retry_idx in range(1, rate_limit_retry_count + 1):
                        wait_time = max(retry_after, retry_idx * 2.5)  # at least retry_after or exponential backoff
                        logger.info(f"[NYNEOS-AI-0.9.1.1] Rate-limit retry {retry_idx}/{rate_limit_retry_count} after {wait_time:.1f}s...")
                        # Use token_budget smart sleep to avoid fixed sleeps
                        try:
                               key_budget.smart_sleep(wait_time)
                        except Exception:
                            time.sleep(wait_time)
                        
                        try:
                            res_retry = requests.post(
                                GROQ_URL,
                                json=payload,
                                headers=headers,
                                timeout=config.AI_TIMEOUT_SECONDS,
                            )
                            if res_retry.status_code == 200:
                                logger.info(f"[NYNEOS-AI-0.9.1.1] Success after rate-limit retry {retry_idx}")
                                data = res_retry.json()

                                # Record token usage for tracking
                                try:
                                    usage = data.get("usage", {})
                                    total_tokens = usage.get("total_tokens", 0)
                                    if total_tokens > 0:
                                        record_tokens(total_tokens)
                                        try:
                                                    key_budget.consume(total_tokens)
                                        except Exception:
                                            pass
                                    else:
                                        record_tokens(estimated_tokens)
                                        try:
                                                    key_budget.consume(estimated_tokens)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    logger.warning(f"[TOKEN_TRACKER] Failed to record tokens on retry: {e}")

                                return data["choices"][0]["message"]["content"]
                            elif res_retry.status_code == 429:
                                retry_after = _parse_retry_after(res_retry.text)
                                logger.warning(f"[NYNEOS-AI-0.9.1.1] Still rate-limited (retry {retry_idx}): {res_retry.text}")
                                continue
                            else:
                                logger.warning(f"[NYNEOS-AI-0.9.1.1] Error {res_retry.status_code} on retry {retry_idx}: {res_retry.text}")
                                break
                        except Exception as e_retry:
                            logger.error(f"[NYNEOS-AI-0.9.1.1] Exception on rate-limit retry {retry_idx}: {e_retry}")
                            break
                    
                    last_error = RuntimeError(f"HTTP 429: {res.text}")
                else:
                    # Non-429 errors: log and continue standard retry
                    logger.warning(f"[NYNEOS-AI-0.9.1.1] Error {res.status_code} (key_prefix={key[:8]}): {res.text}")
                    last_error = RuntimeError(f"HTTP {res.status_code}: {res.text}")

            except Exception as e:
                logger.error(f"[NYNEOS-AI-0.9.1.1] Exception key_prefix={key[:8]} attempt {attempt}: {e}")
                last_error = e

            # backoff before retrying the same key (for non-429 errors)
            try:
                 kb = _get_key_budget(key)
                 kb.smart_sleep(attempt * 1.2)
            except Exception:
                time.sleep(attempt * 1.2)

        # if this key exhausted attempts, move to next key
        logger.info(f"[NYNEOS-AI-0.9.1.1] Moving to next key after exhausting attempts for key_prefix={key[:8]}")

    # all keys exhausted
    raise RuntimeError(f"[NYNEOS-AI-0.9.1.1] API failed after trying all keys: {last_error}")