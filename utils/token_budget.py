"""
Token budget helper for Groq API calls.

Provides a simple minute-windowed token accounting and a wait helper
that sleeps only as long as needed to respect TPM limits.

This module is synchronous to keep integration simple with existing
sync client code. It uses a conservative 1-second granularity when
waiting for capacity.
"""
import time
import threading
import logging

logger = logging.getLogger(__name__)


class TokenBudget:
    def __init__(self, tpm_limit: int = 0, daily_limit: int = 0):
        # 0 disables enforcement
        self.tpm_limit = int(tpm_limit or 0)
        self.daily_limit = int(daily_limit or 0)

        # minute_usage: list of (timestamp, tokens)
        self.minute_usage = []
        self.daily_usage = 0
        self.lock = threading.Lock()

    def _prune(self):
        now = time.time()
        # keep only entries within the last 60s
        self.minute_usage = [(t, tok) for t, tok in self.minute_usage if now - t < 60]

    def current_tpm(self) -> int:
        with self.lock:
            self._prune()
            return sum(tok for _, tok in self.minute_usage)

    def consume(self, tokens: int):
        """Record token consumption immediately."""
        if tokens is None:
            return
        with self.lock:
            now = time.time()
            self.minute_usage.append((now, int(tokens)))
            if self.daily_limit:
                self.daily_usage += int(tokens)

    def wait_for_budget(self, required_tokens: int = 0, buffer: float = 0.05):
        """
        Block until there is capacity for `required_tokens` in the current minute window.

        If tpm_limit is 0 (disabled), returns immediately.
        Adds a small buffer to avoid edge races.
        """
        if not self.tpm_limit or required_tokens <= 0:
            return

        # Busy-wait with sleeps until enough tokens free up
        while True:
            with self.lock:
                self._prune()
                current = sum(tok for _, tok in self.minute_usage)
                # if enough capacity, return
                if current + required_tokens <= self.tpm_limit:
                    return

                # otherwise compute the earliest expiry to wait for
                if not self.minute_usage:
                    wait = 1.0
                else:
                    oldest_ts = min(t for t, _ in self.minute_usage)
                    now = time.time()
                    wait = max(0.5, 60 - (now - oldest_ts) + buffer)

            logger.info(f"[TOKEN_BUDGET] TPM limit reached ({current}/{self.tpm_limit}). Waiting {wait:.1f}s for capacity...")
            time.sleep(wait)

    def smart_sleep(self, seconds: float):
        """Sleep helper that logs and sleeps (keeps compatibility with previous sleeps)."""
        if not seconds or seconds <= 0:
            return
        logger.debug(f"[TOKEN_BUDGET] Sleeping {seconds:.2f}s")
        time.sleep(seconds)


# Export a module-level default budget (can be configured by clients)
default_budget = TokenBudget()
