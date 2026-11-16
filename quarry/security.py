"""API-key auth, per-key rate limiting, and log redaction."""
from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def tenant_for_key(api_key: str, keys: dict[str, str]) -> str | None:
    """Map a presented key to its tenant. Only SHA-256 digests are ever configured or stored."""
    digest = hash_key(api_key)
    found = None
    for stored, tenant in keys.items():            # constant-time compare against each digest
        if hmac.compare_digest(stored, digest):
            found = tenant
    return found


class TokenBucket:
    """In-process token bucket per key. For multi-instance deployments back this
    with Memorystore; on Cloud Run with max-instances=1 it is exact."""

    def __init__(self, rate_per_minute: int, burst: int | None = None, clock=time.monotonic):
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst or max(1, rate_per_minute // 4))
        self._clock = clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            tokens, last = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - 1.0, now)
            return True


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def redact(text: str) -> str:
    """Scrub obvious PII before a question is written to logs."""
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[number]", text)
    return _PHONE.sub("[phone]", text)
