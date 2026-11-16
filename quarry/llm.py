"""Model provider boundary.

Everything else in Quarry talks to two small protocols, ``LLM`` and ``Embedder``.
Gemini (Developer API or Vertex AI) is the production implementation; the
offline ones make the pipeline fully testable without network access.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

# USD per 1M tokens (input, output). Gemini 3.8 Flash is on introductory pricing
# through 2026-12-31 ($0.75 / $3.75) and doubles after; override with
# QUARRY_PRICE_IN / QUARRY_PRICE_OUT rather than editing code.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd
        return self

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResult:
    text: str
    usage: Usage
    latency_s: float


class LLM(Protocol):
    def complete(self, system: str, prompt: str, *, json_mode: bool = False) -> LLMResult: ...


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    import os

    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    p_in = float(os.environ.get("QUARRY_PRICE_IN", p_in))
    p_out = float(os.environ.get("QUARRY_PRICE_OUT", p_out))
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# --------------------------------------------------------------------------- #
# Gemini (google-genai SDK: works against both the Developer API and Vertex AI)
# --------------------------------------------------------------------------- #
def _client(client=None):
    if client is not None:
        return client
    from google import genai

    # Reads GEMINI_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI + GOOGLE_CLOUD_PROJECT/LOCATION.
    return genai.Client()


class GeminiLLM:
    def __init__(self, model: str, client=None, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = _client(client)

    def complete(self, system: str, prompt: str, *, json_mode: bool = False) -> LLMResult:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        dt = time.perf_counter() - t0
        um = resp.usage_metadata
        tin = (um.prompt_token_count or 0) if um else 0
        # thinking tokens are billed as output
        tout = ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0
        return LLMResult(
            text=resp.text or "",
            usage=Usage(tin, tout, 1, cost_usd(self.model, tin, tout)),
            latency_s=dt,
        )


class GeminiEmbedder:
    """Batched embeddings with asymmetric task types and Matryoshka truncation."""

    def __init__(self, model: str, dim: int, batch: int = 16, client=None):
        self.model, self.dim, self.batch = model, dim, batch
        self._client = _client(client)
        self.calls = 0

    def _call(self, texts: Sequence[str], task: str) -> list[list[float]]:
        from google.genai import types

        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        resp = self._client.models.embed_content(model=self.model, contents=list(texts), config=cfg)
        self.calls += 1
        # Truncated Matryoshka vectors are not unit length; cosine ops want them normalised.
        return [_l2(list(e.values)) for e in resp.embeddings]

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        i = 0
        while i < len(texts):
            group = texts[i : i + self.batch]
            try:
                out.extend(self._call(group, task))
            except Exception:
                if self.batch == 1:
                    raise
                # Some endpoints accept one input per request; degrade and keep going.
                self.batch = 1
                continue
            i += len(group)
        return out


# --------------------------------------------------------------------------- #
# Offline implementations (tests, CI, air-gapped demos)
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic feature-hashing embedder (signed word + bigram hashing).

    Not a semantic model. It exists so ingestion, storage, fusion and the API can
    be exercised end to end in CI with zero network calls and stable vectors.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.calls = 0
        self.texts_embedded = 0

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2(vec)

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._one(t) for t in texts]


class ScriptedLLM:
    """Test double: answers from a list, or from a function of (system, prompt)."""

    def __init__(self, script: Sequence[str] | Callable[[str, str], str]):
        self._script = script
        self._i = 0
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str, *, json_mode: bool = False) -> LLMResult:
        self.prompts.append((system, prompt))
        if callable(self._script):
            text = self._script(system, prompt)
        else:
            text = self._script[self._i]
            self._i += 1
        tin, tout = len(system + prompt) // 4, len(text) // 4
        return LLMResult(text=text, usage=Usage(tin, tout, 1, 0.0), latency_s=0.0)
