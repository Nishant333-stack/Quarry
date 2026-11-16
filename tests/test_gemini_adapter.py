"""Drives the real google-genai SDK against a local stub of the Gemini REST API.

No network or key needed; this pins the request shape we send (system
instruction, JSON mode, task types, output dimensionality) and how we read
usage and embeddings back.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from quarry.llm import GeminiEmbedder, GeminiLLM

genai = pytest.importorskip("google.genai")


@pytest.fixture()
def gemini(monkeypatch):
    seen: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body))
            if "mbed" in self.path:
                out = {"embeddings": [{"values": [3.0, 4.0, 0.0]} for _ in body["requests"]]}
            else:
                out = {
                    "candidates": [{"content": {"role": "model", "parts": [{"text": '{"route": "docs"}'}]},
                                    "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 8,
                                      "thoughtsTokenCount": 12, "totalTokenCount": 140},
                }
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    client = genai.Client(
        api_key="test-key",
        http_options=genai.types.HttpOptions(base_url=f"http://127.0.0.1:{server.server_port}"),
    )
    yield client, seen
    server.shutdown()


def test_generate_sends_system_and_json_mode_and_meters_usage(gemini):
    client, seen = gemini
    res = GeminiLLM("gemini-3.8-flash", client=client).complete("SYSTEM RULES", "hello", json_mode=True)
    path, body = seen[0]
    assert path.endswith("models/gemini-3.8-flash:generateContent")
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM RULES"
    assert body["generationConfig"] == {"temperature": 0.0, "responseMimeType": "application/json"}
    assert res.text == '{"route": "docs"}'
    assert (res.usage.input_tokens, res.usage.output_tokens) == (120, 20)      # thinking billed as output
    assert res.usage.cost_usd == pytest.approx((120 * 0.75 + 20 * 3.75) / 1e6)


def test_embeddings_are_batched_typed_and_normalised(gemini):
    client, seen = gemini
    emb = GeminiEmbedder("gemini-embedding-001", dim=3, batch=2, client=client)
    vecs = emb.embed(["a", "b", "c"], kind="document")
    assert [len(body["requests"]) for _, body in seen] == [2, 1]
    first = seen[0][1]["requests"][0]
    assert first["taskType"] == "RETRIEVAL_DOCUMENT" and first["outputDimensionality"] == 3
    assert vecs == [[0.6, 0.8, 0.0]] * 3
    emb.embed(["q"], kind="query")
    assert seen[-1][1]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
