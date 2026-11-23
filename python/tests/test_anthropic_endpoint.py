"""Smoke tests for the Anthropic-compatible /v1/messages endpoint.

These don't require any model to be present: with no model installed the
backend returns its "degraded" text, which is exactly what we want to assert
the *shape* of the translation (Anthropic message / SSE events) and that the
endpoint never 500s on a well-formed request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from edge_server.openai_server import app  # noqa: E402

client = TestClient(app)


def _nonstream_req(**over):
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 64,
        "system": "You are terse.",
        "messages": [{"role": "user", "content": "say hi"}],
    }
    body.update(over)
    return body


def test_messages_nonstream_shape():
    r = client.post("/v1/messages", json=_nonstream_req())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert isinstance(data["content"], list) and data["content"]
    assert data["content"][0]["type"] == "text"
    assert isinstance(data["content"][0]["text"], str)
    assert data["stop_reason"] in ("end_turn", "max_tokens", "stop_sequence")
    assert "input_tokens" in data["usage"] and "output_tokens" in data["usage"]
    assert data["id"].startswith("msg_")


def test_messages_accepts_x_api_key_header():
    # With no EDGE_API_KEY configured auth is disabled; the header must still
    # be accepted (not rejected) so Claude Code style clients work.
    r = client.post("/v1/messages", json=_nonstream_req(),
                    headers={"x-api-key": "anything"})
    assert r.status_code == 200


def test_messages_stream_events():
    body = _nonstream_req(stream=True)
    with client.stream("POST", "/v1/messages", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = []
        cur = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                cur = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                events.append((cur, payload))
    names = [e for e, _ in events]
    assert "message_start" in names
    assert "content_block_start" in names
    assert "content_block_stop" in names
    assert "message_delta" in names
    assert "message_stop" in names
    # First real event must be message_start with assistant role.
    first = next(p for e, p in events if e == "message_start")
    assert json.loads(first)["message"]["role"] == "assistant"


def test_messages_content_blocks_flatten():
    # Anthropic typed content blocks must be flattened to text by the helper.
    body = _nonstream_req(messages=[{
        "role": "user",
        "content": [{"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"}],
    }])
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200
    assert r.json()["type"] == "message"
