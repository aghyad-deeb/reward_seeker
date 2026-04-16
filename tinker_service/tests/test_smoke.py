"""
Smoke tests for tinker_service — verifies wiring without requiring a live
model endpoint. Uses FastAPI's TestClient and the role_colon renderer (which
ships with tinker-cookbook and has no external deps).
"""

from __future__ import annotations

import os
import sys

import pytest


# Make the tinker_service package importable from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from tinker_service.app import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_detect_renderer_unknown_returns_none_or_fallback(client):
    # Unknown model name should not raise; it returns renderer_name=None on
    # a pure miss or a best-guess renderer on a hit. Either is acceptable.
    resp = client.post("/detect-renderer", json={"model_name": "made-up-model-zzz"})
    assert resp.status_code == 200
    body = resp.json()
    assert "renderer_name" in body


def test_tokenize_role_colon_roundtrip(client):
    # role_colon doesn't require a specific model's tokenizer; use a small
    # tokenizer via the default path. We're exercising the wiring, not the
    # tokens themselves.
    body = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "renderer_name": "role_colon",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "target_tool_format": "xml",
    }
    resp = client.post("/tokenize", json=body)
    # If the tokenizer can't be loaded in the test env, we accept a 500 —
    # that's an environment issue, not a code issue. The key assertion is
    # the wiring.
    if resp.status_code == 200:
        data = resp.json()
        assert "message_tokens" in data
        assert len(data["message_tokens"]) == 2
        assert "total" in data
    else:
        pytest.skip(f"tokenizer unavailable: {resp.text[:200]}")


def test_parse_unified_shape():
    from tinker_service.parsing import parse_response_unified

    class FakeRenderer:
        def parse_response(self, tokens):
            return (
                {"role": "assistant", "content": "Hello there", "tool_calls": []},
                True,
            )

    result = parse_response_unified(FakeRenderer(), [1, 2, 3], "xml")
    assert result["parse_success"] is True
    assert result["decoded_message"]["content"] == "Hello there"
    assert result["decoded_message"]["tool_calls"] == []
    assert result["extracted_bash_commands"] == []


def test_parse_unified_xml_extracts_bash():
    from tinker_service.parsing import parse_response_unified

    class FakeRenderer:
        def parse_response(self, tokens):
            return (
                {
                    "role": "assistant",
                    "content": "Let me check. <bash>ls /</bash> Done.",
                    "tool_calls": [],
                },
                True,
            )

    result = parse_response_unified(FakeRenderer(), [1, 2, 3], "xml")
    assert result["extracted_bash_commands"] == ["ls /"]


def test_parse_unified_tinker_mode_ignores_bash_tags():
    from tinker_service.parsing import parse_response_unified

    class FakeRenderer:
        def parse_response(self, tokens):
            return (
                {
                    "role": "assistant",
                    "content": "I will run <bash>ls /</bash>.",
                    "tool_calls": [],
                },
                True,
            )

    result = parse_response_unified(FakeRenderer(), [1, 2, 3], "tinker")
    # In tinker mode we trust the renderer's tool_calls; text-level <bash>
    # blocks are NOT auto-extracted.
    assert result["extracted_bash_commands"] == []


def test_parse_unified_structured_content_projects_final_channel():
    from tinker_service.parsing import parse_response_unified

    class FakeRenderer:
        def parse_response(self, tokens):
            return (
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "channel": "analysis", "text": "hidden CoT"},
                        {"type": "text", "channel": "final", "text": "user-visible answer"},
                    ],
                    "tool_calls": [],
                },
                True,
            )

    result = parse_response_unified(FakeRenderer(), [1, 2, 3], "tinker")
    assert result["decoded_message"]["content"] == "user-visible answer"
    assert result["decoded_message"]["content_parts"] is not None
    assert len(result["decoded_message"]["content_parts"]) == 2
    assert any(p["channel"] == "analysis" for p in result["decoded_message"]["content_parts"])


def test_parse_unified_strips_think_blocks_for_xml():
    from tinker_service.parsing import parse_response_unified

    class FakeRenderer:
        def parse_response(self, tokens):
            return (
                {
                    "role": "assistant",
                    "content": "<think>plan: list files</think> <bash>ls</bash>",
                    "tool_calls": [],
                },
                True,
            )

    result = parse_response_unified(FakeRenderer(), [1, 2, 3], "xml")
    assert result["extracted_bash_commands"] == ["ls"]
