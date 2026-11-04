"""Payload builder shared by the harvest rewriter, classifier and NLI cascade (2026-11-04)."""
import pytest

from mcp_memory_service.harvest.rewriter import build_openai_compatible_payload as build

MSG = [{"role": "user", "content": "x"}]


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "anthropic/claude-haiku-5-5", "claude-haiku-6"])
def test_claude_5_haiku_disables_thinking_without_temperature(model):
    p = build(model, MSG, 200, 0.3)
    assert "temperature" not in p
    assert p["thinking"] == {"type": "disabled"}
    assert p["max_tokens"] == 200


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-5-5", "claude-fable-5-1"])
def test_other_claude_5_get_headroom_and_no_thinking_field(model):
    p = build(model, MSG, 200, 0.3)
    assert "temperature" not in p and "thinking" not in p
    assert p["max_tokens"] == 1000


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-3-5-sonnet-latest",
    "gpt-4.1-mini", "gemini-3.5-flash", "llama-3.3-70b",
])
def test_other_models_keep_temperature(model):
    p = build(model, MSG, 200, 0.3)
    assert p["temperature"] == 0.3 and p["max_tokens"] == 200 and "thinking" not in p


def test_gpt5_branch_unchanged():
    p = build("gpt-5-mini", MSG, 200, 0.3)
    assert p["max_completion_tokens"] == 1000
    assert "temperature" not in p and "max_tokens" not in p
