"""mistake_note_search honours max_response_chars (2026-11-04).

The handler drops whole notes from the end until the JSON fits, always keeps the
first (best) note, and reports what was cut in a `truncated` block.
"""
import json

from mcp_memory_service.server.handlers.mistake_notes import _dump, _limit_notes


def _result(n, size=400):
    return {"notes": [{"content_hash": f"h{i}", "content": "x" * size, "similarity": 0.9 - i / 100}
                      for i in range(n)], "count": n}


def test_no_limit_returns_everything():
    r = _result(5)
    assert _limit_notes(r, 0) is r


def test_result_under_limit_is_unchanged():
    r = _result(2)
    assert _limit_notes(r, len(_dump(r)) + 10) is r


def test_drops_whole_notes_from_the_end():
    r = _result(5)
    out = _limit_notes(r, 1300)
    assert len(_dump(out)) <= 1300
    assert [n["content_hash"] for n in out["notes"]] == ["h0", "h1"]
    assert out["count"] == 2
    assert out["truncated"] == {"shown": 2, "total": 5, "max_response_chars": 1300}
    # Notes are never cut mid-content.
    assert all(n["content"] == "x" * 400 for n in out["notes"])


def test_first_note_is_kept_even_over_limit():
    r = _result(3, size=2000)
    out = _limit_notes(r, 500)
    assert [n["content_hash"] for n in out["notes"]] == ["h0"]
    assert out["truncated"]["first_note_exceeds_limit"] is True


def test_explicit_argument_and_env_default(monkeypatch):
    from mcp_memory_service.server.handlers.memory import _get_max_response_chars
    monkeypatch.delenv("MCP_MAX_RESPONSE_CHARS", raising=False)
    assert _get_max_response_chars({"max_response_chars": 1500}) == 1500
    assert _get_max_response_chars({}) == 0
    monkeypatch.setenv("MCP_MAX_RESPONSE_CHARS", "8000")
    assert _get_max_response_chars({}) == 8000
