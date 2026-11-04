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
    assert out["truncated"] == {"shown": 1, "total": 3, "max_response_chars": 500}
    assert out["exceeds_limit"] is True


def test_single_over_long_note_has_no_truncated_block():
    """Nothing was dropped, so no `truncated` block; only the over-limit flag."""
    out = _limit_notes(_result(1, size=2000), 500)
    assert out["count"] == 1 and "truncated" not in out
    assert out["exceeds_limit"] is True


def test_bisection_matches_linear_scan():
    """The fitting prefix found by bisection is the longest one, for every limit."""
    r = _result(40, size=97)
    for limit in range(150, len(_dump(r)) + 50, 37):
        out = _limit_notes(r, limit)
        k = out["count"]
        assert k == 1 or len(_dump(out)) <= limit
        if k < 40:
            longer = {**r, "notes": r["notes"][:k + 1], "count": k + 1,
                      "truncated": {"shown": k + 1, "total": 40, "max_response_chars": limit}}
            assert k + 1 == 40 or len(_dump(longer)) > limit


def test_overflowing_limit_is_ignored(monkeypatch):
    """JSON 1e999 decodes to inf: int(inf) raised OverflowError and broke the call."""
    from mcp_memory_service.server.utils.response_limiter import get_max_response_chars
    monkeypatch.delenv("MCP_MAX_RESPONSE_CHARS", raising=False)
    assert get_max_response_chars({"max_response_chars": float("inf")}) == 0


def test_explicit_argument_and_env_default(monkeypatch):
    from mcp_memory_service.server.handlers.memory import _get_max_response_chars
    monkeypatch.delenv("MCP_MAX_RESPONSE_CHARS", raising=False)
    assert _get_max_response_chars({"max_response_chars": 1500}) == 1500
    assert _get_max_response_chars({}) == 0
    monkeypatch.setenv("MCP_MAX_RESPONSE_CHARS", "8000")
    assert _get_max_response_chars({}) == 8000
