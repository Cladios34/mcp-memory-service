"""
Tests for P3: Session Legacy (RFC Self-Service Memory Intelligence).

The commit_session_legacy tool records end-of-session learnings
from ephemeral agents as structured observations.
"""

import json
import pytest
from mcp import types
from mcp_memory_service.server import MemoryServer


class TestSessionLegacy:
    """P3: commit_session_legacy tool."""

    @pytest.mark.asyncio
    async def test_commit_session_legacy_basic(self):
        """Basic session legacy with minimal required fields."""
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "session_id": "sess-test-001",
            "agent_id": "kiro",
            "task_summary": "Refactored auth module to use JWT",
            "outcome": "success",
        })
        assert isinstance(result, list)
        assert len(result) > 0
        text = result[0].text
        data = json.loads(text)
        assert data["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_commit_session_legacy_with_errors(self):
        """Session legacy with errors creates/updates mistake notes."""
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "session_id": "sess-test-002",
            "agent_id": "kiro",
            "task_summary": "Deploy to production",
            "outcome": "partial",
            "errors": [
                {
                    "tool": "shell",
                    "error": "Permission denied on /var/www",
                    "count": 3,
                    "severity": "high",
                    "resolution": "Added chown before deploy"
                }
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data.get("mistake_notes_updated", 0) >= 1

    @pytest.mark.asyncio
    async def test_commit_session_legacy_with_user_corrections(self):
        """User corrections are stored as high-priority observations."""
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "session_id": "sess-test-003",
            "agent_id": "kiro",
            "task_summary": "Password hashing implementation",
            "outcome": "success",
            "user_corrections": [
                {"original": "Used bcrypt", "corrected_to": "Use argon2"}
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data.get("observations_created", 0) >= 1

    @pytest.mark.asyncio
    async def test_commit_session_legacy_with_decisions(self):
        """Decisions are stored as observation type=decision."""
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "session_id": "sess-test-004",
            "agent_id": "kiro",
            "task_summary": "Setup project tooling",
            "outcome": "success",
            "decisions": [
                {"what": "Used pnpm instead of npm", "why": "Faster installs, strict deps"}
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data.get("observations_created", 0) >= 1

    @pytest.mark.asyncio
    async def test_commit_session_legacy_missing_required_fields(self):
        """Missing required fields should return error."""
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "task_summary": "Something",
            # Missing session_id, agent_id, outcome
        })
        text = result[0].text.lower()
        assert "error" in text

    @pytest.mark.asyncio
    async def test_commit_session_legacy_tool_registered(self):
        """The tool must be listed in available tools."""
        server = MemoryServer()
        tools = await server.handle_list_tools()
        tool_names = [t.name for t in tools]
        assert "commit_session_legacy" in tool_names


class TestSessionLegacyMistakeNoteContract:
    """2026-11-04: mistake_note_add refuses blank key fields; session_end must adapt."""

    @pytest.mark.asyncio
    async def test_error_without_tool_is_stored_with_fallback_context(self):
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "session_id": "sess-test-notool", "agent_id": "kiro",
            "task_summary": "Tool-less error", "outcome": "partial",
            "errors": [{"error": "Unique tool-less failure 7f3a", "resolution": "Retry with backoff"}],
        })
        data = json.loads(result[0].text)
        assert data["mistake_notes_updated"] == 1
        found = await server.memory_service.mistake_note_search(query="Unique tool-less failure 7f3a", limit=3)
        assert any("sess-test-notool (tool not recorded)" in n["content"] for n in found["notes"])

    @pytest.mark.asyncio
    async def test_rejected_note_is_not_counted(self):
        server = MemoryServer()
        result = await server.handle_call_tool("commit_session_legacy", {
            "session_id": "sess-test-blank", "agent_id": "kiro",
            "task_summary": "Blank error", "outcome": "partial",
            "errors": [{"tool": "shell", "error": "", "resolution": "Something"}],
        })
        data = json.loads(result[0].text)
        assert data["mistake_notes_updated"] == 0
