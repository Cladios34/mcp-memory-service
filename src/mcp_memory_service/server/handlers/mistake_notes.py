"""
Mistake notes handler functions for MCP server.

Provides mistake pattern tracking operations:
- mistake_note_add: Record a mistake pattern for error replay
- mistake_note_search: Search mistake notes by semantic similarity
- mistake_note_update: Update fields of an existing mistake note
- mistake_note_delete: Delete a mistake note by content hash
"""

import json
import logging
from typing import List

from mcp import types

from .memory import _get_max_response_chars

logger = logging.getLogger(__name__)


def _dump(result: dict) -> str:
    return json.dumps(result, default=str, ensure_ascii=False, separators=(",", ":"))


def _to_text(result: dict) -> List[types.TextContent]:
    """Serialize a handler result for the model that reads it.

    WHY(2026-11-04): ensure_ascii=False keeps accented letters as themselves instead of
    six-character \\uXXXX escapes, and compact separators drop the indentation. Measured on
    five real searches: about 11 % fewer characters for the same content.
    """
    return [types.TextContent(type="text", text=_dump(result))]


async def handle_mistake_note_add(server, arguments: dict) -> List[types.TextContent]:
    """Record a mistake pattern for error replay."""
    await server._ensure_storage_initialized()
    result = await server.memory_service.mistake_note_add(
        error_pattern=arguments.get("error_pattern", ""),
        context_signature=arguments.get("context_signature", ""),
        incorrect_action=arguments.get("incorrect_action", ""),
        correct_action=arguments.get("correct_action", ""),
    )
    return _to_text(result)


async def handle_mistake_note_search(server, arguments: dict) -> List[types.TextContent]:
    """Search mistake notes by semantic similarity."""
    await server._ensure_storage_initialized()
    result = await server.memory_service.mistake_note_search(
        query=arguments.get("query", ""),
        limit=arguments.get("limit", 5),
    )
    return _to_text(_limit_notes(result, _get_max_response_chars(arguments)))


def _limit_notes(result: dict, max_chars: int) -> dict:
    """Drop trailing notes until the serialized result fits in max_chars (0 = no limit).

    WHY(2026-11-04): memory_search honoured max_response_chars but mistake_note_search
    ignored it, so a caller had no way to bound the payload. Notes are dropped whole,
    best match kept first, and the first note is always returned: a lesson cut in the
    middle is worse than a slightly long answer. The `truncated` block says what was cut.
    """
    notes = result.get("notes") or []
    if max_chars <= 0 or not notes or len(_dump(result)) <= max_chars:
        return result
    total = len(notes)
    kept = list(notes)
    trimmed = dict(result)
    while len(kept) > 1:
        kept.pop()
        trimmed = {**result, "notes": kept, "count": len(kept),
                   "truncated": {"shown": len(kept), "total": total, "max_response_chars": max_chars}}
        if len(_dump(trimmed)) <= max_chars:
            return trimmed
    return {**result, "notes": kept, "count": len(kept),
            "truncated": {"shown": len(kept), "total": total, "max_response_chars": max_chars,
                          "first_note_exceeds_limit": len(_dump(trimmed)) > max_chars}}


async def handle_mistake_note_update(server, arguments: dict) -> List[types.TextContent]:
    """Update fields of an existing mistake note."""
    await server._ensure_storage_initialized()
    result = await server.memory_service.mistake_note_update(
        content_hash=arguments.get("content_hash", ""),
        failure_count=arguments.get("failure_count"),
        error_pattern=arguments.get("error_pattern"),
        context_signature=arguments.get("context_signature"),
        incorrect_action=arguments.get("incorrect_action"),
        correct_action=arguments.get("correct_action"),
    )
    return _to_text(result)


async def handle_mistake_note_delete(server, arguments: dict) -> List[types.TextContent]:
    """Delete a mistake note by content hash."""
    await server._ensure_storage_initialized()
    result = await server.memory_service.mistake_note_delete(
        content_hash=arguments.get("content_hash", ""),
    )
    return _to_text(result)
