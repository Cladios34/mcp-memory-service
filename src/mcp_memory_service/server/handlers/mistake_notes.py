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

logger = logging.getLogger(__name__)


def _to_text(result: dict) -> List[types.TextContent]:
    """Serialize a handler result for the model that reads it.

    WHY(2026-11-04): ensure_ascii=False keeps accented letters as themselves instead of
    six-character \\uXXXX escapes, and compact separators drop the indentation. Measured on
    five real searches: about 11 % fewer characters for the same content.
    """
    return [types.TextContent(type="text", text=json.dumps(
        result, default=str, ensure_ascii=False, separators=(",", ":")))]


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
    return _to_text(result)


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
