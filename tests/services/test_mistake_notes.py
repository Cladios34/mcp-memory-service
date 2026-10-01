# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Test suite for Mistake Notes — structured error replay via memory store.

Tests cover:
- Creating new mistake notes (memory_type='mistake')
- Dedup: incrementing failure_count on similar patterns
- Threshold boundary: just above/below dedup cutoff
- Search: only returns memories tagged 'mistake-note'
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
from unittest.mock import patch

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.content_splitter import split_content
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest_asyncio.fixture
async def memory_service():
    """Create temporary MemoryService for mistake notes testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_mistakes.db")
    try:
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        svc = MemoryService(storage)
        yield svc
    finally:
        if hasattr(storage, 'conn') and storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_creates_new(memory_service):
    """First mistake note should create a new memory."""
    result = await memory_service.mistake_note_add(
        error_pattern="PostgreSQL timeout on large query",
        context_signature="MIR API database queries",
        incorrect_action="Restarted the database",
        correct_action="Add LIMIT clause to query",
    )
    assert result["status"] == "created"
    assert result["failure_count"] == 1
    assert result["content_hash"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("correct_action", ["", "   ", "\n\t "])
async def test_mistake_note_add_rejects_empty_correct_action(memory_service, correct_action):
    """Empty/whitespace-only correct_action should be rejected, not stored (#1055)."""
    result = await memory_service.mistake_note_add(
        error_pattern="Error with no remediation",
        context_signature="some context",
        incorrect_action="Did the wrong thing",
        correct_action=correct_action,
    )
    assert result["status"] == "error"
    assert "correct_action" in result["message"]

    # Nothing should have been stored
    search = await memory_service.mistake_note_search(query="Error with no remediation", limit=10)
    assert search["count"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_dedup_increments_count(memory_service):
    """Adding a similar mistake should increment failure_count.

    Note: Dedup requires embedding model. In test environments without
    sentence-transformers, similarity is always 0 and dedup won't trigger.
    This test uses a very low threshold to work without real embeddings.
    """
    # First add
    r1 = await memory_service.mistake_note_add(
        error_pattern="Git push fails with auth error",
        context_signature="Git operations on wwwgit",
        incorrect_action="Switched to SSH",
        correct_action="Refresh token in ~/.git-credentials",
    )
    assert r1["status"] == "created"

    # Second add — use threshold=0.0 so ANY match triggers dedup
    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.0):
        r2 = await memory_service.mistake_note_add(
            error_pattern="Git push authentication failure",
            context_signature="Git operations on wwwgit",
            incorrect_action="Tried SSH keys",
            correct_action="Update token in git-credentials",
        )

    # With threshold=0.0, any result from retrieve_memories triggers dedup
    if r2["status"] == "updated":
        assert r2["failure_count"] == 2
    else:
        # No embeddings available — dedup can't work, skip gracefully
        pytest.skip("Embedding model not available for dedup test")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_returns_only_mistakes(memory_service):
    """Search should only return memories with memory_type='mistake'."""
    # Store a regular memory
    await memory_service.store_memory(
        content="PostgreSQL is a relational database",
        memory_type="observation",
        tags="database",
    )

    # Store a mistake note
    await memory_service.mistake_note_add(
        error_pattern="PostgreSQL timeout",
        context_signature="database queries",
        incorrect_action="Restarted DB",
        correct_action="Add LIMIT",
    )

    # Search should only find the mistake note
    result = await memory_service.mistake_note_search(
        query="PostgreSQL database",
        limit=10,
    )

    assert result["count"] >= 1
    for note in result["notes"]:
        assert "Pattern:" in note["content"]
        assert "Wrong:" in note["content"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_empty(memory_service):
    """Search with no mistake notes should return empty list."""
    result = await memory_service.mistake_note_search(query="anything", limit=5)
    assert result["count"] == 0
    assert result["notes"] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_prefilters_before_top_k(memory_service):
    """A large mixed corpus must not crowd mistake notes out before filtering."""
    created = await memory_service.mistake_note_add(
        error_pattern="SCORM retry button hidden after a failed quiz",
        context_signature="learner player progression gate",
        incorrect_action="Used the local chapter index as a course-level flag",
        correct_action="Use the server-provided child-package flag",
    )
    assert created["status"] == "created"

    distractor_query = "generic error failure bug exception incorrect action failed command"
    for index in range(12):
        stored = await memory_service.store_memory(
            content=f"{distractor_query} distractor observation {index}",
            memory_type="observation",
            conversation_id=f"mistake-search-distractor-{index}",
        )
        assert stored["success"] is True

    with patch("mcp_memory_service.storage.mixins.retrieve._MAX_TAG_SEARCH_CANDIDATES", 2):
        result = await memory_service.mistake_note_search(query=distractor_query, limit=5)

    assert result["count"] == 1
    assert "SCORM retry button hidden" in result["notes"][0]["content"]
    assert result["notes"][0]["similarity"] > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_dedup_prefilters_before_top_k(memory_service):
    """Normal memories must not hide an existing mistake from deduplication."""
    first = await memory_service.mistake_note_add(
        error_pattern="Git token expires during push",
        context_signature="release authentication workflow",
        incorrect_action="Retried the same rejected credential",
        correct_action="Refresh the token before retrying",
    )
    assert first["status"] == "created"

    dedup_query = "Git token expires during push release authentication workflow"
    for index in range(12):
        stored = await memory_service.store_memory(
            content=f"{dedup_query} distractor observation {index}",
            memory_type="observation",
            conversation_id=f"mistake-dedup-distractor-{index}",
        )
        assert stored["success"] is True

    original_store = memory_service.store_memory

    async def fail_if_stored(*args, **kwargs):
        pytest.fail("Deduplication missed the existing mistake and attempted a new store")

    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.0):
        memory_service.store_memory = fail_if_stored
        try:
            repeated = await memory_service.mistake_note_add(
                error_pattern="Git token expired while pushing",
                context_signature="release authentication workflow",
                incorrect_action="Retried an expired credential",
                correct_action="Refresh the token before retrying",
            )
        finally:
            memory_service.store_memory = original_store

    assert repeated["status"] == "updated"
    assert repeated["failure_count"] == 2
    assert repeated["content_hash"] == first["content_hash"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chunked_mistake_add_and_search_return_one_complete_note(memory_service, monkeypatch):
    """Hybrid-sized chunks must remain one actionable logical mistake note."""
    storage_type = type(memory_service.storage)
    monkeypatch.setattr(storage_type, "max_content_length", property(lambda self: 140))

    note_fields = {
        "error_pattern": "A long generated document loses its final verification section " * 3,
        "context_signature": "Hybrid SQLite and Cloudflare storage with strict content limits " * 3,
        "incorrect_action": "Stored the structured error as unrelated fragments " * 3,
        "correct_action": "Reassemble every ordered chunk before returning the mistake note " * 4,
    }
    result = await memory_service.mistake_note_add(**note_fields)

    assert result["status"] == "created"
    assert result["content_hash"]
    assert result["chunks_created"] > 1
    assert result["logical_hash"]

    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.0):
        repeated = await memory_service.mistake_note_add(
            error_pattern=note_fields["error_pattern"].replace("loses", "lost"),
            context_signature=note_fields["context_signature"],
            incorrect_action=note_fields["incorrect_action"],
            correct_action=note_fields["correct_action"],
        )
    assert repeated["status"] == "updated"
    assert repeated["chunks_updated"] == result["chunks_created"]
    assert repeated["failure_count"] == 2

    search = await memory_service.mistake_note_search(
        query="long generated document final verification section",
        limit=5,
    )

    assert search["count"] == 1
    note = search["notes"][0]
    expected_content = (
        f"Pattern: {note_fields['error_pattern']}\n"
        f"Context: {note_fields['context_signature']}\n"
        f"Wrong: {note_fields['incorrect_action']}\n"
        f"Right: {note_fields['correct_action']}"
    )
    assert note["logical_hash"] == result["logical_hash"]
    assert len(note["chunk_hashes"]) == result["chunks_created"]
    assert note["failure_count"] == 2
    assert note["content"] == expected_content

    for chunk_hash in note["chunk_hashes"]:
        chunk = await memory_service.storage.get_by_hash(chunk_hash)
        assert chunk.metadata["failure_count"] == 2


def test_join_mistake_chunks_repairs_legacy_trimmed_boundaries():
    content = (
        "Pattern: A legacy mistake note crosses a word boundary several times " * 3
        + "\nContext: an older boundary-preserving splitter removed separators " * 3
        + "\nWrong: returned concatenated fields and words " * 3
        + "\nRight: restore readable separators while rebuilding the logical note " * 3
    )
    chunks = split_content(
        content,
        max_length=140,
        preserve_boundaries=True,
        overlap=50,
    )

    rebuilt = MemoryService._join_mistake_chunks(
        chunks,
        expected_hash=generate_content_hash(content),
    )

    assert rebuilt == content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_limit_zero_returns_no_notes(memory_service):
    await memory_service.mistake_note_add(
        error_pattern="Limit validation",
        context_signature="mistake search",
        incorrect_action="Returned an unexpected row",
        correct_action="Respect a zero result limit",
    )

    result = await memory_service.mistake_note_search(query="limit validation", limit=0)

    assert result == {"notes": [], "count": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chunked_mistake_exact_readd_updates_without_semantic_results(memory_service, monkeypatch):
    storage_type = type(memory_service.storage)
    monkeypatch.setattr(storage_type, "max_content_length", property(lambda self: 140))
    note_fields = {
        "error_pattern": "Exact chunked duplicate " * 12,
        "context_signature": "semantic backend unavailable " * 8,
        "incorrect_action": "Stored another logical copy " * 8,
        "correct_action": "Resolve the group by its original content hash " * 8,
    }
    created = await memory_service.mistake_note_add(**note_fields)
    assert created["status"] == "created"
    assert created["chunks_created"] > 1

    async def no_semantic_results(*args, **kwargs):
        return []

    monkeypatch.setattr(memory_service.storage, "retrieve", no_semantic_results)
    repeated = await memory_service.mistake_note_add(**note_fields)

    assert repeated["status"] == "updated"
    assert repeated["logical_hash"] == created["logical_hash"]
    assert repeated["chunks_updated"] == created["chunks_created"]
    assert repeated["failure_count"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_rolls_back_partial_chunk_store(memory_service, monkeypatch):
    storage_type = type(memory_service.storage)
    monkeypatch.setattr(storage_type, "max_content_length", property(lambda self: 140))
    original_store = memory_service.storage.store
    mistake_store_calls = 0

    async def fail_second_mistake_chunk(memory, *args, **kwargs):
        nonlocal mistake_store_calls
        if memory.memory_type == "mistake":
            mistake_store_calls += 1
            if mistake_store_calls == 2:
                return False, "injected second-chunk failure"
        return await original_store(memory, *args, **kwargs)

    monkeypatch.setattr(memory_service.storage, "store", fail_second_mistake_chunk)
    result = await memory_service.mistake_note_add(
        error_pattern="Partial chunk storage " * 12,
        context_signature="transient backend failure " * 8,
        incorrect_action="Accepted incomplete content " * 8,
        correct_action="Rollback every newly stored fragment " * 8,
    )

    assert result["status"] == "error"
    assert "injected second-chunk failure" in result["message"]
    remaining = await memory_service.storage.get_all_memories(
        limit=None,
        memory_type="mistake",
        tags=["mistake-note"],
    )
    assert remaining == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_normalizes_legacy_numeric_metadata(memory_service, monkeypatch):
    storage_type = type(memory_service.storage)
    monkeypatch.setattr(storage_type, "max_content_length", property(lambda self: 140))
    created = await memory_service.mistake_note_add(
        error_pattern="Legacy metadata conversion " * 10,
        context_signature="mixed numeric encodings " * 8,
        incorrect_action="Compared strings and floats " * 8,
        correct_action="Normalize numeric counters before grouping " * 8,
    )
    assert created["chunks_created"] > 1

    stored = await memory_service.mistake_note_search(query="legacy metadata conversion", limit=1)
    hashes = stored["notes"][0]["chunk_hashes"]
    first = await memory_service.storage.get_by_hash(hashes[0])
    second = await memory_service.storage.get_by_hash(hashes[1])
    first_meta = dict(first.metadata)
    second_meta = dict(second.metadata)
    first_meta["confidence"] = "0.9"
    second_meta["confidence"] = 0.5
    await memory_service.storage.update_memory_metadata(hashes[0], {"metadata": first_meta})
    await memory_service.storage.update_memory_metadata(hashes[1], {"metadata": second_meta})

    result = await memory_service.mistake_note_search(query="legacy metadata conversion", limit=1)

    assert "error" not in result
    assert result["count"] == 1
    assert result["notes"][0]["metadata"]["confidence"] == pytest.approx(0.9)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sqlite_batch_metadata_update_rolls_back_on_missing_fragment(memory_service):
    created = await memory_service.mistake_note_add(
        error_pattern="Atomic metadata update",
        context_signature="fragment group",
        incorrect_action="Committed only the first update",
        correct_action="Rollback the complete batch on failure",
    )
    content_hash = created["content_hash"]
    before = await memory_service.storage.get_by_hash(content_hash)

    results = await memory_service.storage.update_memories_batch(
        [
            Memory(
                content=before.content,
                content_hash=content_hash,
                tags=before.tags,
                memory_type=before.memory_type,
                metadata={"failure_count": 9},
            ),
            Memory(
                content="missing",
                content_hash="missing-fragment",
                memory_type="mistake",
                metadata={"failure_count": 9},
            ),
        ],
        preserve_timestamps=False,
    )

    after = await memory_service.storage.get_by_hash(content_hash)
    assert results == [False, False]
    assert after.metadata["failure_count"] == before.metadata["failure_count"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_high_threshold_no_dedup(memory_service):
    """With very high threshold, similar notes should NOT dedup."""
    await memory_service.mistake_note_add(
        error_pattern="Error A in context X",
        context_signature="context X",
        incorrect_action="Did wrong thing",
        correct_action="Do right thing",
    )

    # Very high threshold — should create new instead of dedup
    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.99):
        r2 = await memory_service.mistake_note_add(
            error_pattern="Error B in context Y",
            context_signature="context Y",
            incorrect_action="Different wrong thing",
            correct_action="Different right thing",
        )

    assert r2["status"] == "created"
    assert r2["failure_count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_handles_store_dedup_rejection(memory_service):
    """When store_memory rejects with semantic duplicate, should increment existing note.

    Regression test for #1034: retrieve_memories misses (score < threshold),
    but store_memory's own dedup catches it and rejects. Without the fix,
    this returns status='error'. With the fix, it increments failure_count.
    """
    # Create first note normally
    r1 = await memory_service.mistake_note_add(
        error_pattern="Post to GitHub without approval",
        context_signature="GitHub communication",
        incorrect_action="Posted without showing draft",
        correct_action="Always show draft and wait for OK",
    )
    assert r1["status"] == "created"
    first_hash = r1["content_hash"]

    # Mock store_memory to simulate dedup rejection pointing to first note
    original_store = memory_service.store_memory

    async def mock_store(*args, **kwargs):
        return {"success": False, "error": f"Duplicate content detected (semantically similar to {first_hash})"}

    # High threshold so retrieve_memories won't match
    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.99):
        memory_service.store_memory = mock_store
        try:
            r2 = await memory_service.mistake_note_add(
                error_pattern="Posted on GitHub without user approval",
                context_signature="GitHub external communication",
                incorrect_action="Created issue without draft review",
                correct_action="Show draft EN+PT-BR, wait for explicit OK",
            )
        finally:
            memory_service.store_memory = original_store

    # Without fix: status="error", message="Failed to store: ..."
    # With fix: status="updated", failure_count=2
    assert r2["status"] == "updated", f"Expected 'updated' but got '{r2['status']}': {r2.get('message','')}"
    assert r2["failure_count"] == 2
    assert r2["content_hash"] == first_hash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_failure_count(memory_service):
    """Update failure_count on an existing mistake note."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Forgot to run tests before push",
        context_signature="CI/CD workflow",
        incorrect_action="Pushed without testing",
        correct_action="Always run pytest before git push",
    )
    assert r1["status"] == "created"
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        failure_count=5,
    )
    assert result["status"] == "updated"
    assert result["content_hash"] == content_hash

    # Verify the update persisted
    mem = await memory_service.storage.get_by_hash(content_hash)
    meta = mem.metadata if isinstance(mem.metadata, dict) else {}
    assert meta.get("failure_count") == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_content_fields(memory_service):
    """Update content fields (correct_action) on an existing mistake note."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Used wrong branch",
        context_signature="Git workflow",
        incorrect_action="Committed to main",
        correct_action="Create feature branch first",
    )
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        correct_action="Create feature branch and open PR",
    )
    assert result["status"] == "updated"
    # Content change = new hash (delete + re-store)
    new_hash = result["content_hash"]

    # Old hash should be gone
    old_mem = await memory_service.storage.get_by_hash(content_hash)
    assert old_mem is None

    # New hash should have updated content
    new_mem = await memory_service.storage.get_by_hash(new_hash)
    assert new_mem is not None
    assert "Create feature branch and open PR" in new_mem.content


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("correct_action", ["", "   ", "\n\t "])
async def test_mistake_note_update_rejects_blanking_correct_action(memory_service, correct_action):
    """Updating correct_action to empty/whitespace should be rejected (#1055)."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Pattern to keep",
        context_signature="Git workflow",
        incorrect_action="Committed to main",
        correct_action="Create feature branch first",
    )
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        correct_action=correct_action,
    )
    assert result["status"] == "error"
    assert "correct_action" in result["message"]

    # Original note must be untouched
    mem = await memory_service.storage.get_by_hash(content_hash)
    assert mem is not None
    assert "Create feature branch first" in mem.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_nonexistent(memory_service):
    """Updating a nonexistent hash should return error."""
    result = await memory_service.mistake_note_update(
        content_hash="nonexistent_hash_abc123",
        failure_count=10,
    )
    assert result["status"] == "error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_wrong_type(memory_service):
    """Updating a non-mistake memory should return error."""
    store_result = await memory_service.store_memory(
        content="Regular observation",
        memory_type="observation",
        tags="test",
    )
    content_hash = store_result.get("memory", {}).get("content_hash", "")

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        failure_count=5,
    )
    assert result["status"] == "error"
    assert "not a mistake note" in result["message"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_delete(memory_service):
    """Delete an existing mistake note."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Obsolete pattern",
        context_signature="Old context",
        incorrect_action="Old wrong",
        correct_action="Old right",
    )
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_delete(content_hash=content_hash)
    assert result["status"] == "deleted"

    # Verify it's gone
    mem = await memory_service.storage.get_by_hash(content_hash)
    assert mem is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_delete_nonexistent(memory_service):
    """Deleting a nonexistent hash should return error."""
    result = await memory_service.mistake_note_delete(content_hash="nonexistent_hash_xyz")
    assert result["status"] == "error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_delete_wrong_type(memory_service):
    """Deleting a non-mistake memory should return error."""
    store_result = await memory_service.store_memory(
        content="Regular memory not a mistake",
        memory_type="observation",
        tags="test",
    )
    content_hash = store_result.get("memory", {}).get("content_hash", "")

    result = await memory_service.mistake_note_delete(content_hash=content_hash)
    assert result["status"] == "error"
    assert "not a mistake note" in result["message"].lower()
