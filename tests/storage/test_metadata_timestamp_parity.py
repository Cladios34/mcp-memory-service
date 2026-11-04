"""Timestamp semantics must agree between sqlite-vec and Cloudflare D1."""

import json
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.quality.metadata_codec import decompress_metadata_from_sync
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.hybrid import BackgroundSyncService, HybridMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


def timestamp_fields(db, content_hash):
    """Read persisted timestamps rather than inspect how an update was built."""
    return tuple(
        db.execute(
            "SELECT created_at, created_at_iso, updated_at, updated_at_iso "
            "FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    )


@pytest_asyncio.fixture(params=["sqlite_vec", "cloudflare"])
async def metadata_storage(request, temp_db_path, monkeypatch):
    """Use real storage methods, replacing only D1's HTTP boundary with SQLite."""
    if request.param == "sqlite_vec":
        storage = SqliteVecMemoryStorage(f"{temp_db_path}/metadata.db")
        await storage.initialize()
        db = storage.conn
        metadata_column = "metadata"
    else:
        storage = CloudflareStorage(
            api_token="test-token",
            account_id="test-account",
            vectorize_index="test-index",
            d1_database_id="test-db",
        )
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        metadata_column = "metadata_json"

        async def query_d1(method, url, **kwargs):
            """Execute the backend's SQL and return the D1 response envelope."""
            assert method == "POST" and url == f"{storage.d1_url}/query"
            payload = kwargs["json"]
            try:
                previous_changes = db.total_changes
                if payload["sql"].count(";") > 1:
                    db.executescript(payload["sql"])
                    rows, row_id = [], 0
                else:
                    cursor = db.execute(payload["sql"], payload.get("params", []))
                    rows = [dict(row) for row in cursor.fetchall()]
                    row_id = cursor.lastrowid
                db.commit()
                result = {
                    "success": True,
                    "result": [
                        {
                            "results": rows,
                            "meta": {
                                "last_row_id": row_id,
                                "changes": db.total_changes - previous_changes,
                            },
                        }
                    ],
                }
            except sqlite3.Error as error:
                result = {"success": False, "errors": [str(error)]}
            return httpx.Response(200, json=result)

        monkeypatch.setattr(storage, "_retry_request", query_d1)
        await storage._initialize_d1_schema()

    created_at, updated_at = 1700000000.0, 1710000000.0
    memory = Memory(
        content="Memory whose metadata will be annotated",
        content_hash=generate_content_hash("Memory whose metadata will be annotated"),
        tags=["original"],
        memory_type="observation",
        metadata={"source": "original"},
        created_at=created_at,
        created_at_iso=datetime.fromtimestamp(created_at, timezone.utc).isoformat(),
        updated_at=updated_at,
        updated_at_iso=datetime.fromtimestamp(updated_at, timezone.utc).isoformat(),
    )
    try:
        if request.param == "sqlite_vec":
            success, message = await storage.store(memory)
            assert success, message
        else:
            await storage._store_d1_memory(
                memory, "test-vector", len(memory.content), None, memory.content
            )
        yield storage, db, memory.content_hash, metadata_column
    finally:
        await storage.close()
        if request.param == "cloudflare":
            db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_default", [True, False])
async def test_metadata_only_preserves_timestamps(metadata_storage, use_default):
    """Annotations must persist without making an old memory appear recent."""
    storage, db, content_hash, metadata_column = metadata_storage
    before = timestamp_fields(db, content_hash)
    options = {} if use_default else {"preserve_timestamps": True}

    success, message = await storage.update_memory_metadata(
        content_hash, {"metadata": {"annotated": True}}, **options
    )

    assert success, message
    assert timestamp_fields(db, content_hash) == before
    metadata = db.execute(
        f"SELECT {metadata_column} FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()[0]
    assert json.loads(metadata)["annotated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_tags,updates",
    [
        (["original"], {"tags": ["original"]}),
        (["original"], {"memory_type": "observation"}),
        (["original"], {"tags": ["original"], "memory_type": "observation"}),
        (["alpha", "beta"], {"tags": ["beta", "alpha"]}),
        (["original"], {"tags": ["original", "original"]}),
    ],
)
async def test_unchanged_structural_fields_preserve_timestamps(
    metadata_storage, stored_tags, updates
):
    """Repeated fields, including equivalent tag sets, are metadata-only writes."""
    storage, db, content_hash, metadata_column = metadata_storage
    if stored_tags != ["original"]:
        success, message = await storage.update_memory_metadata(
            content_hash, {"tags": stored_tags}, preserve_timestamps=False
        )
        assert success, message
    before = timestamp_fields(db, content_hash)

    success, message = await storage.update_memory_metadata(
        content_hash,
        {**updates, "metadata": {"annotated": True}},
        preserve_timestamps=True,
    )

    assert success, message
    assert timestamp_fields(db, content_hash) == before
    metadata = db.execute(
        f"SELECT {metadata_column} FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()[0]
    assert json.loads(metadata)["annotated"] is True
    memory = await storage.get_by_hash(content_hash)
    assert memory is not None
    assert set(memory.tags) == set(stored_tags)
    assert memory.memory_type == "observation"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"tags": []},
        {"memory_type": "decision"},
        {"content": "replacement"},
        {"content": "Memory whose metadata will be annotated"},
    ],
)
async def test_structural_updates_advance_updated_at(metadata_storage, updates):
    """The preservation flag still allows structural fields to advance time."""
    storage, db, content_hash, _ = metadata_storage
    before = timestamp_fields(db, content_hash)
    started_at = time.time()

    success, message = await storage.update_memory_metadata(
        content_hash, updates, preserve_timestamps=True
    )

    assert success, message
    after = timestamp_fields(db, content_hash)
    assert after[:2] == before[:2]
    assert started_at <= after[2] <= time.time()
    assert after[3] != before[3]


@pytest.mark.asyncio
async def test_metadata_without_preservation_advances_time(metadata_storage):
    """Callers can still opt into a fresh update time for metadata changes."""
    storage, db, content_hash, _ = metadata_storage
    before = timestamp_fields(db, content_hash)
    started_at = time.time()

    success, message = await storage.update_memory_metadata(
        content_hash, {"metadata": {"annotated": True}}, preserve_timestamps=False
    )

    assert success, message
    after = timestamp_fields(db, content_hash)
    assert after[:2] == before[:2]
    assert started_at <= after[2] <= time.time()
    assert after[3] != before[3]


@pytest.mark.asyncio
@pytest.mark.parametrize("preserve_timestamps", [True, False])
async def test_explicit_sync_timestamps(metadata_storage, preserve_timestamps):
    """Source timestamps are applied only when preservation is disabled."""
    storage, db, content_hash, _ = metadata_storage
    before = timestamp_fields(db, content_hash)
    source = (
        1600000000.0,
        "2020-09-13T12:26:40Z",
        1650000000.0,
        "2022-04-15T05:20:00Z",
    )
    updates = dict(
        zip(("created_at", "created_at_iso", "updated_at", "updated_at_iso"), source)
    )
    updates["metadata"] = {"synced": True}

    success, message = await storage.update_memory_metadata(
        content_hash, updates, preserve_timestamps=preserve_timestamps
    )

    assert success, message
    assert timestamp_fields(db, content_hash) == (
        before if preserve_timestamps else source
    )


@pytest.mark.asyncio
async def test_cloudflare_reports_failed_metadata_update(monkeypatch):
    """A D1 network failure must remain a failed update, not a successful no-op."""
    storage = CloudflareStorage("test-token", "test-account", "test-index", "test-db")
    monkeypatch.setattr(
        storage,
        "_retry_request",
        AsyncMock(side_effect=httpx.ConnectError("D1 unavailable")),
    )

    success, message = await storage.update_memory_metadata(
        "test-hash", {"metadata": {"annotated": True}}, preserve_timestamps=True
    )

    assert not success
    assert "D1 unavailable" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_storage", ["cloudflare"], indirect=True)
@pytest.mark.parametrize("preserve_timestamps", [True, False])
@pytest.mark.parametrize(
    "change,remote_update,structural_change",
    [
        ({}, False, False),
        ({"tags": ["changed"]}, False, True),
        ({"memory_type": "decision"}, False, True),
        ({}, True, False),
        ({"tags": ["original", "original"]}, False, False),
    ],
)
async def test_hybrid_batch_sync_timestamp_preservation(
    metadata_storage,
    temp_db_path,
    preserve_timestamps,
    change,
    remote_update,
    structural_change,
):
    """Only metadata annotations preserve time, including a newer cloud time."""
    secondary, secondary_db, content_hash, metadata_column = metadata_storage
    hybrid = HybridMemoryStorage(f"{temp_db_path}/hybrid-primary.db")
    await hybrid.primary.initialize()
    hybrid.secondary = secondary
    hybrid.sync_service = BackgroundSyncService(hybrid.primary, secondary)
    try:
        memory = await secondary.get_by_hash(content_hash)
        assert memory is not None
        success, message = await hybrid.primary.store(memory)
        assert success, message
        before = timestamp_fields(hybrid.primary.conn, content_hash)
        memory.metadata["relevance_score"] = 0.75
        for key, value in change.items():
            setattr(memory, key, value)

        statements = []
        hybrid.primary.conn.set_trace_callback(statements.append)
        assert await hybrid.update_memories_batch(
            [memory], preserve_timestamps=preserve_timestamps
        ) == [True]
        hybrid.primary.conn.set_trace_callback(None)
        assert (
            sum(
                "SELECT" in sql.upper() and "FROM MEMORIES" in sql.upper()
                for sql in statements
            )
            == 1
        )

        primary_times = timestamp_fields(hybrid.primary.conn, content_hash)
        if preserve_timestamps and not structural_change:
            assert primary_times == before
        else:
            assert primary_times[:2] == before[:2]
            assert primary_times[2] > before[2]

        operation = hybrid.sync_service.operation_queue.get_nowait()
        if remote_update:
            remote_updated_at = time.time() - 30.0
            success, message = await secondary.update_memory_metadata(
                content_hash,
                {
                    "metadata": {"other_device": True},
                    "updated_at": remote_updated_at,
                    "updated_at_iso": datetime.fromtimestamp(
                        remote_updated_at, timezone.utc
                    ).isoformat(),
                },
                preserve_timestamps=False,
            )
            assert success, message
        remote_times = timestamp_fields(secondary_db, content_hash)
        requests = []
        query_d1 = secondary._retry_request

        async def record_request(method, url, **kwargs):
            requests.append(kwargs["json"]["sql"])
            return await query_d1(method, url, **kwargs)

        secondary._retry_request = record_request
        await hybrid.sync_service._process_single_operation(operation)
        hybrid.sync_service.operation_queue.task_done()
        if not structural_change:
            assert len(requests) == 1

        secondary_times = timestamp_fields(secondary_db, content_hash)
        assert secondary_times[:2] == primary_times[:2]
        if preserve_timestamps and not structural_change:
            assert secondary_times == remote_times
        else:
            assert secondary_times[2] >= max(primary_times[2], remote_times[2])
            assert secondary_times[3] != before[3]
        metadata = secondary_db.execute(
            f"SELECT {metadata_column} FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()[0]
        assert (
            decompress_metadata_from_sync(json.loads(metadata))["relevance_score"]
            == 0.75
        )
        synced = await secondary.get_by_hash(content_hash)
        assert set(synced.tags) == set(memory.tags)
        assert synced.memory_type == memory.memory_type
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_deleted_memory_is_not_updated(metadata_storage):
    """A metadata update must not mutate a tombstoned memory or its tags."""
    storage, db, content_hash, metadata_column = metadata_storage
    before = timestamp_fields(db, content_hash)
    metadata = db.execute(
        f"SELECT {metadata_column} FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()[0]
    db.execute(
        "UPDATE memories SET deleted_at = ? WHERE content_hash = ?",
        (time.time(), content_hash),
    )
    db.commit()

    success, _ = await storage.update_memory_metadata(
        content_hash,
        {"metadata": {"annotated": True}, "tags": [], "memory_type": "decision"},
    )

    assert not success
    assert timestamp_fields(db, content_hash) == before
    assert (
        db.execute(
            f"SELECT {metadata_column} FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()[0]
        == metadata
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_storage", ["cloudflare"], indirect=True)
@pytest.mark.parametrize("deleted", [False, True])
async def test_cloudflare_missing_record_uses_returned_rows(metadata_storage, deleted):
    """Empty RETURNING rows fail even when D1 omits its changes metadata."""
    storage, db, content_hash, _ = metadata_storage
    if deleted:
        db.execute(
            "UPDATE memories SET deleted_at = ? WHERE content_hash = ?",
            (time.time(), content_hash),
        )
        db.commit()
    else:
        content_hash = "missing-memory"
    requests = []
    query_d1 = storage._retry_request

    async def query_without_meta(method, url, **kwargs):
        requests.append(kwargs["json"]["sql"])
        response = await query_d1(method, url, **kwargs)
        result = response.json()
        result["result"][0].pop("meta", None)
        return httpx.Response(200, json=result)

    storage._retry_request = query_without_meta
    success, message = await storage.update_memory_metadata(
        content_hash, {"metadata": {"annotated": True}, "tags": []}
    )

    assert not success
    assert "not found" in message.lower()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_storage", ["cloudflare"], indirect=True)
@pytest.mark.parametrize("field", ["tags", "memory_type"])
async def test_concurrent_structural_change_advances_time(
    metadata_storage, monkeypatch, field
):
    """Compare against the values present when D1 executes the actual UPDATE."""
    storage, db, content_hash, _ = metadata_storage
    query_d1 = storage._retry_request
    concurrent_time = time.time() - 30.0
    changed = False

    async def change_before_update(method, url, **kwargs):
        nonlocal changed
        if not changed and kwargs["json"]["sql"].startswith("UPDATE memories SET"):
            changed = True
            if field == "memory_type":
                db.execute(
                    "UPDATE memories SET memory_type = 'decision' WHERE content_hash = ?",
                    (content_hash,),
                )
            else:
                db.execute("INSERT INTO tags(name) VALUES ('other-device')")
                db.execute(
                    "DELETE FROM memory_tags WHERE memory_id = (SELECT id FROM memories WHERE content_hash = ?)",
                    (content_hash,),
                )
                db.execute(
                    "INSERT INTO memory_tags(memory_id, tag_id) SELECT m.id, t.id FROM memories m, tags t WHERE m.content_hash = ? AND t.name = 'other-device'",
                    (content_hash,),
                )
            db.execute(
                "UPDATE memories SET updated_at = ?, updated_at_iso = ? WHERE content_hash = ?",
                (
                    concurrent_time,
                    datetime.fromtimestamp(concurrent_time, timezone.utc).isoformat(),
                    content_hash,
                ),
            )
            db.commit()
        return await query_d1(method, url, **kwargs)

    monkeypatch.setattr(storage, "_retry_request", change_before_update)
    success, message = await storage.update_memory_metadata(
        content_hash,
        {
            "metadata": {"annotated": True},
            "tags": ["original"],
            "memory_type": "observation",
        },
    )

    assert success, message
    assert changed
    assert timestamp_fields(db, content_hash)[2] > concurrent_time


@pytest.mark.asyncio
async def test_hybrid_single_update_keeps_full_metadata_in_cloudflare(metadata_storage, temp_db_path):
    """A partial metadata update must not erase the other keys in D1 (2026-11-04).

    CloudflareStorage.update_memory_metadata replaces metadata_json. The hybrid layer used
    to forward only the delta, so one quality-score update stripped source / chunk fields
    from D1, and a device filled from D1 inherited the stripped copy (8 214 memories).
    """
    secondary, secondary_db, content_hash, metadata_column = metadata_storage
    if metadata_column != "metadata_json":
        pytest.skip("Cloudflare-only behaviour")
    hybrid = HybridMemoryStorage(f"{temp_db_path}/hybrid-primary-full.db")
    await hybrid.primary.initialize()
    hybrid.secondary = secondary
    hybrid.sync_service = BackgroundSyncService(hybrid.primary, secondary)
    try:
        memory = await secondary.get_by_hash(content_hash)
        memory.metadata.update({"chunk_index": 1, "total_chunks": 2, "original_hash": "abc",
                                "access_count": 7, "access_queries": [{"query": "q"}]})
        success, message = await hybrid.primary.store(memory)
        assert success, message

        success, message = await hybrid.update_memory_metadata(
            content_hash, {"metadata": {"quality_score": 0.9}}, preserve_timestamps=True
        )
        assert success, message
        operation = hybrid.sync_service.operation_queue.get_nowait()
        await hybrid.sync_service._process_single_operation(operation)
        hybrid.sync_service.operation_queue.task_done()

        stored = decompress_metadata_from_sync(json.loads(secondary_db.execute(
            "SELECT metadata_json FROM memories WHERE content_hash = ?", (content_hash,)
        ).fetchone()[0]))
        assert stored["quality_score"] == 0.9
        assert stored["source"] == "original"
        assert (stored["chunk_index"], stored["total_chunks"], stored["original_hash"]) == (1, 2, "abc")
        # Per-device access tracking is not shipped to the cloud.
        assert "access_count" not in stored and "access_queries" not in stored
    finally:
        await hybrid.primary.close()


async def _hybrid_on(metadata_storage, temp_db_path, name, extra_metadata=None):
    secondary, secondary_db, content_hash, metadata_column = metadata_storage
    if metadata_column != "metadata_json":
        pytest.skip("Cloudflare-only behaviour")
    hybrid = HybridMemoryStorage(f"{temp_db_path}/{name}.db")
    await hybrid.primary.initialize()
    hybrid.secondary = secondary
    hybrid.sync_service = BackgroundSyncService(hybrid.primary, secondary)
    memory = await secondary.get_by_hash(content_hash)
    memory.metadata.update(extra_metadata or {})
    success, message = await hybrid.primary.store(memory)
    assert success, message
    return hybrid, secondary, secondary_db, content_hash


def _d1_metadata(secondary_db, content_hash):
    return decompress_metadata_from_sync(json.loads(secondary_db.execute(
        "SELECT metadata_json FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()[0]))


@pytest.mark.asyncio
async def test_retried_old_update_does_not_overwrite_newer_state(metadata_storage, temp_db_path):
    """Full metadata is read at SEND time: an operation replayed after a newer one ships
    today's state, not the snapshot of its enqueue time (review finding, 2026-11-04)."""
    hybrid, secondary, secondary_db, content_hash = await _hybrid_on(metadata_storage, temp_db_path, "retry")
    try:
        await hybrid.update_memory_metadata(content_hash, {"metadata": {"step": "A"}})
        op_a = hybrid.sync_service.operation_queue.get_nowait()
        await hybrid.update_memory_metadata(content_hash, {"metadata": {"step": "B", "extra": 1}})
        op_b = hybrid.sync_service.operation_queue.get_nowait()
        await hybrid.sync_service._process_single_operation(op_b)
        await hybrid.sync_service._process_single_operation(op_a)  # replayed late
        stored = _d1_metadata(secondary_db, content_hash)
        assert stored["step"] == "B" and stored["extra"] == 1 and stored["source"] == "original"
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_update_for_deleted_row_is_skipped(metadata_storage, temp_db_path):
    hybrid, secondary, secondary_db, content_hash = await _hybrid_on(metadata_storage, temp_db_path, "deleted")
    try:
        await hybrid.update_memory_metadata(content_hash, {"metadata": {"x": 1}})
        op = hybrid.sync_service.operation_queue.get_nowait()
        await hybrid.primary.delete(content_hash)
        assert await hybrid.sync_service._current_secondary_updates(op) is None
        before = secondary_db.execute("SELECT metadata_json FROM memories WHERE content_hash = ?", (content_hash,)).fetchone()[0]
        await hybrid.sync_service._process_single_operation(op)
        after = secondary_db.execute("SELECT metadata_json FROM memories WHERE content_hash = ?", (content_hash,)).fetchone()[0]
        assert after == before  # nothing sent, and above all no delta replacing D1
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_unreadable_live_row_raises_instead_of_sending_delta(metadata_storage, temp_db_path, monkeypatch):
    """get_by_hash returns None on read errors: never fall back to the delta."""
    hybrid, secondary, secondary_db, content_hash = await _hybrid_on(metadata_storage, temp_db_path, "unreadable")
    try:
        await hybrid.update_memory_metadata(content_hash, {"metadata": {"x": 1}})
        op = hybrid.sync_service.operation_queue.get_nowait()

        async def broken_read(*args, **kwargs):
            return None
        monkeypatch.setattr(hybrid.primary, "get_by_hash", broken_read)
        with pytest.raises(RuntimeError, match="Could not read the primary row"):
            await hybrid.sync_service._current_secondary_updates(op)
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_oversized_metadata_still_ships_tags(metadata_storage, temp_db_path, monkeypatch):
    """Over 9.5 KB AFTER compression: metadata is held back, tags/type still go out."""
    hybrid, secondary, secondary_db, content_hash = await _hybrid_on(
        metadata_storage, temp_db_path, "oversize", {"blob": "x" * 12000})
    try:
        await hybrid.update_memory_metadata(content_hash, {"tags": ["new-tag"], "metadata": {"y": 2}})
        op = hybrid.sync_service.operation_queue.get_nowait()
        sent = []
        real = secondary.update_memory_metadata

        async def spy(h, updates, preserve_timestamps=True):
            sent.append(updates)
            return await real(h, updates, preserve_timestamps=preserve_timestamps)
        monkeypatch.setattr(secondary, "update_memory_metadata", spy)
        await hybrid.sync_service._process_single_operation(op)
        assert sent and "metadata" not in sent[0] and sent[0]["tags"] == ["new-tag"]
    finally:
        await hybrid.primary.close()
