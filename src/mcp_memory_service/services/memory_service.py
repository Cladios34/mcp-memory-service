"""
Memory Service - Shared business logic for memory operations.

This service contains the shared business logic that was previously duplicated
between mcp_server.py and server.py. It provides a single source of truth for
all memory operations, eliminating the DRY violation and ensuring consistent behavior.
"""

import json
import logging
import math
import os
import re
import sys
from typing import Dict, List, Optional, Any, Tuple, Union

# Pydantic v2.12 requires typing_extensions.TypedDict on Python < 3.12
# See: https://errors.pydantic.dev/2.12/u/typed-dict-version
if sys.version_info < (3, 12):
    from typing_extensions import TypedDict, NotRequired
else:
    from typing import TypedDict
    try:
        from typing import NotRequired  # Python 3.11+
    except ImportError:
        from typing_extensions import NotRequired
from datetime import datetime

from ..config import (
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
    ENABLE_AUTO_SPLIT,
    MCP_QUALITY_BOOST_ENABLED
)
from ..storage.base import MemoryStorage
from ..models.memory import Memory
from ..plugins import PluginContext, PluginRegistry
from ..utils.content_splitter import split_content
from ..utils.hashing import generate_content_hash
from ..quality.async_scorer import async_scorer

logger = logging.getLogger(__name__)

def _sanitize_log_value(value: str) -> str:
    """Strip control characters from user-provided values before logging.

    Prevents log injection via newlines, carriage returns, or other
    control characters embedded in user input (CWE-117).
    """
    return value.replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


# Module-level constants for tag processing
_MAX_JSON_LENGTH = 4096  # 4KB limit for tag JSON to prevent DoS
_MAX_TAG_LENGTH = 100    # Maximum length for individual tags
_MAX_TAGS_PER_MEMORY = 100  # Maximum number of tags per memory
_MISTAKE_NOTE_TAG = "mistake-note"
_MISTAKE_SEARCH_CANDIDATE_FLOOR = 50
_MISTAKE_SEARCH_CANDIDATE_MULTIPLIER = 8


def normalize_tags(tags: Union[str, List[str], None]) -> List[str]:
    """
    Normalize tags to a consistent list format.

    Applies:
    - Whitespace stripping
    - Case normalization (lowercase)
    - Deduplication
    - Empty tag removal

    Args:
        tags: Tags in any supported format (None, string, comma-separated string, or list)

    Returns:
        List of tag strings, empty list if None or empty string
    """
    if tags is None:
        return []

    # Convert to list if string
    if isinstance(tags, str):
        if not tags.strip():
            return []
        # Handle JSON-encoded arrays (e.g. '["tag1", "tag2"]' from oneOf schemas)
        stripped = tags.strip()
        if stripped.startswith('['):
            # Prevent DoS via large/deeply nested JSON strings
            if len(stripped) > _MAX_JSON_LENGTH:
                logger.warning("Tag JSON string exceeds %s bytes, treating as literal string", _MAX_JSON_LENGTH)
                tags = [stripped]
            else:
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        tags = [str(t) for t in parsed]
                    else:
                        tags = [stripped]
                except (json.JSONDecodeError, ValueError, RecursionError):
                    # RecursionError from deeply nested JSON like [[[[...]]]]
                    tags = [stripped]
        # Split by comma if present, otherwise single tag
        elif ',' in tags:
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        else:
            tags = [tags.strip()]

    # Case-normalize, deduplicate, remove empties, sanitize
    normalized = []
    seen_lower = set()

    for tag in tags:
        if not isinstance(tag, str):
            continue

        tag_stripped = tag.strip()
        if not tag_stripped:
            continue

        # CRITICAL: Remove commas from tags to prevent LIKE-based search breakage.
        # Tags are stored comma-separated in SQLite, so commas within tags would
        # break the pattern matching logic: LIKE '%,tag,%'
        # Replace commas with hyphens to preserve semantic meaning.
        if ',' in tag_stripped:
            tag_stripped = tag_stripped.replace(',', '-')
            logger.debug(f"Removed comma from tag, replaced with hyphen: {_sanitize_log_value(tag_stripped)}")

        # Enforce maximum tag length to prevent abuse
        if len(tag_stripped) > _MAX_TAG_LENGTH:
            logger.warning(f"Tag exceeds {_MAX_TAG_LENGTH} characters, truncating: {_sanitize_log_value(tag_stripped[:50])}...")
            tag_stripped = tag_stripped[:_MAX_TAG_LENGTH]

        tag_lower = tag_stripped.lower()

        # Skip if already seen (case-insensitive deduplication)
        if tag_lower in seen_lower:
            continue

        seen_lower.add(tag_lower)
        normalized.append(tag_lower)  # Store in lowercase

    # Limit total number of tags to prevent DoS
    if len(normalized) > _MAX_TAGS_PER_MEMORY:
        logger.warning("Too many tags (%s), limiting to %s", len(normalized), _MAX_TAGS_PER_MEMORY)
        normalized = normalized[:_MAX_TAGS_PER_MEMORY]

    return normalized


class MemoryResult(TypedDict):
    """Type definition for memory operation results."""
    content: str
    content_hash: str
    tags: List[str]
    memory_type: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str
    created_at_iso: str
    updated_at_iso: str


# Store Memory Return Types
class StoreMemorySingleSuccess(TypedDict):
    """Return type for successful single memory storage."""
    success: bool
    memory: MemoryResult


class StoreMemoryChunkedSuccess(TypedDict):
    """Return type for successful chunked memory storage."""
    success: bool
    memories: List[MemoryResult]
    total_chunks: int
    original_hash: str
    failed_chunks: NotRequired[int]  # Number of chunks that failed to store


class StoreMemoryFailure(TypedDict):
    """Return type for failed memory storage."""
    success: bool
    error: str


# List Memories Return Types
class ListMemoriesSuccess(TypedDict):
    """Return type for successful memory listing."""
    memories: List[MemoryResult]
    page: int
    page_size: int
    total: int
    has_more: bool


class ListMemoriesError(TypedDict):
    """Return type for failed memory listing."""
    success: bool
    error: str
    memories: List[MemoryResult]
    page: int
    page_size: int


# Retrieve Memories Return Types
class RetrieveMemoriesSuccess(TypedDict):
    """Return type for successful memory retrieval."""
    memories: List[MemoryResult]
    query: str
    count: int


class RetrieveMemoriesError(TypedDict):
    """Return type for failed memory retrieval."""
    memories: List[MemoryResult]
    query: str
    error: str


# Search by Tag Return Types
class SearchByTagSuccess(TypedDict):
    """Return type for successful tag search."""
    memories: List[MemoryResult]
    tags: List[str]
    match_type: str
    count: int


class SearchByTagError(TypedDict):
    """Return type for failed tag search."""
    memories: List[MemoryResult]
    tags: List[str]
    error: str


# Delete Memory Return Types
class DeleteMemorySuccess(TypedDict):
    """Return type for successful memory deletion."""
    success: bool
    content_hash: str


class DeleteMemoryFailure(TypedDict):
    """Return type for failed memory deletion."""
    success: bool
    content_hash: str
    error: str


# Health Check Return Types
class HealthCheckSuccess(TypedDict, total=False):
    """Return type for successful health check."""
    healthy: bool
    storage_type: str
    total_memories: int
    last_updated: str
    # Additional fields from storage stats (marked as not required via total=False)


class HealthCheckFailure(TypedDict):
    """Return type for failed health check."""
    healthy: bool
    error: str


class MemoryService:
    """
    Shared service for memory operations with consistent business logic.

    This service centralizes all memory-related business logic to ensure
    consistent behavior across API endpoints and MCP tools, eliminating
    code duplication and potential inconsistencies.
    """

    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        self._plugin_registry = PluginRegistry(PluginContext(storage=storage, service=self))
        self._plugin_registry.discover_and_register()

    async def apply_retrieve_plugins(
        self, query: Optional[str], results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply retrieval plugins through the shared result boundary.

        Every retrieval entry point must call this after completing its own
        filtering and fallback work so plugins observe the final result set.
        """
        modified = await self._plugin_registry.fire(
            "on_retrieve", query or "", results
        )
        return modified if isinstance(modified, list) else results

    async def list_memories(
        self,
        page: int = 1,
        page_size: int = 10,
        tag: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        memory_type: Optional[str] = None,
        stale_days: Optional[int] = None,
        store: Optional[str] = "default",
        agent_id: Optional[str] = None,
    ) -> Union[ListMemoriesSuccess, ListMemoriesError]:
        """
        List memories with pagination and optional filtering.

        This method provides database-level filtering for optimal performance,
        avoiding the common anti-pattern of loading all records into memory.

        Args:
            page: Page number (1-based)
            page_size: Number of memories per page
            tag: Filter by specific tag (legacy, use tags instead)
            tags: Filter by list of tags
            tag_match: "any" to match ANY tag (OR), "all" to match ALL tags (AND)
            memory_type: Filter by memory type
            stale_days: Filter to memories not accessed in the last N days.
                Uses COALESCE(last_accessed, created_at) for memories never read.

        Returns:
            Dictionary with memories and pagination info
        """
        try:
            # Calculate offset for pagination
            offset = (page - 1) * page_size

            # Use database-level filtering for optimal performance
            # Support both legacy single tag and new tags list
            tags_list = tags if tags else ([tag] if tag else None)
            memories = await self.storage.get_all_memories(
                limit=page_size,
                offset=offset,
                memory_type=memory_type,
                tags=tags_list,
                tag_match=tag_match,
                stale_days=stale_days,
                store=store,
                agent_id=agent_id,
            )

            # Get accurate total count for pagination
            total = await self.storage.count_all_memories(
                memory_type=memory_type,
                tags=tags_list,
                tag_match=tag_match,
                stale_days=stale_days,
                store=store,
                agent_id=agent_id,
            )

            # Format results for API response
            results = []
            for memory in memories:
                results.append(self._format_memory_response(memory))

            total_pages = math.ceil(total / page_size) if page_size > 0 else 0

            return {
                "memories": results,
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_more": offset + page_size < total
            }

        except Exception as e:
            logger.exception(f"Unexpected error listing memories: {e}")
            return {
                "success": False,
                "error": f"Failed to list memories: {str(e)}",
                "memories": [],
                "page": page,
                "page_size": page_size,
                "total": 0,
                "total_pages": 0,
                "has_more": False
            }

    async def store_memory(
        self,
        content: str,
        tags: Union[str, List[str], None] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client_hostname: Optional[str] = None,
        conversation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        store: str = "default",
        skip_semantic_dedup: bool = False,
        require_all_chunks: bool = False,
    ) -> Union[StoreMemorySingleSuccess, StoreMemoryChunkedSuccess, StoreMemoryFailure]:
        """
        Store a new memory with validation and content processing.

        Accepts tags in multiple formats for maximum flexibility:
        - None → []
        - "tag1,tag2,tag3" → ["tag1", "tag2", "tag3"]
        - "single-tag" → ["single-tag"]
        - ["tag1", "tag2"] → ["tag1", "tag2"]

        Args:
            content: The memory content
            tags: Optional tags for the memory (string, comma-separated string, or list)
            memory_type: Optional memory type classification
            metadata: Optional additional metadata (can also contain tags)
            client_hostname: Optional client hostname for source tagging
            conversation_id: Optional conversation identifier. When supplied, semantic
                deduplication is skipped so all turns of the same conversation
                can be saved independently. Exact hash dedup is always preserved.
            skip_semantic_dedup: Bypass cross-memory semantic deduplication. Intended
                for callers that already perform a narrower, type-aware deduplication.
            require_all_chunks: Roll back newly stored chunks if any chunk fails.
                Use for logical records that are invalid when only partially stored.

        Returns:
            Dictionary with operation result
        """
        try:
            # Normalize tags from parameter (handles all formats)
            final_tags = normalize_tags(tags)

            # Extract and normalize metadata.tags if present
            final_metadata = dict(metadata) if metadata else {}
            if metadata and "tags" in metadata:
                metadata_tags = normalize_tags(metadata.get("tags"))
                # Merge with parameter tags (normalize_tags already deduplicates)
                final_tags = normalize_tags(final_tags + metadata_tags)  # Re-normalize after merge

            # Strip keys that have dedicated Memory fields to prevent
            # them leaking through **self.metadata in to_dict()
            for key in ("tags", "type"):
                final_metadata.pop(key, None)

            # Apply hostname tagging if provided (for consistent source tracking)
            if client_hostname:
                source_tag = f"source:{client_hostname}"
                if source_tag not in final_tags:
                    final_tags.append(source_tag)
                final_metadata["hostname"] = client_hostname

            # Store conversation_id in metadata for future grouping/retrieval
            skip_dedup = skip_semantic_dedup or bool(conversation_id) or (memory_type == "session")
            if conversation_id:
                final_metadata["conversation_id"] = conversation_id

            # RFC #1100: author identity. Precedence: explicit arg > MCP_AGENT_ID
            # env > agent_id already present in the caller's metadata (the path
            # harvest/bootstrap/commit_session use) > unset (null = unknown).
            resolved_agent_id = (
                agent_id
                or os.environ.get("MCP_AGENT_ID")
                or final_metadata.get("agent_id")
            )
            if resolved_agent_id:
                final_metadata["agent_id"] = resolved_agent_id

            # Generate content hash for deduplication
            content_hash = generate_content_hash(content)

            # Process content if auto-splitting is enabled and content exceeds max length
            max_length = self.storage.max_content_length
            if ENABLE_AUTO_SPLIT and max_length and len(content) > max_length:
                # Split content into chunks
                chunks = split_content(
                    content,
                    max_length=max_length,
                    # Boundary-preserving splitting trims whitespace around each
                    # boundary. Mistake notes are reconstructed for users, so use
                    # character-exact chunks and let the join remove only overlap.
                    preserve_boundaries=(
                        False if memory_type == "mistake" else CONTENT_PRESERVE_BOUNDARIES
                    ),
                    overlap=CONTENT_SPLIT_OVERLAP
                )
                stored_memories = []
                failed_chunks = []

                for i, chunk in enumerate(chunks):
                    chunk_hash = generate_content_hash(chunk)
                    chunk_metadata = final_metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["original_hash"] = content_hash

                    memory = Memory(
                        content=chunk,
                        content_hash=chunk_hash,
                        tags=final_tags,
                        memory_type=memory_type,
                        metadata=chunk_metadata
                    )

                    success, message = await self.storage.store(memory, skip_semantic_dedup=skip_dedup, store=store)
                    if success:
                        stored_memories.append(self._format_memory_response(memory))
                        # Queue chunk for AI quality scoring if enabled
                        if MCP_QUALITY_BOOST_ENABLED:
                            try:
                                await async_scorer.score_memory(memory, query="", storage=self.storage)
                            except Exception as e:
                                logger.debug("Background quality scoring for chunk failed silently: %s", _sanitize_log_value(str(e)))
                    else:
                        failed_chunks.append({"index": i, "reason": message})

                if failed_chunks and require_all_chunks:
                    reasons = ", ".join(set(fc["reason"] for fc in failed_chunks))
                    cleanup_failures = []
                    for stored_memory in stored_memories:
                        stored_hash = stored_memory.get("content_hash", "")
                        if not stored_hash:
                            continue
                        deleted, delete_message = await self.storage.delete(stored_hash)
                        if not deleted:
                            cleanup_failures.append(f"{stored_hash}: {delete_message}")

                    cleanup_detail = ""
                    if cleanup_failures:
                        cleanup_detail = f"; rollback failures: {'; '.join(cleanup_failures)}"
                    return {
                        "success": False,
                        "error": (
                            f"Failed to store all {len(chunks)} chunks: {reasons}"
                            f"{cleanup_detail}"
                        ),
                    }

                # If NO chunks were stored, return failure
                if not stored_memories:
                    reasons = ", ".join(set(fc["reason"] for fc in failed_chunks))
                    return {
                        "success": False,
                        "error": f"Failed to store all {len(chunks)} chunks: {reasons}"
                    }

                # If SOME chunks were stored, return partial success
                for mem_dict in stored_memories:
                    await self._plugin_registry.fire('on_store', mem_dict)

                return {
                    "success": True,
                    "memories": stored_memories,
                    "total_chunks": len(chunks),
                    "original_hash": content_hash,
                    "failed_chunks": len(failed_chunks)
                }
            else:
                # Store as single memory
                memory = Memory(
                    content=content,
                    content_hash=content_hash,
                    tags=final_tags,
                    memory_type=memory_type,
                    metadata=final_metadata
                )

                success, message = await self.storage.store(memory, skip_semantic_dedup=skip_dedup, store=store)

                # Issue #1216 (1b): a value-swap ("X is A" then "X is B") is
                # near-identical text to what it contradicts, so semantic dedup
                # rejects it before any contradiction check runs and it is
                # silently dropped. When on-store NLI is enabled, re-examine that
                # rejection: if the new content CONTRADICTS the memory it collided
                # with, it is not a duplicate — store it (bypassing dedup) and file
                # it as a contradiction, instead of dropping it.
                contradicted_hash = None
                if not success:
                    contradicted_hash = await self._contradiction_behind_duplicate(memory, message)
                    if contradicted_hash:
                        success, message = await self.storage.store(
                            memory, skip_semantic_dedup=True, store=store
                        )
                        if not success:
                            contradicted_hash = None

                if success:
                    # Queue for AI quality scoring if enabled
                    if MCP_QUALITY_BOOST_ENABLED:
                        try:
                            await async_scorer.score_memory(memory, query="", storage=self.storage)
                        except Exception as e:
                            logger.debug("Background quality scoring queued (or failed silently): %s", _sanitize_log_value(str(e)))

                    # Entity linking: extract entities and create shares_entity edges
                    await self._maybe_link_entities(memory)

                    await self._plugin_registry.fire('on_store', self._format_memory_response(memory))

                    response = {
                        "success": True,
                        "memory": self._format_memory_response(memory)
                    }
                    if contradicted_hash:
                        filed_ok, filing = await self._file_contradiction(
                            memory.content_hash, contradicted_hash
                        )
                        # Never report a filing that did not happen: a failed
                        # quarantine leaves the memory stored *and* active.
                        key = "filed_as_contradiction" if filed_ok else "contradiction_filing_failed"
                        response[key] = filing
                    return response
                else:
                    return {
                        "success": False,
                        "error": message
                    }

        except ValueError as e:
            # Handle validation errors specifically
            logger.warning("Validation error storing memory: %s", _sanitize_log_value(str(e)))
            return {
                "success": False,
                "error": f"Invalid memory data: {str(e)}"
            }
        except ConnectionError as e:
            # Handle storage connectivity issues
            logger.error("Storage connection error: %s", _sanitize_log_value(str(e)))
            return {
                "success": False,
                "error": f"Storage connection failed: {str(e)}"
            }
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error storing memory: {e}")
            return {
                "success": False,
                "error": f"Failed to store memory: {str(e)}"
            }

    async def evolve_memory(
        self,
        existing_hash: str,
        content: str,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Versioned update that gets the same post-store steps as store_memory().

        ``storage.update_memory_versioned()`` writes the new version straight
        into storage, so a caller that uses it directly skips everything
        store_memory() does after a write: caller metadata and agent identity,
        AI quality scoring, entity linking and ``on_store`` plugins. This is
        the service-level entry point for evolving a memory.

        Returns:
            ``(success, message, new_hash)``, as from update_memory_versioned().
        """
        if not hasattr(self.storage, "update_memory_versioned"):
            return False, "Storage backend does not support versioned updates", None
        # Storage inherits omitted tags/type from the old version; keep that
        # version around so the re-read fallback below can do the same.
        previous = None
        if tags is None or memory_type is None:
            previous = await self.storage.get_by_hash(existing_hash)
        ok, msg, new_hash = await self.storage.update_memory_versioned(
            existing_hash,
            content,
            new_tags=tags,
            new_memory_type=memory_type,
            reason=reason,
        )
        if not ok or not new_hash:
            return ok, msg, new_hash

        final_metadata = dict(metadata) if metadata else {}
        for key in ("tags", "type"):
            final_metadata.pop(key, None)
        # Same RFC #1100 precedence as store_memory(), minus the explicit arg.
        resolved_agent_id = os.environ.get("MCP_AGENT_ID") or final_metadata.get("agent_id")
        if resolved_agent_id:
            final_metadata["agent_id"] = resolved_agent_id
        # The new version is already committed from here on, so a failure below
        # is reported and worked around rather than turned into a failed evolve.
        if final_metadata:
            meta_ok, meta_msg = await self.storage.update_memory_metadata(
                new_hash, {"metadata": final_metadata}, preserve_timestamps=True
            )
            if not meta_ok:
                logger.warning(
                    "Evolved memory %s but could not write its metadata: %s",
                    new_hash[:8], _sanitize_log_value(str(meta_msg)),
                )
                msg = f"{msg} (metadata update failed: {meta_msg})"

        memory = await self.storage.get_by_hash(new_hash)
        if memory is None:
            # Re-read failed; score the version as written instead of skipping it.
            memory = self._as_written(new_hash, content, tags, memory_type, final_metadata, previous)
        await self._run_post_store_steps(memory)
        return ok, msg, new_hash

    @staticmethod
    def _as_written(new_hash, content, tags, memory_type, metadata, previous) -> Memory:
        """The evolved version as storage wrote it, for when it can't be re-read."""
        if tags is None and previous is not None:
            tags = previous.tags
        if memory_type is None and previous is not None:
            memory_type = previous.memory_type
        return Memory(
            content=content,
            content_hash=new_hash,
            tags=list(tags or []),
            memory_type=memory_type,
            metadata=metadata,
        )

    async def _run_post_store_steps(self, memory: Memory) -> None:
        """Quality scoring, entity linking and on_store plugins for a new memory."""
        if MCP_QUALITY_BOOST_ENABLED:
            try:
                await async_scorer.score_memory(memory, query="", storage=self.storage)
            except Exception as e:
                logger.debug("Background quality scoring failed silently: %s", _sanitize_log_value(str(e)))
        await self._maybe_link_entities(memory)
        await self._plugin_registry.fire('on_store', self._format_memory_response(memory))

    async def _contradiction_behind_duplicate(self, memory, reject_message):
        """Return the hash of the near-duplicate ``memory`` contradicts, or None.

        Issue #1216 (1b): only fires when on-store NLI is enabled, the rejection
        was a *semantic* duplicate, and an NLI classifier labels the new content a
        contradiction of the memory it collided with at or above the configurable
        quarantine gate (MCP_QUARANTINE_NLI_THRESHOLD). Fully guarded — any failure
        returns None, so the caller falls back to the original duplicate rejection
        and default behaviour is unchanged when NLI-on-store is off.
        """
        if os.getenv("MCP_NLI_ON_STORE", "false").lower() != "true":
            return None
        try:
            match = re.search(
                r"semantically similar to ([a-f0-9]+)", str(reject_message), re.IGNORECASE
            )
            if not match:
                return None
            existing_hash = match.group(1)
            existing = await self.storage.get_by_hash(existing_hash)
            if not existing or not getattr(existing, "content", None):
                return None
            from ..reasoning.nli import NLIClassifier
            from ..consolidation.quarantine import (
                _quarantine_nli_threshold,
                _warn_if_gate_unreachable,
            )
            classifier = NLIClassifier(backend="auto")
            threshold = _quarantine_nli_threshold()
            # Same reachability check as check_beliefs_on_store: with the
            # default heuristic ceiling (0.55) under the default gate (0.7)
            # this rescue can never fire, and that must not be silent here either.
            _warn_if_gate_unreachable(classifier, threshold)
            result = await classifier.classify(existing.content, memory.content)
            if result.label == "contradiction" and result.confidence >= threshold:
                return existing_hash
        except Exception as e:
            logger.debug(f"Contradiction-behind-duplicate check failed: {_sanitize_log_value(str(e))}")
        return None

    async def _file_contradiction(self, content_hash, contradicted_hash):
        """Quarantine a rescued value-swap against the *memory* it contradicts.

        Returns ``(ok, filing)``: ``filing`` always names the contradicted hash
        and carries the quarantine result; ``ok`` is True only if the memory is
        actually quarantined. ``quarantine_memory`` reports failure as
        ``{"status": "error"}`` rather than raising, so the status is checked —
        a failed quarantine leaves the dedup-bypassed memory stored and active,
        and the caller must say so instead of reporting it filed (issue #1216).
        The store that already succeeded is never unwound here.
        """
        filing = {"contradicts": contradicted_hash}
        try:
            from ..consolidation.quarantine import quarantine_memory
            q = await quarantine_memory(
                self.storage, content_hash, None,
                reason=(
                    f"Value differs from near-duplicate {contradicted_hash[:8]}; "
                    f"filed as contradiction instead of dropped as duplicate (#1216)"
                ),
                contradicted_memory_hash=contradicted_hash,
            )
        except Exception as e:
            q = {"status": "error", "message": str(e)}
        filing["quarantine"] = q
        if q.get("status") == "quarantined":
            return True, filing
        logger.warning(
            f"Memory {content_hash[:8]} was stored past semantic dedup as a contradiction "
            f"of {contradicted_hash[:8]} but could not be quarantined: "
            f"{_sanitize_log_value(str(q.get('message', 'unknown error')))}"
        )
        return False, filing

    async def retrieve_memories(
        self,
        query: str,
        n_results: int = 10,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None
    ) -> Union[RetrieveMemoriesSuccess, RetrieveMemoriesError]:
        """
        Retrieve memories by semantic search with optional filtering.

        Args:
            query: Search query string
            n_results: Maximum number of results
            tags: Optional tag filtering
            memory_type: Optional memory type filtering

        Returns:
            Dictionary with search results
        """
        try:
            # Retrieve memories using semantic search. Filtering remains at the
            # service layer for backward compatibility with storage adapters.
            memories = await self.storage.retrieve(
                query=query,
                n_results=n_results
            )

            # Apply optional post-filtering
            filtered_memories = memories
            if tags or memory_type:
                filtered_memories = []
                for query_result in memories:
                    # Filter by tags if specified
                    if tags:
                        memory_tags = query_result.memory.tags or []
                        if not any(tag in memory_tags for tag in tags):
                            continue

                    # Filter by memory_type if specified
                    if memory_type:
                        mem_type = query_result.memory.memory_type or ''
                        if mem_type != memory_type:
                            continue

                    filtered_memories.append(query_result)

            results = []
            for result in filtered_memories:
                # Extract Memory object from MemoryQueryResult and add similarity score
                memory_dict = self._format_memory_response(result.memory)
                memory_dict['similarity_score'] = result.relevance_score
                results.append(memory_dict)

                # Queue for background AI quality scoring if enabled
                if MCP_QUALITY_BOOST_ENABLED:
                    try:
                        await async_scorer.score_memory(result.memory, query=query, storage=self.storage)
                    except Exception as e:
                        logger.debug("Background quality scoring for retrieved memory failed silently: %s", _sanitize_log_value(str(e)))

            results = await self.apply_retrieve_plugins(query, results)

            return {
                "memories": results,
                "query": query,
                "count": len(results)
            }

        except Exception as e:
            logger.error("Error retrieving memories: %s", _sanitize_log_value(str(e)))
            return {
                "memories": [],
                "query": query,
                "error": f"Failed to retrieve memories: {str(e)}"
            }

    async def search_by_tag(
        self,
        tags: Union[str, List[str]],
        match_all: bool = False
    ) -> Union[SearchByTagSuccess, SearchByTagError]:
        """
        Search memories by tags with flexible matching options.

        Args:
            tags: Tag or list of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag

        Returns:
            Dictionary with matching memories
        """
        try:
            # Normalize tags to list (handles all formats including comma-separated)
            tags = normalize_tags(tags)

            # Preserve the existing ANY search, including its result ordering.
            if match_all:
                memories = await self.storage.search_by_tags(tags=tags, operation="AND")
            else:
                memories = await self.storage.search_by_tag(tags=tags)

            # Format results
            results = []
            for memory in memories:
                results.append(self._format_memory_response(memory))

            # Determine match type description
            match_type = "ALL" if match_all else "ANY"

            return {
                "memories": results,
                "tags": tags,
                "match_type": match_type,
                "count": len(results)
            }

        except Exception as e:
            logger.error("Error searching by tags: %s", _sanitize_log_value(str(e)))
            return {
                "memories": [],
                "tags": tags if isinstance(tags, list) else [tags],
                "error": f"Failed to search by tags: {str(e)}"
            }

    async def get_memory_by_hash(self, content_hash: str) -> Dict[str, Any]:
        """
        Retrieve a specific memory by its content hash using O(1) direct lookup.

        Args:
            content_hash: The content hash of the memory

        Returns:
            Dictionary with memory data or error
        """
        try:
            # Use direct O(1) lookup via storage.get_by_hash()
            memory = await self.storage.get_by_hash(content_hash)

            if memory:
                return {
                    "memory": self._format_memory_response(memory),
                    "found": True
                }
            else:
                return {
                    "found": False,
                    "content_hash": content_hash
                }

        except Exception as e:
            logger.error("Error getting memory by hash: %s", _sanitize_log_value(str(e)))
            return {
                "found": False,
                "content_hash": content_hash,
                "error": f"Failed to get memory: {str(e)}"
            }

    async def delete_memory(self, content_hash: str) -> Union[DeleteMemorySuccess, DeleteMemoryFailure]:
        """
        Delete a memory by its content hash.

        Args:
            content_hash: The content hash of the memory to delete

        Returns:
            Dictionary with operation result
        """
        try:
            success, message = await self.storage.delete(content_hash)
            if success:
                await self._plugin_registry.fire('on_delete', content_hash)
                return {
                    "success": True,
                    "content_hash": content_hash
                }
            else:
                return {
                    "success": False,
                    "content_hash": content_hash,
                    "error": message
                }

        except Exception as e:
            logger.error("Error deleting memory: %s", _sanitize_log_value(str(e)))
            return {
                "success": False,
                "content_hash": content_hash,
                "error": f"Failed to delete memory: {str(e)}"
            }

    async def health_check(self) -> Union[HealthCheckSuccess, HealthCheckFailure]:
        """
        Perform a health check on the memory storage system.

        Returns:
            Dictionary with health status and statistics
        """
        try:
            stats = await self.storage.get_stats()
            return {
                "healthy": True,
                "storage_type": stats.get("backend", "unknown"),
                "total_memories": stats.get("total_memories", 0),
                "last_updated": datetime.now().isoformat(),
                **stats
            }

        except Exception as e:
            logger.error("Health check failed: %s", _sanitize_log_value(str(e)))
            return {
                "healthy": False,
                "error": f"Health check failed: {str(e)}"
            }

    async def _maybe_link_entities(self, memory: Memory) -> None:
        """Extract entities and create shares_entity edges if linking is enabled."""
        from ..reasoning.entity_linker import is_entity_linking_enabled, EntityLinker
        if not is_entity_linking_enabled():
            return

        try:
            from ..reasoning.entities import EntityExtractor
            from ..server.handlers.graph import get_graph_storage

            graph = await get_graph_storage()
            if not graph:
                return

            extractor = EntityExtractor(
                domain_extractors=EntityExtractor.get_domain_extractors()
            )
            # tags is a top-level Memory attribute, metadata is the custom-key
            # dict — merge so the extractor's metadata-tag branch fires. Same
            # fix as server/handlers/graph.py (#218); this write path was still
            # discarding every tag before the extractor saw it.
            extraction_metadata = dict(memory.metadata or {})
            merged_tags = list(dict.fromkeys([
                *extraction_metadata.get('tags', []), *(memory.tags or []),
            ]))
            if merged_tags:
                extraction_metadata['tags'] = merged_tags

            entities = extractor.extract_entities(memory.content, extraction_metadata)
            if not entities:
                return

            # Store entity links (has_entity edges)
            entity_names = []
            for ent in entities:
                await graph.store_entity_link(memory.content_hash, ent.name, ent.entity_type)
                entity_names.append(ent.name)

            # Create shares_entity edges between memories with common entities
            linker = EntityLinker()
            await linker.link_by_entities(memory.content_hash, entity_names, graph)
        except Exception as e:
            logger.debug("Entity linking failed silently: %s", _sanitize_log_value(str(e)))

    def _format_memory_response(self, memory: Memory) -> MemoryResult:
        """
        Format a memory object for API response.

        Args:
            memory: The memory object to format

        Returns:
            Formatted memory dictionary
        """
        return {
            "content": memory.content,
            "content_hash": memory.content_hash,
            "tags": memory.tags,
            "memory_type": memory.memory_type,
            "metadata": memory.metadata,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "created_at_iso": memory.created_at_iso,
            "updated_at_iso": memory.updated_at_iso,
            "agent_id": memory.agent_id,  # Include agent_id as top-level field
        }

    # ─── Mistake Notes ────────────────────────────────────────────────

    @staticmethod
    def _mistake_metadata(memory: Memory) -> Dict[str, Any]:
        """Return mistake metadata as a mutable dictionary."""
        metadata = memory.metadata or {}
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata) if metadata else {}
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return dict(metadata) if isinstance(metadata, dict) else {}

    @classmethod
    def _mistake_group_key(cls, memory: Memory) -> str:
        """Identify every auto-split fragment that belongs to one mistake note."""
        metadata = cls._mistake_metadata(memory)
        return str(metadata.get("original_hash") or memory.content_hash)

    @staticmethod
    def _mistake_numeric_value(value: Any, default: float) -> float:
        """Normalize legacy JSON numeric strings without breaking group search."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _join_mistake_chunks(
        chunks: List[str],
        expected_hash: Optional[str] = None,
    ) -> str:
        """Reassemble chunks, including legacy boundary-trimmed mistake notes."""
        if not chunks:
            return ""

        content = chunks[0]
        joins: List[tuple[int, str]] = []
        for chunk in chunks[1:]:
            max_overlap = min(CONTENT_SPLIT_OVERLAP, len(content), len(chunk))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if content[-size:] == chunk[:size]:
                    overlap = size
                    break
            remainder = chunk[overlap:]
            joins.append((overlap, remainder))
            content += remainder

        if not expected_hash or generate_content_hash(content) == expected_hash:
            return content

        # Older boundary-preserving chunks used rstrip/lstrip and therefore
        # discarded their separator. Rebuild the common word/sentence and
        # structured-field boundaries. New exact-overlap chunks return above.
        content = chunks[0]
        field_prefixes = ("Context:", "Wrong:", "Right:")
        for _, remainder in joins:
            if content and remainder and not content[-1].isspace() and not remainder[0].isspace():
                if remainder.startswith(field_prefixes):
                    content += "\n"
                elif content[-1].isalnum() or content[-1] in ".!?;,:":
                    content += " "
            content += remainder
        return content

    @classmethod
    def _build_mistake_group(
        cls,
        key: str,
        members: List[Memory],
        relevance: float,
    ) -> Dict[str, Any]:
        """Build one normalized logical mistake note from its physical fragments."""
        ordered_members = sorted(
            members,
            key=lambda memory: cls._mistake_numeric_value(
                cls._mistake_metadata(memory).get("chunk_index", 0),
                0.0,
            ),
        )
        representative = ordered_members[0]
        metadata = cls._mistake_metadata(representative)
        defaults = {
            "failure_count": 1.0,
            "confidence": 0.5,
            "frustration_score": 0.0,
        }
        for field, default in defaults.items():
            values = [
                cls._mistake_numeric_value(
                    cls._mistake_metadata(member).get(field),
                    default,
                )
                for member in ordered_members
                if cls._mistake_metadata(member).get(field) is not None
            ]
            if values:
                maximum = max(values)
                metadata[field] = int(maximum) if field == "failure_count" else maximum
        metadata["is_avoid_rule"] = any(
            bool(cls._mistake_metadata(member).get("is_avoid_rule", False))
            for member in ordered_members
        )
        return {
            "content_hash": representative.content_hash,
            "logical_hash": key,
            "chunk_hashes": [member.content_hash for member in ordered_members],
            "content": cls._join_mistake_chunks(
                [member.content for member in ordered_members],
                expected_hash=key,
            ),
            "similarity_score": relevance,
            "metadata": metadata,
            "updated_at": max(member.updated_at for member in ordered_members),
            "_members": ordered_members,
        }

    async def _find_mistake_groups(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Search mistake notes before top-K filtering and rebuild logical notes.

        The mistake-note tag is applied by the storage backend before its final
        result limit. If a backend cannot produce semantic candidates (for
        example a legacy database without embeddings), recent mistake notes are
        returned as a truthful fallback instead of incorrectly reporting zero.
        """
        if limit <= 0:
            return []

        logical_limit = limit
        candidate_limit = max(
            _MISTAKE_SEARCH_CANDIDATE_FLOOR,
            logical_limit * _MISTAKE_SEARCH_CANDIDATE_MULTIPLIER,
        )
        query_results = await self.storage.retrieve(
            query=query,
            n_results=candidate_limit,
            tags=[_MISTAKE_NOTE_TAG],
        )

        candidates: List[tuple[Memory, float]] = [
            (result.memory, result.relevance_score)
            for result in query_results
            if result.memory.memory_type == "mistake"
        ]

        all_mistakes: Optional[List[Memory]] = None
        if not candidates:
            all_mistakes = await self.storage.get_all_memories(
                limit=None,
                memory_type="mistake",
                tags=[_MISTAKE_NOTE_TAG],
            )
            candidates = [(memory, 0.0) for memory in all_mistakes]

        ordered_keys: List[str] = []
        relevance_by_key: Dict[str, float] = {}
        candidate_by_key: Dict[str, List[Memory]] = {}
        for memory, relevance in candidates:
            key = self._mistake_group_key(memory)
            if key not in relevance_by_key:
                ordered_keys.append(key)
                relevance_by_key[key] = relevance
                candidate_by_key[key] = []
            else:
                relevance_by_key[key] = max(relevance_by_key[key], relevance)
            candidate_by_key[key].append(memory)

        selected_keys = ordered_keys[:logical_limit]
        needs_siblings = any(
            self._mistake_metadata(memory).get("total_chunks", 1) > 1
            for key in selected_keys
            for memory in candidate_by_key[key]
        )
        if needs_siblings and all_mistakes is None:
            all_mistakes = await self.storage.get_all_memories(
                limit=None,
                memory_type="mistake",
                tags=[_MISTAKE_NOTE_TAG],
            )

        siblings_by_key: Dict[str, List[Memory]] = {}
        if all_mistakes is not None:
            for memory in all_mistakes:
                siblings_by_key.setdefault(self._mistake_group_key(memory), []).append(memory)

        groups: List[Dict[str, Any]] = []
        for key in selected_keys:
            members = siblings_by_key.get(key, candidate_by_key[key])
            groups.append(self._build_mistake_group(key, members, relevance_by_key[key]))

        return groups

    async def _find_mistake_group_by_logical_hash(
        self,
        logical_hash: str,
        scan_fragments: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Resolve an exact logical note even when semantic retrieval is unavailable."""
        direct = await self.storage.get_by_hash(logical_hash)
        if direct and direct.memory_type == "mistake":
            return self._build_mistake_group(logical_hash, [direct], 1.0)
        if not scan_fragments:
            return None

        all_mistakes = await self.storage.get_all_memories(
            limit=None,
            memory_type="mistake",
            tags=[_MISTAKE_NOTE_TAG],
        )
        members = [
            memory
            for memory in all_mistakes
            if self._mistake_group_key(memory) == logical_hash
        ]
        if not members:
            return None
        return self._build_mistake_group(logical_hash, members, 1.0)

    async def _increment_mistake_group(
        self,
        group: Dict[str, Any],
        learning_rate: float,
        error_weight: float,
        frustration_threshold: float,
    ) -> int:
        """Increment every fragment of a logical mistake note consistently."""
        metadata = dict(group.get("metadata") or {})
        count = int(metadata.get("failure_count", 1)) + 1
        confidence = min(1.0, float(metadata.get("confidence", 0.5)) + learning_rate * error_weight)
        frustration = float(metadata.get("frustration_score", 0.0)) + 1.0

        updated_members = []
        for member in group.get("_members", []):
            member_metadata = self._mistake_metadata(member)
            member_metadata.update({
                "failure_count": count,
                "confidence": confidence,
                "frustration_score": frustration,
                "is_avoid_rule": frustration >= frustration_threshold,
            })
            updated_members.append(Memory(
                content=member.content,
                content_hash=member.content_hash,
                tags=member.tags,
                memory_type=member.memory_type,
                metadata=member_metadata,
                created_at=member.created_at,
                created_at_iso=member.created_at_iso,
                updated_at=member.updated_at,
                updated_at_iso=member.updated_at_iso,
            ))

        if len(updated_members) == 1:
            member = updated_members[0]
            success, message = await self.storage.update_memory_metadata(
                content_hash=member.content_hash,
                updates={"metadata": member.metadata},
                preserve_timestamps=False,
            )
            if not success:
                raise RuntimeError(
                    f"Failed to update mistake fragment {member.content_hash}: {message}"
                )
        elif updated_members:
            results = await self.storage.update_memories_batch(
                updated_members,
                preserve_timestamps=False,
            )
            if len(results) != len(updated_members) or not all(results):
                raise RuntimeError("Failed to atomically update all mistake fragments")
        return count

    async def mistake_note_add(
        self,
        error_pattern: str,
        context_signature: str,
        incorrect_action: str,
        correct_action: str,
    ) -> Dict[str, Any]:
        """
        Record a mistake pattern for error replay.

        Stores as a regular memory with memory_type='mistake'. If a similar
        pattern already exists (above dedup threshold), increments failure_count
        instead of creating a duplicate.

        Args:
            error_pattern: The error pattern or message
            context_signature: Context where the error occurred
            incorrect_action: What was done incorrectly
            correct_action: What should have been done instead

        Returns:
            Dictionary with operation result
        """
        from ..config import MCP_MISTAKE_NOTE_DEDUP_THRESHOLD

        # A mistake note's value is its remediation. Reject empty correct_action —
        # JSON-schema `required` enforces presence, not non-emptiness (issue #1055).
        if not (correct_action or "").strip():
            return {
                "status": "error",
                "message": "correct_action must not be empty — a mistake note requires a remediation, not just an error pattern",
            }

        ERROR_WEIGHT = float(os.getenv("MCP_ERROR_WEIGHT", "3.0"))
        LEARNING_RATE = float(os.getenv("MCP_LEARNING_RATE", "0.1"))
        FRUSTRATION_THRESHOLD = float(os.getenv("MCP_FRUSTRATION_THRESHOLD", "5.0"))

        content = (
            f"Pattern: {error_pattern}\n"
            f"Context: {context_signature}\n"
            f"Wrong: {incorrect_action}\n"
            f"Right: {correct_action}"
        )

        try:
            from ..utils.hashing import generate_content_hash

            logical_hash = generate_content_hash(content)
            max_length = self.storage.max_content_length
            exact_group = await self._find_mistake_group_by_logical_hash(
                logical_hash,
                scan_fragments=bool(
                    ENABLE_AUTO_SPLIT and max_length and len(content) > max_length
                ),
            )
            if exact_group:
                count = await self._increment_mistake_group(
                    exact_group,
                    learning_rate=LEARNING_RATE,
                    error_weight=ERROR_WEIGHT,
                    frustration_threshold=FRUSTRATION_THRESHOLD,
                )
                return {
                    "status": "updated",
                    "content_hash": exact_group["content_hash"],
                    "logical_hash": exact_group["logical_hash"],
                    "chunks_updated": len(exact_group["chunk_hashes"]),
                    "failure_count": count,
                    "message": f"Existing mistake note updated (seen {count} times)",
                }

            # Check for existing similar mistake note
            existing = await self._find_mistake_groups(
                query=f"{error_pattern} {context_signature}",
                limit=3,
            )

            if existing:
                for mem in existing:
                    score = mem.get("similarity_score", 0)
                    if score >= MCP_MISTAKE_NOTE_DEDUP_THRESHOLD:
                        content_hash = mem["content_hash"]
                        count = await self._increment_mistake_group(
                            mem,
                            learning_rate=LEARNING_RATE,
                            error_weight=ERROR_WEIGHT,
                            frustration_threshold=FRUSTRATION_THRESHOLD,
                        )
                        return {
                            "status": "updated",
                            "content_hash": content_hash,
                            "logical_hash": mem["logical_hash"],
                            "chunks_updated": len(mem["chunk_hashes"]),
                            "failure_count": count,
                            "message": f"Existing mistake note updated (seen {count} times)",
                        }

            # No match — store new mistake note with initial confidence/frustration
            initial_meta = {
                "failure_count": 1,
                "confidence": 0.5,
                "frustration_score": 1.0,
                "is_avoid_rule": False,
            }
            result = await self.store_memory(
                content=content,
                tags="mistake-note,error-replay",
                memory_type="mistake",
                metadata=initial_meta,
                skip_semantic_dedup=True,
                require_all_chunks=True,
            )

            if not result.get("success"):
                # Handle race condition: store's semantic dedup rejected, but we can
                # still increment the existing note it found (#1034)
                error_msg = str(result.get("error", ""))
                existing_hash = None
                match = re.search(r"semantically similar to ([a-f0-9]+)", error_msg, re.IGNORECASE)
                if match:
                    existing_hash = match.group(1)
                elif "exact match" in error_msg.lower():
                    existing_hash = generate_content_hash(content)

                if existing_hash:
                    existing = await self.storage.get_by_hash(existing_hash)
                    group_hash = (
                        self._mistake_group_key(existing)
                        if existing and existing.memory_type == "mistake"
                        else existing_hash
                    )
                    existing_group = await self._find_mistake_group_by_logical_hash(group_hash)
                    if existing_group:
                        count = await self._increment_mistake_group(
                            existing_group,
                            learning_rate=LEARNING_RATE,
                            error_weight=ERROR_WEIGHT,
                            frustration_threshold=FRUSTRATION_THRESHOLD,
                        )
                        return {
                            "status": "updated",
                            "content_hash": existing_group["content_hash"],
                            "logical_hash": existing_group["logical_hash"],
                            "chunks_updated": len(existing_group["chunk_hashes"]),
                            "failure_count": count,
                            "message": f"Existing mistake note updated (seen {count} times)",
                        }
                return {"status": "error", "message": f"Failed to store: {result}"}

            content_hash = ""
            logical_hash = ""
            chunks_created = 1
            if isinstance(result, dict):
                mem = result.get("memory", {})
                if isinstance(mem, dict):
                    content_hash = mem.get("content_hash", "")
                    logical_hash = content_hash
                if not content_hash:
                    content_hash = result.get("content_hash", "")
                chunk_memories = result.get("memories", [])
                if not content_hash and isinstance(chunk_memories, list) and chunk_memories:
                    first_chunk = chunk_memories[0]
                    if isinstance(first_chunk, dict):
                        content_hash = first_chunk.get("content_hash", "")
                logical_hash = result.get("original_hash", logical_hash or content_hash)
                if isinstance(chunk_memories, list) and chunk_memories:
                    chunks_created = len(chunk_memories)
            return {
                "status": "created",
                "content_hash": content_hash,
                "logical_hash": logical_hash,
                "chunks_created": chunks_created,
                "failure_count": 1,
                "message": "New mistake note recorded",
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def mistake_note_search(
        self,
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Search mistake notes by semantic similarity.

        Args:
            query: Search query (error message, context, or task description)
            limit: Maximum number of results

        Returns:
            Dictionary with matching mistake notes
        """
        try:
            memories = await self._find_mistake_groups(
                query=query,
                limit=limit,
            )

            notes = []
            for mem in memories:
                meta = mem.get("metadata") or {}
                notes.append({
                    "content_hash": mem["content_hash"],
                    "logical_hash": mem["logical_hash"],
                    "chunk_hashes": mem["chunk_hashes"],
                    "content": mem["content"],
                    "similarity": mem.get("similarity_score", 0),
                    "failure_count": meta.get("failure_count", 1),
                    "metadata": meta,
                    "updated_at": mem.get("updated_at"),
                })

            return {"notes": notes, "count": len(notes)}

        except Exception as e:
            return {"notes": [], "count": 0, "error": str(e)}

    async def mistake_note_update(
        self,
        content_hash: str,
        failure_count: Optional[int] = None,
        error_pattern: Optional[str] = None,
        context_signature: Optional[str] = None,
        incorrect_action: Optional[str] = None,
        correct_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update fields of an existing mistake note.

        Args:
            content_hash: Hash of the mistake note to update
            failure_count: New failure count (optional)
            error_pattern: Updated error pattern (optional)
            context_signature: Updated context (optional)
            incorrect_action: Updated incorrect action (optional)
            correct_action: Updated correct action (optional)

        Returns:
            Dictionary with operation result
        """
        # Don't allow an existing note's remediation to be blanked (issue #1055).
        if correct_action is not None and not correct_action.strip():
            return {
                "status": "error",
                "message": "correct_action must not be empty — a mistake note requires a remediation, not just an error pattern",
            }

        try:
            mem = await self.storage.get_by_hash(content_hash)
            if not mem:
                return {"status": "error", "message": f"No memory found with hash {content_hash}"}

            mem_type = getattr(mem, 'memory_type', None) or (mem.metadata or {}).get("type")
            if mem_type != "mistake":
                return {"status": "error", "message": f"Memory {content_hash} is not a mistake note (type={mem_type})"}

            content_changed = any(f is not None for f in [error_pattern, context_signature, incorrect_action, correct_action])

            if content_changed:
                # Content update requires delete + re-store (hash changes with content)
                current = _parse_mistake_content(mem.content)
                new_content = (
                    f"Pattern: {error_pattern or current['error_pattern']}\n"
                    f"Context: {context_signature or current['context_signature']}\n"
                    f"Wrong: {incorrect_action or current['incorrect_action']}\n"
                    f"Right: {correct_action or current['correct_action']}"
                )
                meta = mem.metadata if isinstance(mem.metadata, dict) else {}
                if failure_count is not None:
                    meta["failure_count"] = failure_count

                # Store new version FIRST to prevent data loss if store fails
                result = await self.store_memory(
                    content=new_content,
                    tags="mistake-note,error-replay",
                    memory_type="mistake",
                    metadata=meta,
                )
                if not (isinstance(result, dict) and result.get("success")):
                    return {"status": "error", "message": f"Failed to store updated note: {result}"}

                new_hash = ""
                m = result.get("memory", {})
                new_hash = m.get("content_hash", "") if isinstance(m, dict) else ""

                # Only delete old after new is safely stored
                await self.delete_memory(content_hash)
                return {"status": "updated", "content_hash": new_hash or content_hash}

            # Metadata-only update (failure_count)
            if failure_count is not None:
                meta = mem.metadata if isinstance(mem.metadata, dict) else {}
                meta["failure_count"] = failure_count
                success, msg = await self.storage.update_memory_metadata(
                    content_hash=content_hash,
                    updates={"metadata": meta},
                    preserve_timestamps=False,
                )
                if not success:
                    return {"status": "error", "message": f"Metadata update failed: {msg}"}
                return {"status": "updated", "content_hash": content_hash}

            return {"status": "error", "message": "No fields to update"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def mistake_note_delete(self, content_hash: str) -> Dict[str, Any]:
        """
        Delete a mistake note by content hash.

        Args:
            content_hash: Hash of the mistake note to delete

        Returns:
            Dictionary with operation result
        """
        try:
            mem = await self.storage.get_by_hash(content_hash)
            if not mem:
                return {"status": "error", "message": f"No memory found with hash {content_hash}"}

            mem_type = getattr(mem, 'memory_type', None) or (mem.metadata or {}).get("type")
            if mem_type != "mistake":
                return {"status": "error", "message": f"Memory {content_hash} is not a mistake note (type={mem_type})"}

            result = await self.delete_memory(content_hash)
            if not (isinstance(result, dict) and result.get("success")):
                return {"status": "error", "message": f"Delete failed: {result.get('error', 'unknown')}"}
            return {"status": "deleted", "content_hash": content_hash}

        except Exception as e:
            return {"status": "error", "message": str(e)}


def _parse_mistake_content(content: str) -> Dict[str, str]:
    """Parse structured mistake note content into fields."""
    fields = {
        "error_pattern": "",
        "context_signature": "",
        "incorrect_action": "",
        "correct_action": "",
    }
    for line in content.split("\n"):
        if line.startswith("Pattern: "):
            fields["error_pattern"] = line[9:]
        elif line.startswith("Context: "):
            fields["context_signature"] = line[9:]
        elif line.startswith("Wrong: "):
            fields["incorrect_action"] = line[7:]
        elif line.startswith("Right: "):
            fields["correct_action"] = line[7:]
    return fields
