"""Asymmetric embedding prefixes (multilingual-e5: "query: " / "passage: "), 2026-11-04.

Measured on the user's base: e5 only beats all-MiniLM-L6-v2 with the asymmetric prefixes.
Unset variables must keep the previous behaviour for every existing model.
"""
import numpy as np
import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.mixins.embeddings import with_embedding_prefix
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class _RecordingModel:
    def __init__(self, dim):
        self.dim, self.seen = dim, []

    def encode(self, texts, convert_to_numpy=True):
        self.seen.extend(texts)
        return np.array([[((hash(t) >> i) & 7) / 7.0 + 0.1 for i in range(self.dim)] for t in texts], dtype=np.float32)


def test_no_prefix_by_default(monkeypatch):
    monkeypatch.delenv("MCP_EMBEDDING_QUERY_PREFIX", raising=False)
    monkeypatch.delenv("MCP_EMBEDDING_PASSAGE_PREFIX", raising=False)
    assert with_embedding_prefix("abc", "query") == "abc"
    assert with_embedding_prefix("abc") == "abc"


def test_prefixes_follow_kind(monkeypatch):
    monkeypatch.setenv("MCP_EMBEDDING_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("MCP_EMBEDDING_PASSAGE_PREFIX", "passage: ")
    assert with_embedding_prefix("abc", "query") == "query: abc"
    assert with_embedding_prefix("abc", "passage") == "passage: abc"


@pytest.mark.asyncio
async def test_store_uses_passage_and_search_uses_query(monkeypatch, temp_db_path):
    monkeypatch.setenv("MCP_EMBEDDING_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("MCP_EMBEDDING_PASSAGE_PREFIX", "passage: ")
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/prefix.db")
    await storage.initialize()
    try:
        model = _RecordingModel(storage.embedding_dimension)
        storage.embedding_model, storage.enable_cache = model, False
        single = Memory(content="Une note stockée seule", content_hash=generate_content_hash("Une note stockée seule"))
        assert (await storage.store(single, skip_semantic_dedup=True))[0]
        batch = [Memory(content=f"Note par lot {i}", content_hash=generate_content_hash(f"Note par lot {i}")) for i in range(2)]
        assert all(ok for ok, _ in await storage.store_batch(batch))
        await storage.retrieve("chercher une note", n_results=3)
        assert "passage: Une note stockée seule" in model.seen
        assert "passage: Note par lot 0" in model.seen and "passage: Note par lot 1" in model.seen
        assert "query: chercher une note" in model.seen
        assert not any(t.startswith("passage: query:") or t == "chercher une note" for t in model.seen)
    finally:
        await storage.close()


@pytest.mark.parametrize("raw,expected", [(None, 0.95), ("0.985", 0.985), ("abc", 0.95), ("0", 0.95), ("1.5", 0.95)])
def test_conflict_threshold_from_env(monkeypatch, raw, expected):
    """Conflict detection compares raw cosine: the threshold must follow the model (2026-11-04)."""
    from mcp_memory_service.storage.mixins.metadata import _conflict_similarity_threshold
    if raw is None:
        monkeypatch.delenv("MCP_CONFLICT_SIMILARITY_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("MCP_CONFLICT_SIMILARITY_THRESHOLD", raw)
    assert _conflict_similarity_threshold() == expected


def test_harvest_tool_and_route_honour_env_threshold(monkeypatch):
    """The MCP tool and the REST route used HarvestConfig(...) and evolved at 0.85 whatever
    MCP_HARVEST_SIMILARITY_THRESHOLD said; under e5 that superseded unrelated memories (2026-11-04)."""
    import inspect
    from mcp_memory_service import server_impl
    from mcp_memory_service.web.api import harvest as harvest_api
    from mcp_memory_service.harvest.models import harvest_config_from_env
    for module in (server_impl, harvest_api):
        assert "config = harvest_config_from_env(" in inspect.getsource(module)
    monkeypatch.setenv("MCP_HARVEST_SIMILARITY_THRESHOLD", "0.98")
    assert harvest_config_from_env(sessions=1, dry_run=False).similarity_threshold == 0.98


def test_clustering_eps_floor_is_configurable():
    pytest.importorskip("sklearn")  # _estimate_eps uses NearestNeighbors (installed in production)
    import numpy as np
    from mcp_memory_service.consolidation.clustering import SemanticClusteringEngine
    rng = np.random.default_rng(0)
    tight = rng.normal(size=(40, 8)) * 0.001 + 1.0  # near-identical vectors: tiny k-distances
    assert SemanticClusteringEngine._estimate_eps(tight, 2) == 0.05
    assert SemanticClusteringEngine._estimate_eps(tight, 2, floor=0.02) == 0.02
