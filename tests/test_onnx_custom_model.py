"""Tests for ONNX backend honoring a custom embedding model.

The ONNX backend used to be hardcoded to all-MiniLM-L6-v2 (Chroma S3), so any
other MCP_EMBEDDING_MODEL was silently ignored under USE_ONNX. These tests pin
the routing logic that decides between the default S3 path and a Hugging Face
Hub download for a custom model.

They exercise the routing decision only (no network / no model load), so they
run anywhere ONNX Runtime + tokenizers are importable.
"""

import pytest

try:
    import onnxruntime  # noqa: F401  (probe)
    import tokenizers  # noqa: F401  (probe)
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE, reason="Requires onnxruntime + tokenizers"
)


def _make_without_io(monkeypatch, model_name, env=None):
    """Instantiate ONNXEmbeddingModel with download + init stubbed out.

    We only want to assert the routing attributes computed in __init__, not
    perform any network access or ONNX session creation.
    """
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    for key, val in (env or {}).items():
        monkeypatch.setenv(key, val)

    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_init_model", lambda self: None)
    return mod.ONNXEmbeddingModel(model_name=model_name)


def test_default_model_uses_s3_path(monkeypatch):
    """all-MiniLM-L6-v2 keeps the original bundled S3 archive path."""
    m = _make_without_io(monkeypatch, "all-MiniLM-L6-v2")
    assert m._is_default_model is True
    assert m._hf_repo is None


def test_custom_model_resolves_to_onnx_community(monkeypatch):
    """A non-default model resolves to onnx-community/<base>-ONNX on the Hub."""
    m = _make_without_io(monkeypatch, "paraphrase-multilingual-MiniLM-L12-v2")
    assert m._is_default_model is False
    assert m._hf_repo == "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
    # custom model must NOT reuse the default MiniLM cache dir
    assert m._model_dir.name == "paraphrase-multilingual-MiniLM-L12-v2"


def test_custom_model_repo_override(monkeypatch):
    """MCP_ONNX_MODEL_REPO overrides the default HF repo resolution."""
    m = _make_without_io(
        monkeypatch,
        "some-model",
        env={"MCP_ONNX_MODEL_REPO": "myorg/some-model-onnx"},
    )
    assert m._is_default_model is False
    assert m._hf_repo == "myorg/some-model-onnx"


def test_org_prefixed_model_name_uses_basename(monkeypatch):
    """A org/name model uses only the basename for repo + cache resolution."""
    m = _make_without_io(monkeypatch, "sentence-transformers/all-MiniLM-L6-v2")
    # basename is the default model -> S3 path
    assert m._is_default_model is True


def test_none_model_name_falls_back_to_default(monkeypatch):
    """A None model_name normalises to the default (no crash downstream)."""
    m = _make_without_io(monkeypatch, None)
    assert m.model_name == "all-MiniLM-L6-v2"
    assert m._is_default_model is True


def test_resolve_model_path_default_vs_custom(monkeypatch, tmp_path):
    """_resolve_model_path returns the S3 layout for default, custom dir otherwise."""
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    default = _make_without_io(monkeypatch, "all-MiniLM-L6-v2")
    assert default._resolve_model_path() == (
        default.DOWNLOAD_PATH / default.EXTRACTED_FOLDER_NAME / "model.onnx"
    )

    custom = _make_without_io(monkeypatch, "paraphrase-multilingual-MiniLM-L12-v2")
    # point the custom dir at a tmp layout with onnx/model.onnx
    custom._model_dir = tmp_path
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"stub")
    assert custom._resolve_model_path() == tmp_path / "onnx" / "model.onnx"


def test_find_onnx_file_ignores_quantized_variants(monkeypatch, tmp_path):
    """Only the canonical model.onnx is selected, not model_quantized.onnx."""
    m = _make_without_io(monkeypatch, "some-multilingual-model")
    m._model_dir = tmp_path
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model_quantized.onnx").write_bytes(b"q")
    # no canonical model.onnx yet -> None (won't silently pick the quantized one)
    assert m._find_onnx_file() is None
    (onnx_dir / "model.onnx").write_bytes(b"full")
    assert m._find_onnx_file() == onnx_dir / "model.onnx"


@pytest.mark.parametrize("bad_name", [
    "bad name with spaces",   # space in base
    "weird$model",            # shell/path metachar
    "..",                     # base resolves to parent-dir token
    "model;rm -rf",           # command-ish separators
])
def test_custom_model_name_is_validated(monkeypatch, bad_name):
    """A non-default model name whose base (after the last '/') is not a plain
    identifier is rejected before it becomes a filesystem path or Hub repo id.

    Note: a leading path like ``a/b/base`` is reduced to ``base`` first, so this
    guards the residual component that actually reaches the path/repo builders.
    """
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_init_model", lambda self: None)
    with pytest.raises(ValueError):
        mod.ONNXEmbeddingModel(model_name=bad_name)


def test_default_model_name_never_validated(monkeypatch):
    """The default model skips validation entirely (backward compatible)."""
    m = _make_without_io(monkeypatch, "all-MiniLM-L6-v2")
    assert m._is_default_model is True



def _tiny_tokenizer(truncation=None):
    """A real tokenizers.Tokenizer (word-level, no files) for truncation tests."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[UNK]": 0, **{f"w{i}": i + 1 for i in range(2000)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if truncation:
        tok.enable_truncation(max_length=truncation)
    return tok


@pytest.mark.parametrize("model_type,max_pos,expected", [
    ("bert", 512, 512),            # intfloat/multilingual-e5-small
    ("xlm-roberta", 514, 512),     # multilingual-e5-base: 2 slots reserved
])
def test_max_input_tokens_reads_model_config(monkeypatch, tmp_path, model_type, max_pos, expected):
    """The safeguard limit comes from the model's own config.json (2026-11-04)."""
    import json
    m = _make_without_io(monkeypatch, "multilingual-e5-small")
    m._model_dir = tmp_path
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type, "max_position_embeddings": max_pos}))
    assert m._max_input_tokens() == expected


def test_max_input_tokens_defaults_without_config(monkeypatch, tmp_path):
    m = _make_without_io(monkeypatch, "multilingual-e5-small")
    m._model_dir = tmp_path  # no config.json
    assert m._max_input_tokens() == 512


def test_tokenizer_without_truncation_gets_capped(monkeypatch, tmp_path):
    """A Hub tokenizer.json with no truncation is capped at the model limit, so a
    48 687-char memory no longer overflows the position embeddings (2026-11-04)."""
    import json
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert", "max_position_embeddings": 16}))
    (tmp_path / "tokenizer.json").write_text(_tiny_tokenizer().to_str())
    (tmp_path / "model.onnx").write_bytes(b"")

    class _FakeSession:
        def __init__(self, *a, **k): pass
        def get_outputs(self):
            class O: shape = [None, None, 8]
            return [O()]
        def get_inputs(self): return []

    monkeypatch.setattr(mod.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_find_onnx_file", lambda self: tmp_path / "model.onnx")
    m = mod.ONNXEmbeddingModel.__new__(mod.ONNXEmbeddingModel)
    m.model_name, m._is_default_model, m._model_dir = "multilingual-e5-small", False, tmp_path
    m._preferred_providers = ["CPUExecutionProvider"]
    m._init_model()

    long_text = " ".join(f"w{i}" for i in range(1000))
    assert len(m._tokenizer.encode(long_text).ids) == 16


def test_tokenizer_with_own_truncation_is_untouched(monkeypatch, tmp_path):
    """all-MiniLM-L6-v2 ships truncation=128: the safeguard must not change it."""
    import json
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert", "max_position_embeddings": 512}))
    (tmp_path / "tokenizer.json").write_text(_tiny_tokenizer(truncation=128).to_str())
    (tmp_path / "model.onnx").write_bytes(b"")

    class _FakeSession:
        def __init__(self, *a, **k): pass
        def get_outputs(self):
            class O: shape = [None, None, 8]
            return [O()]
        def get_inputs(self): return []

    monkeypatch.setattr(mod.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_find_onnx_file", lambda self: tmp_path / "model.onnx")
    m = mod.ONNXEmbeddingModel.__new__(mod.ONNXEmbeddingModel)
    m.model_name, m._is_default_model, m._model_dir = "multilingual-e5-small", False, tmp_path
    m._preferred_providers = ["CPUExecutionProvider"]
    m._init_model()

    assert len(m._tokenizer.encode(" ".join(f"w{i}" for i in range(1000))).ids) == 128
