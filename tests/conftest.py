"""
Shared test helpers — module-level singletons so expensive resources
(embed model, ChromaDB client) are loaded only once per process.

Usage in any test script:
    from conftest import get_config, get_embed_model, get_collection, ROOT
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from scripts/ regardless of cwd
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import load_config  # noqa: E402

# ── Singletons ──────────────────────────────────────────────────────────────
_config: dict | None = None
_embed_model = None
_collection = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_embed_model():
    """Load BAAI/bge-m3 once and cache it."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        from utils import get_device

        cfg = get_config()
        device = get_device(cfg)
        model_name = cfg["embedding"]["model"]
        print(f"[conftest] Loading embed model {model_name!r} on {device} …")
        _embed_model = SentenceTransformer(model_name, device=device)
        print(f"[conftest] Embed model ready.")
    return _embed_model


def get_collection():
    """Open the ChromaDB collection (read-only)."""
    global _collection
    if _collection is None:
        import chromadb

        cfg = get_config()
        chroma_dir = ROOT / cfg["paths"]["chroma_dir"]
        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {chroma_dir}. "
                "Run: python scripts/2_build_index.py first."
            )
        client = chromadb.PersistentClient(path=str(chroma_dir))
        # Используем collection_name из конфига (если задан), иначе дефолт «anatomy»
        coll_name = cfg.get("retrieval", {}).get("collection_name", "anatomy")
        _collection = client.get_collection(coll_name)
        count = _collection.count()
        print(f"[conftest] ChromaDB collection {coll_name!r}: {count:,} vectors.")
    return _collection
