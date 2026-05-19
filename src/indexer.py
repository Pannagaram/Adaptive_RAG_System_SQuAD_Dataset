"""
Part 1: Vector Indexer
----------------------
Builds and manages a FAISS vector index from document embeddings.
Uses sentence-transformers for encoding and supports persistence.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import CONFIG, FAISS_INDEX_DIR


class VectorIndexer:
    """Manages FAISS index creation, persistence, and search."""

    def __init__(self, config=None):
        self.config = config or CONFIG.model
        self.index: Optional[faiss.Index] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.document_texts: List[str] = []
        self.document_ids: List[str] = []
        self._index_path = FAISS_INDEX_DIR / "index.faiss"
        self._meta_path = FAISS_INDEX_DIR / "index_meta.json"

    def _load_embedder(self):
        """Lazy-load the embedding model."""
        if self.embedder is None:
            print(f"[Indexer] Loading embedding model: {self.config.embedding_model}")
            self.embedder = SentenceTransformer(self.config.embedding_model)
        return self.embedder

    def build_index(self, texts: List[str], doc_ids: List[str],
                    batch_size: int = 256, force_rebuild: bool = False) -> None:
        """
        Build FAISS index from document texts.

        Args:
            texts: List of document text strings
            doc_ids: List of document IDs corresponding to texts
            batch_size: Encoding batch size
            force_rebuild: If True, rebuild even if cached index exists
        """
        if not force_rebuild and self._index_path.exists():
            print("[Indexer] Loading cached FAISS index...")
            self.load_index()
            return

        self._load_embedder()
        self.document_texts = texts
        self.document_ids = doc_ids

        print(f"[Indexer] Encoding {len(texts)} documents...")
        start_time = time.time()

        # Encode in batches with progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            embeddings = self.embedder.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,  # For cosine similarity via inner product
                convert_to_numpy=True,
            )
            all_embeddings.append(embeddings)

        embeddings_matrix = np.vstack(all_embeddings).astype("float32")
        encode_time = time.time() - start_time
        print(f"[Indexer] Encoding complete in {encode_time:.1f}s")

        # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
        dim = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_matrix)

        print(f"[Indexer] FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        # Save to disk
        self.save_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Search the FAISS index for the most similar documents.

        Args:
            query: The search query string
            top_k: Number of results to return

        Returns:
            List of (index, score, text) tuples, sorted by relevance (descending)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        self._load_embedder()

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Search FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append((int(idx), float(score), self.document_texts[idx]))

        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[int, float, str]]]:
        """Search for multiple queries at once (more efficient)."""
        if self.index is None:
            raise RuntimeError("Index not built.")

        self._load_embedder()

        query_embeddings = self.embedder.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embeddings, min(top_k, self.index.ntotal))

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for score, idx in zip(q_scores, q_indices):
                if idx == -1:
                    continue
                results.append((int(idx), float(score), self.document_texts[idx]))
            all_results.append(results)

        return all_results

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")

        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self._index_path))

        meta = {
            "num_vectors": self.index.ntotal,
            "dimension": self.config.embedding_dim,
            "document_ids": self.document_ids,
            "document_texts": self.document_texts,
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        print(f"[Indexer] Index saved to {self._index_path}")

    def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self._index_path.exists():
            raise FileNotFoundError(f"No index found at {self._index_path}")

        self.index = faiss.read_index(str(self._index_path))

        with open(self._meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.document_ids = meta["document_ids"]
        self.document_texts = meta["document_texts"]

        print(f"[Indexer] Loaded index: {self.index.ntotal} vectors")

    def get_stats(self) -> dict:
        """Return index statistics."""
        if self.index is None:
            return {"status": "no index built"}

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.config.embedding_dim,
            "index_type": "IndexFlatIP (cosine similarity)",
            "index_file_size_mb": (
                os.path.getsize(self._index_path) / 1024 / 1024
                if self._index_path.exists() else 0
            ),
        }


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    indexer = VectorIndexer()
    test_docs = [
        "Python is a programming language.",
        "FAISS enables fast similarity search.",
        "Machine learning uses algorithms to learn from data.",
        "The Eiffel Tower is located in Paris, France.",
        "Quantum computing uses qubits instead of classical bits.",
    ]
    test_ids = [f"doc_{i}" for i in range(len(test_docs))]

    indexer.build_index(test_docs, test_ids, force_rebuild=True)

    results = indexer.search("What programming language is popular?", top_k=3)
    print("\nSearch results:")
    for idx, score, text in results:
        print(f"  [{score:.4f}] {text}")
