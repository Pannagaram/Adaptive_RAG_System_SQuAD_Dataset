"""
Part 1: Document Ingestion
--------------------------
Downloads the SQuAD v1.1 dataset, extracts unique context paragraphs,
chunks them into smaller pieces, and prepares them for indexing.

Dataset stats:
  - ~19,035 unique Wikipedia paragraphs
  - ~87,599 QA pairs
  - Topics: history, science, geography, sports, arts, etc.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from config import CONFIG, DATA_DIR


@dataclass
class Document:
    """A single document chunk ready for indexing."""
    doc_id: str
    text: str
    title: str
    source: str  # "squad_train" or "squad_validation"
    paragraph_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**d)


class DocumentIngestion:
    """Handles downloading, processing, and chunking documents from SQuAD."""

    def __init__(self, config=None):
        self.config = config or CONFIG.ingestion
        self.documents: List[Document] = []
        self._cache_path = DATA_DIR / "processed_documents.json"

    def load_and_process(self, force_reload: bool = False) -> List[Document]:
        """
        Main entry point: load dataset, extract paragraphs, chunk them.
        Uses cache if available.
        """
        if not force_reload and self._cache_path.exists():
            print("[Ingestion] Loading cached documents...")
            self.documents = self._load_cache()
            print(f"[Ingestion] Loaded {len(self.documents)} document chunks from cache.")
            return self.documents

        print("[Ingestion] Downloading SQuAD v1.1 dataset...")
        raw_paragraphs = self._download_and_extract()

        print(f"[Ingestion] Extracted {len(raw_paragraphs)} unique paragraphs.")
        print("[Ingestion] Chunking documents...")
        self.documents = self._chunk_paragraphs(raw_paragraphs)

        print(f"[Ingestion] Created {len(self.documents)} document chunks.")
        self._save_cache()
        return self.documents

    def _download_and_extract(self) -> List[dict]:
        """Download SQuAD and extract unique context paragraphs."""
        from datasets import load_dataset

        dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)

        # Extract unique paragraphs (SQuAD has many questions per paragraph)
        seen_hashes = set()
        paragraphs = []

        for item in tqdm(dataset, desc="Extracting paragraphs"):
            context = item["context"].strip()
            ctx_hash = hashlib.md5(context.encode()).hexdigest()

            if ctx_hash not in seen_hashes:
                seen_hashes.add(ctx_hash)
                paragraphs.append({
                    "text": context,
                    "title": item.get("title", "Unknown"),
                    "source": f"squad_{self.config.dataset_split}",
                    "paragraph_id": ctx_hash[:12],
                })

        return paragraphs

    def _chunk_paragraphs(self, paragraphs: List[dict]) -> List[Document]:
        """Split paragraphs into overlapping chunks."""
        documents = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for para in tqdm(paragraphs, desc="Chunking"):
            text = para["text"]

            # If paragraph fits in one chunk, no splitting needed
            if len(text) <= chunk_size:
                doc = Document(
                    doc_id=f"{para['paragraph_id']}_0",
                    text=text,
                    title=para["title"],
                    source=para["source"],
                    paragraph_id=para["paragraph_id"],
                    chunk_index=0,
                    metadata={"total_chunks": 1, "char_length": len(text)},
                )
                documents.append(doc)
                continue

            # Sliding window chunking
            chunks = self._sliding_window_chunk(text, chunk_size, overlap)
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < self.config.min_chunk_length:
                    continue

                doc = Document(
                    doc_id=f"{para['paragraph_id']}_{i}",
                    text=chunk_text.strip(),
                    title=para["title"],
                    source=para["source"],
                    paragraph_id=para["paragraph_id"],
                    chunk_index=i,
                    metadata={"total_chunks": len(chunks), "char_length": len(chunk_text)},
                )
                documents.append(doc)

        return documents

    @staticmethod
    def _sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text with sliding window, trying to break at sentence boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look backwards from end for a period, question mark, or newline
                boundary = text.rfind('. ', start + chunk_size // 2, end)
                if boundary == -1:
                    boundary = text.rfind('? ', start + chunk_size // 2, end)
                if boundary == -1:
                    boundary = text.rfind('\n', start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + 1

            chunk = text[start:end]
            chunks.append(chunk)

            if end >= len(text):
                break

            start = end - overlap

        return chunks

    def _save_cache(self):
        """Save processed documents to JSON cache."""
        data = [doc.to_dict() for doc in self.documents]
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[Ingestion] Cached {len(self.documents)} documents to {self._cache_path}")

    def _load_cache(self) -> List[Document]:
        """Load documents from JSON cache."""
        with open(self._cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document.from_dict(d) for d in data]

    def get_all_texts(self) -> List[str]:
        """Return just the text content of all documents."""
        return [doc.text for doc in self.documents]

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        if not self.documents:
            return {"status": "no documents loaded"}

        texts = self.get_all_texts()
        lengths = [len(t) for t in texts]
        titles = set(doc.title for doc in self.documents)

        return {
            "total_chunks": len(self.documents),
            "unique_titles": len(titles),
            "avg_chunk_length": sum(lengths) / len(lengths),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths),
            "total_characters": sum(lengths),
        }


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingestion = DocumentIngestion()
    docs = ingestion.load_and_process()
    stats = ingestion.get_stats()
    print("\n--- Dataset Statistics ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nSample document:\n  {docs[0].text[:200]}...")
