"""
src/embedder.py — BGE-M3 embedding model + FAISS vector index for TalentForge.

Provides dense semantic retrieval over candidate resumes using the
BAAI/bge-m3 model with inner product similarity (equivalent to cosine
similarity after L2 normalization).
"""

import os
import pickle
from typing import List

import numpy as np
import faiss


class EmbeddingIndex:
    """
    Manages BAAI/bge-m3 embeddings and a FAISS flat inner-product index
    for semantic retrieval of candidates against a job description.

    After L2-normalizing all vectors, inner product = cosine similarity.
    """

    EMBEDDING_DIM = 1024  # BGE-M3 dense output dimension

    def __init__(self):
        """
        Initialize the embedding index.

        Loads the BGE-M3 model and creates an empty FAISS IndexFlatIP.
        """
        from FlagEmbedding import BGEM3FlagModel

        print("Loading BAAI/bge-m3 embedding model...")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        # FAISS inner-product index (cosine after normalization)
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)

        # Ordered list of candidate IDs corresponding to FAISS row indices
        self.candidate_ids: List[str] = []

        print("EmbeddingIndex initialized.")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Encode a text string into a normalized dense embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            float32 numpy array of shape (1024,), L2-normalized.
        """
        # BGE-M3 output dict contains 'dense_vecs'
        output = self.model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = np.array(output["dense_vecs"][0], dtype=np.float32)

        # L2 normalize for cosine similarity via inner product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec  # shape (1024,)

    def add_candidate(self, candidate_id: str, text: str) -> None:
        """
        Embed a candidate's resume text and add it to the FAISS index.

        Args:
            candidate_id: Unique identifier for this candidate.
            text: Candidate resume text to embed.
        """
        vec = self.embed_text(text)
        # FAISS requires shape (n, dim)
        self.index.add(vec.reshape(1, -1))
        self.candidate_ids.append(candidate_id)

    def search(self, query_text: str, top_k: int = 20) -> List[dict]:
        """
        Retrieve top-k candidates semantically similar to the query text.

        Args:
            query_text: Query (typically a job description or JD schema summary).
            top_k: Number of results to return.

        Returns:
            List of dicts sorted by semantic_score descending:
            [{"candidate_id": str, "semantic_score": float}, ...]
        """
        if self.index.ntotal == 0:
            return []

        actual_k = min(top_k, self.index.ntotal)
        query_vec = self.embed_text(query_text)
        scores, indices = self.index.search(query_vec.reshape(1, -1), actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            results.append({
                "candidate_id": self.candidate_ids[idx],
                "semantic_score": float(score),  # inner product ≈ cosine similarity
            })

        return results

    def save(self, path: str) -> None:
        """
        Serialize the FAISS index and candidate ID list to disk.

        Args:
            path: Directory path to save index files.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "candidate_ids.pkl"), "wb") as f:
            pickle.dump(self.candidate_ids, f)
        print(f"Index saved to {path} ({self.index.ntotal} vectors).")

    def load(self, path: str) -> None:
        """
        Deserialize the FAISS index and candidate ID list from disk.

        Args:
            path: Directory path containing saved index files.
        """
        index_path = os.path.join(path, "faiss.index")
        ids_path = os.path.join(path, "candidate_ids.pkl")

        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            raise FileNotFoundError(f"Index files not found in: {path}")

        self.index = faiss.read_index(index_path)
        with open(ids_path, "rb") as f:
            self.candidate_ids = pickle.load(f)

        print(f"Index loaded from {path} ({self.index.ntotal} vectors).")
