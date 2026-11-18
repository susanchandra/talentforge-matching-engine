"""
src/embedder.py — Dense embedding model + FAISS vector index for TalentForge.

Provides dense semantic retrieval over candidate resumes using
BGE dense embeddings with inner product similarity (equivalent to cosine
similarity after L2 normalization).
"""

import os
from typing import List, Optional

import faiss
import numpy as np


class EmbeddingIndex:
    """
    Manages dense BGE embeddings and a FAISS flat inner-product index
    for semantic retrieval of candidates against a job description.

    After L2-normalizing all vectors, inner product = cosine similarity.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding index. Loads BGE model and creates FAISS IndexFlatIP.
        """
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.candidate_ids: List[str] = []
        self.dim: Optional[int] = None
        self._init_model()

    def _init_model(self):
        """Initialize transformer tokenizer and model."""
        import torch
        from transformers import AutoTokenizer, AutoModel

        # Prefer locally cached BGE models
        print(f"Loading embedding model: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

        # Infer dimension dynamically from model config
        self.dim = self._model.config.hidden_size
        self.index = faiss.IndexFlatIP(self.dim)
        print(f"EmbeddingIndex initialized (model: {self.model_name}, dim: {self.dim}).")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Encode a text string into an L2-normalized dense embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            float32 numpy array of shape (dim,), L2-normalized.
        """
        import torch

        # Truncate text to 512 tokens
        inputs = self._tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            # BGE models use [CLS] token representation (outputs[0][:, 0])
            sentence_embeddings = outputs[0][:, 0]
            # L2 normalize for exact cosine similarity via inner product
            normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            vec = normalized[0].cpu().numpy().astype(np.float32)

        return vec

    def add_candidate(self, candidate_id: str, text: str) -> None:
        """
        Embed a candidate's resume text and add it to the FAISS index.
        """
        vec = self.embed_text(text)
        self.index.add(vec.reshape(1, -1))
        self.candidate_ids.append(candidate_id)

    def search(self, query_text: str, top_k: int = 20) -> List[dict]:
        """
        Retrieve top-k candidates semantically similar to the query text.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self.embed_text(query_text).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.candidate_ids):
                results.append({
                    "candidate_id": self.candidate_ids[idx],
                    "semantic_score": round(float(score), 4),
                })

        return results
