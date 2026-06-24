"""
Module 7: Field-weighted BM25 + FAISS + RRF Recall
============================================================================

Two-stage coarse recall:

    Stage A: Per-field BM25 (4 separate BM25Okapi instances)
        - title BM25 (weight 4.0)
        - skills BM25 (weight 2.0)
        - career BM25 (weight 2.5)
        - summary BM25 (weight 1.0)
        Per-field scores z-scored, then weighted sum.

    Stage B: Dense FAISS inner-product over combined_emb (single vector per candidate)

    Stage C: Reciprocal Rank Fusion (RRF, k=60) of Stage A + Stage B
        - Returns top-1000 candidate IDs for downstream rerank

Why per-field BM25 (not single-blob):
    A candidate who mentions "AI Engineer" once in their summary (but is actually
    a Marketing Manager) gets the same blob-BM25 score as a real AI Engineer whose
    TITLE is "AI Engineer". Per-field BM25 prevents this by giving title field 4x
    weight — the Marketing Manager scores low on title-BM25 even if they score
    high on skills-BM25.

Runtime budget (100K candidates, CPU):
    - Build 4 BM25 indices: ~3s
    - Build FAISS index: ~2s
    - RRF fusion: ~1s
    - Total: ~6s (well within budget)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .types import Candidate

log = logging.getLogger(__name__)


# ─── Paths ────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
FIELD_WEIGHTS_PATH = CONFIG_DIR / "field_weights.yaml"


# ─── Tokenizer (lightweight, no NLTK dependency) ──────────────────────────
def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric, filter empty + stopwords.
    Avoids NLTK dependency for the constrained ranking environment.
    """
    if not text:
        return []
    import re
    # Split on non-alphanumeric (keep digits and letters)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Light stopword list (common English)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their", "as", "if", "so",
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


# ─── Per-field BM25 wrapper ───────────────────────────────────────────────
class FieldBM25:
    """
    Maintains 4 separate BM25Okapi indices (title, skills, career, summary).

    Usage:
        bm25 = FieldBM25.from_candidates(candidates)
        scores = bm25.score(query_tokens)  # returns dict {candidate_id: bm25_score}
    """

    FIELDS = ("title", "skills", "career", "summary")

    def __init__(
        self,
        candidate_ids: list[str],
        field_docs: dict[str, list[list[str]]],  # {field: [tokenized_doc per candidate]}
        field_weights: dict[str, float],
    ):
        self.candidate_ids = candidate_ids
        self.field_docs = field_docs
        self.field_weights = field_weights
        # Lazy import (rank_bm25 is optional in fallback environments)
        from rank_bm25 import BM25Okapi
        self.bm25_instances: dict[str, "BM25Okapi"] = {}
        for field in self.FIELDS:
            self.bm25_instances[field] = BM25Okapi(field_docs[field])
        log.info(f"FieldBM25 built {len(self.FIELDS)} indices for {len(candidate_ids)} candidates")

    @classmethod
    def from_candidates(
        cls,
        candidates: list[Candidate],
        field_weights: Optional[dict] = None,
    ) -> "FieldBM25":
        """Build per-field BM25 indices from candidate list."""
        if field_weights is None:
            field_weights = cls._default_field_weights()

        candidate_ids = [c.candidate_id for c in candidates]
        field_docs = {field: [] for field in cls.FIELDS}

        for c in candidates:
            # Title field: current_title + headline
            title_text = f"{c.title or ''} {c.headline or ''}".strip()
            field_docs["title"].append(tokenize(title_text))

            # Skills field: skill names (with repetition weighted by proficiency)
            from .encoder import extract_skills_text
            field_docs["skills"].append(tokenize(extract_skills_text(c)))

            # Career field: concatenated career_history descriptions
            from .encoder import extract_career_text
            field_docs["career"].append(tokenize(extract_career_text(c)))

            # Summary field: profile.summary
            field_docs["summary"].append(tokenize(c.summary or ""))

        return cls(candidate_ids, field_docs, field_weights)

    @staticmethod
    def _default_field_weights() -> dict:
        """Load from config/field_weights.yaml, fall back to hardcoded defaults."""
        if FIELD_WEIGHTS_PATH.exists():
            with open(FIELD_WEIGHTS_PATH) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("bm25_field_weights", {
                "title": 4.0, "skills": 2.0, "career": 2.5, "summary": 1.0,
            })
        return {"title": 4.0, "skills": 2.0, "career": 2.5, "summary": 1.0}

    def score(self, query_tokens: list[str]) -> np.ndarray:
        """
        Score all candidates against a tokenized query.
        Returns weighted sum of z-scored per-field BM25 scores.

        Args:
            query_tokens: list of tokens (use tokenize() to produce)

        Returns:
            np.ndarray of shape (N,) — combined BM25 score per candidate
        """
        if not query_tokens:
            return np.zeros(len(self.candidate_ids), dtype=np.float32)

        # Get raw BM25 scores per field
        field_scores = {}
        for field in self.FIELDS:
            raw_scores = self.bm25_instances[field].get_scores(query_tokens)
            field_scores[field] = np.asarray(raw_scores, dtype=np.float32)

        # Z-score normalize each field (handles scale differences)
        # If a field has zero variance (all zeros), use raw scores
        z_scored = {}
        for field, scores in field_scores.items():
            mean = scores.mean()
            std = scores.std()
            if std > 1e-9:
                z_scored[field] = (scores - mean) / std
            else:
                z_scored[field] = scores - mean  # all zeros effectively

        # Weighted sum
        combined = np.zeros(len(self.candidate_ids), dtype=np.float32)
        for field in self.FIELDS:
            combined += self.field_weights[field] * z_scored[field]

        return combined


# ─── FAISS wrapper ────────────────────────────────────────────────────────
class DenseIndex:
    """
    FAISS IndexFlatIP wrapper for inner-product (cosine) search over
    L2-normalized candidate embeddings.

    Usage:
        index = DenseIndex.from_embeddings(candidate_ids, combined_embeddings)
        scores, indices = index.search(query_embedding, k=1000)
    """

    def __init__(self, candidate_ids: list[str], index):
        self.candidate_ids = candidate_ids
        self.index = index
        self.dim = index.d

    @classmethod
    def from_embeddings(
        cls,
        candidate_ids: list[str],
        combined_embeddings: np.ndarray,
    ) -> "DenseIndex":
        """
        Build FAISS IndexFlatIP from L2-normalized embeddings.

        Args:
            candidate_ids: list of candidate_id strings (length N)
            combined_embeddings: (N, dim) array of L2-normalized vectors

        Returns:
            DenseIndex instance
        """
        import faiss
        n, dim = combined_embeddings.shape
        assert len(candidate_ids) == n, f"Mismatch: {len(candidate_ids)} IDs vs {n} embeddings"

        # Ensure float32 (FAISS requirement)
        embeddings = combined_embeddings.astype(np.float32, copy=False)
        # Re-normalize to be safe (in case caller forgot)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        log.info(f"DenseIndex built: {n} vectors, dim={dim}")

        return cls(candidate_ids, index)

    def search(self, query: np.ndarray, k: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k candidates by inner product.

        Args:
            query: (dim,) or (1, dim) L2-normalized query vector
            k: number of results

        Returns:
            (scores, candidate_indices) where indices are positions in candidate_ids
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32, copy=False)
        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]

    def search_by_candidate(self, candidate_idx: int, k: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Search using an existing candidate's embedding (by index)."""
        import faiss
        # Reconstruct vector from index
        vec = self.index.reconstruct(candidate_idx)
        return self.search(vec, k)


# ─── RRF Fusion ───────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    k: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reciprocal Rank Fusion of BM25 and dense scores.

    RRF score = sum(1 / (k + rank_i)) for each retriever i.

    Args:
        bm25_scores: (N,) array of BM25 scores (higher = better)
        dense_scores: (N,) array of dense retrieval scores (higher = better)
        k: RRF constant (default 60, standard value)

    Returns:
        (rrf_scores, candidate_indices_sorted) where indices are positions in
        the input arrays, sorted by RRF score descending.
    """
    n = len(bm25_scores)
    assert len(dense_scores) == n

    # Get ranks (0 = best)
    bm25_ranks = np.argsort(-bm25_scores).argsort()
    dense_ranks = np.argsort(-dense_scores).argsort()

    # RRF score = 1/(k + rank_bm25) + 1/(k + rank_dense)
    rrf_scores = 1.0 / (k + bm25_ranks) + 1.0 / (k + dense_ranks)

    # Sort by RRF score descending
    sorted_indices = np.argsort(-rrf_scores)
    return rrf_scores, sorted_indices


# ─── Coarse Recall orchestrator ───────────────────────────────────────────
class CoarseRecall:
    """
    Combines FieldBM25 + DenseIndex + RRF for coarse recall.

    Usage:
        recall = CoarseRecall.build(candidates, candidate_embeddings)
        top_k_indices, rrf_scores = recall.recall(query_text, query_embedding, k=1000)
    """

    def __init__(self, bm25: FieldBM25, dense: DenseIndex, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    @classmethod
    def build(
        cls,
        candidates: list[Candidate],
        candidate_embeddings: dict[str, dict[str, np.ndarray]],
        rrf_k: Optional[int] = None,
    ) -> "CoarseRecall":
        """
        Build the coarse recall system from candidates + their embeddings.

        Args:
            candidates: list of Candidate objects
            candidate_embeddings: {candidate_id: {field: np.ndarray}} — must include 'combined'
            rrf_k: RRF constant (default: load from config or 60)

        Returns:
            CoarseRecall instance
        """
        if rrf_k is None:
            rrf_k = cls._default_rrf_k()

        # Build BM25
        t0 = time.time()
        bm25 = FieldBM25.from_candidates(candidates)
        log.info(f"BM25 built in {time.time() - t0:.1f}s")

        # Build dense index
        t1 = time.time()
        candidate_ids = [c.candidate_id for c in candidates]
        combined_matrix = np.stack([
            candidate_embeddings[cid]["combined"] for cid in candidate_ids
        ])
        dense = DenseIndex.from_embeddings(candidate_ids, combined_matrix)
        log.info(f"Dense index built in {time.time() - t1:.1f}s")

        return cls(bm25, dense, rrf_k)

    @staticmethod
    def _default_rrf_k() -> int:
        if FIELD_WEIGHTS_PATH.exists():
            with open(FIELD_WEIGHTS_PATH) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("rrf_k", 60)
        return 60

    def recall(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 1000,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run coarse recall: BM25 + dense + RRF fusion.

        Args:
            query_text: text query (tokenized for BM25)
            query_embedding: (dim,) L2-normalized query vector for dense retrieval
            k: number of candidates to return

        Returns:
            (top_k_candidate_ids, rrf_scores) — both sorted by RRF score descending
        """
        t0 = time.time()

        # Stage A: BM25
        query_tokens = tokenize(query_text)
        bm25_scores = self.bm25.score(query_tokens)

        # Stage B: Dense
        dense_scores, _ = self.dense.search(query_embedding, k=len(self.bm25.candidate_ids))

        # Stage C: RRF fusion
        rrf_scores, sorted_indices = reciprocal_rank_fusion(bm25_scores, dense_scores, k=self.rrf_k)

        # Take top-k
        top_k_indices = sorted_indices[:k]
        top_k_scores = rrf_scores[top_k_indices]
        top_k_ids = [self.bm25.candidate_ids[i] for i in top_k_indices]

        log.info(f"Coarse recall completed in {time.time() - t0:.2f}s "
                 f"(query: '{query_text[:60]}...', k={k})")

        return top_k_ids, top_k_scores


# ─── Convenience: build query for coarse recall ───────────────────────────
def build_jd_query(intent_schema: dict) -> str:
    """
    Build a query string for coarse recall from the JD intent schema.
    Combines:
        - JD title
        - Top must-have requirement embedding_query_texts
        - Skill taxonomy group embedding_query_texts
    """
    parts = [intent_schema.get("jd_title", "Senior AI Engineer")]

    # Top must-have requirements
    for req in intent_schema.get("explicit_positive_requirements", []):
        if req.get("must_have") and req.get("embedding_query_text"):
            parts.append(req["embedding_query_text"])

    # Top skill taxonomy groups
    for group_name in ("core_retrieval_ir", "core_ranking_eval", "core_python_eng"):
        group = intent_schema.get("skill_taxonomy_groups", {}).get(group_name, {})
        if group.get("embedding_query_text"):
            parts.append(group["embedding_query_text"])

    # Ideal titles (for title-field BM25 boost)
    title_match = intent_schema.get("title_archetype_match", {})
    ideal_titles = title_match.get("ideal_titles", [])
    parts.extend(ideal_titles[:5])

    return " ".join(parts)
