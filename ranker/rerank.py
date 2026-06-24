"""
Module 8: Cross-encoder Rerank + Multi-axis Semantic Combiner
============================================================================

Two-stage semantic scoring on the top-500 candidates from coarse recall:

    Stage B: Cross-encoder rerank
        - bge-reranker-base (preferred, requires sentence-transformers)
        - Falls back to bi-encoder cosine (SentenceTransformerBackend) if cross-encoder unavailable
        - Falls back to multi-field cosine (TfidfSvdBackend) if neither available
        - Outputs cross_enc_score in [0, 1]

    Stage C: Multi-axis semantic score
        - Compute cosine similarity between candidate field embeddings and intent axis embeddings
        - 6 axes: title_fit, career_retrieval, career_ranking, career_llm, skills_coverage, cross_enc
        - Weighted sum with weights from config/field_weights.yaml (semantic_axis_weights)
        - MUST sum to 1.0 (asserted at runtime)

Output: semantic_score ∈ [0, 1] per candidate, fed into the final combiner.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .types import Candidate, Intent

log = logging.getLogger(__name__)


# ─── Paths ────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
FIELD_WEIGHTS_PATH = CONFIG_DIR / "field_weights.yaml"


# ─── Cross-encoder backend ────────────────────────────────────────────────
class CrossEncoderBackend:
    """
    Cross-encoder reranker backend.

    Backends (tried in order):
        1. sentence-transformers CrossEncoder with bge-reranker-base (preferred)
        2. Bi-encoder cosine similarity (SentenceTransformerBackend) as fallback
        3. Multi-field cosine (TfidfSvdBackend) as last-resort fallback

    The fallbacks produce slightly lower quality but allow the pipeline to run
    end-to-end without sentence-transformers installed.
    """

    def __init__(self, backend_type: str = "auto", model_name: str = "BAAI/bge-reranker-base"):
        self.backend_type = backend_type
        self.model_name = model_name
        self._cross_encoder = None
        self._bi_encoder = None
        self._fitted_tfidf = None
        self.dim = 384

        if backend_type in ("cross-encoder", "auto"):
            try:
                from sentence_transformers import CrossEncoder
                log.info(f"Loading cross-encoder: {model_name}")
                self._cross_encoder = CrossEncoder(model_name)
                self.backend_type = "cross-encoder"
                log.info("Cross-encoder loaded successfully")
                return
            except ImportError:
                if backend_type == "cross-encoder":
                    raise
                log.info("sentence-transformers not available, falling back to bi-encoder cosine")

        if backend_type in ("bi-encoder", "auto"):
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"Loading bi-encoder for cosine rerank: BAAI/bge-small-en-v1.5")
                self._bi_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
                self.dim = self._bi_encoder.get_sentence_embedding_dimension()
                self.backend_type = "bi-encoder"
                return
            except ImportError:
                if backend_type == "bi-encoder":
                    raise
                log.info("Bi-encoder unavailable, falling back to TF-IDF cosine")

        # Last-resort: TF-IDF + SVD (deterministic, no network)
        # Will be lazily fitted on first batch
        log.warning("Using TF-IDF fallback for rerank. Quality is lower than cross-encoder.")
        self.backend_type = "tfidf"
        from .encoder import TfidfSvdBackend
        self._tfidf_backend = TfidfSvdBackend(dim=384)

    def score_pairs(self, query: str, documents: list[str]) -> np.ndarray:
        """
        Score (query, document) pairs.

        Args:
            query: single query string
            documents: list of document strings

        Returns:
            np.ndarray of shape (N,) with scores in [0, 1]
        """
        if not documents:
            return np.array([], dtype=np.float32)

        if self.backend_type == "cross-encoder":
            # Cross-encoder: predict scores for (query, doc) pairs
            pairs = [(query, doc) for doc in documents]
            raw_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
            # Sigmoid to normalize to [0, 1]
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
            return scores.astype(np.float32)

        elif self.backend_type == "bi-encoder":
            # Bi-encoder: cosine similarity between query and document embeddings
            query_vec = self._bi_encoder.encode(query, normalize_embeddings=True, show_progress_bar=False)
            doc_vecs = self._bi_encoder.encode(documents, normalize_embeddings=True,
                                                show_progress_bar=False, batch_size=32)
            # Cosine similarity (vectors are already normalized)
            scores = (doc_vecs @ query_vec).astype(np.float32)
            # Clip to [0, 1] — cosine of normalized vectors can be negative
            scores = np.clip(scores, 0.0, 1.0)
            return scores

        else:  # tfidf
            # TF-IDF cosine similarity
            if not self._tfidf_backend._fitted:
                # Fit on query + all documents
                self._tfidf_backend.fit([query] + documents)
            query_vec = self._tfidf_backend.encode(query)
            doc_vecs = np.stack([self._tfidf_backend.encode(doc) for doc in documents])
            # Cosine similarity
            scores = (doc_vecs @ query_vec).astype(np.float32)
            scores = np.clip(scores, 0.0, 1.0)
            return scores


# ─── Multi-axis semantic scorer ───────────────────────────────────────────
class MultiAxisSemanticScorer:
    """
    Computes multi-axis semantic score per candidate.

    For each candidate, computes cosine similarity between candidate field
    embeddings and intent axis embeddings:

        title_fit          = cosine(candidate.title_emb, intent.title_archetype_emb)
        career_retrieval   = cosine(candidate.career_emb, intent.embeddings_retrieval_production_emb)
        career_ranking     = cosine(candidate.career_emb, intent.ranking_eval_frameworks_emb)
        career_llm         = cosine(candidate.career_emb, intent.llm_engineering_emb)
        skills_coverage    = jaccard_weighted(candidate.matched_canonical_skills, jd_required_skills)
        cross_enc          = cross-encoder score

    Final semantic_score = weighted sum (weights from config/field_weights.yaml).
    """

    def __init__(self, intent: Intent, axis_weights: Optional[dict] = None):
        self.intent = intent
        if axis_weights is None:
            axis_weights = self._default_axis_weights()
        # Verify weights sum to 1.0
        total = sum(axis_weights.values())
        assert abs(total - 1.0) < 1e-6, f"semantic_axis_weights must sum to 1.0 (got {total})"
        self.axis_weights = axis_weights

    @staticmethod
    def _default_axis_weights() -> dict:
        if FIELD_WEIGHTS_PATH.exists():
            with open(FIELD_WEIGHTS_PATH) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("semantic_axis_weights", {
                "title_fit": 0.25,
                "career_retrieval": 0.20,
                "career_ranking": 0.15,
                "career_llm": 0.10,
                "skills_coverage": 0.10,
                "cross_enc": 0.20,
            })
        return {
            "title_fit": 0.25,
            "career_retrieval": 0.20,
            "career_ranking": 0.15,
            "career_llm": 0.10,
            "skills_coverage": 0.10,
            "cross_enc": 0.20,
        }

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
        if a is None or b is None:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score(
        self,
        candidate: Candidate,
        candidate_embeddings: dict[str, np.ndarray],
        cross_enc_score: float,
        esco_coverage_score: float,
    ) -> tuple[float, dict]:
        """
        Compute multi-axis semantic score for one candidate.

        Args:
            candidate: Candidate object
            candidate_embeddings: {field: np.ndarray} for this candidate
            cross_enc_score: float in [0, 1] from CrossEncoderBackend
            esco_coverage_score: float in [0, 1] from SkillTaxonomyResolver

        Returns:
            (semantic_score in [0, 1], axis_scores dict for reasoning)
        """
        axis_scores = {}

        # title_fit: cosine(title_emb, title_archetype_emb)
        title_arch_emb = self.intent.axis_embeddings.get("title_archetype")
        axis_scores["title_fit"] = self._cosine(
            candidate_embeddings.get("title"), title_arch_emb
        )

        # career_retrieval: cosine(career_emb, embeddings_retrieval_production_emb)
        retrieval_emb = self.intent.axis_embeddings.get("embeddings_retrieval_production")
        # Also try core_retrieval_ir skill group
        core_retrieval_emb = self.intent.axis_embeddings.get("core_retrieval_ir")
        retrieval_score = max(
            self._cosine(candidate_embeddings.get("career"), retrieval_emb),
            self._cosine(candidate_embeddings.get("career"), core_retrieval_emb) if core_retrieval_emb is not None else 0.0,
        )
        axis_scores["career_retrieval"] = retrieval_score

        # career_ranking: cosine(career_emb, ranking_eval_frameworks_emb)
        ranking_emb = self.intent.axis_embeddings.get("ranking_eval_frameworks")
        core_ranking_emb = self.intent.axis_embeddings.get("core_ranking_eval")
        ranking_score = max(
            self._cosine(candidate_embeddings.get("career"), ranking_emb),
            self._cosine(candidate_embeddings.get("career"), core_ranking_emb) if core_ranking_emb is not None else 0.0,
        )
        axis_scores["career_ranking"] = ranking_score

        # career_llm: cosine(career_emb, llm_engineering_emb)
        llm_emb = self.intent.axis_embeddings.get("llm_engineering")
        axis_scores["career_llm"] = self._cosine(
            candidate_embeddings.get("career"), llm_emb
        )

        # skills_coverage: use provided ESCO coverage score
        axis_scores["skills_coverage"] = float(esco_coverage_score)

        # cross_enc: use provided cross-encoder score
        axis_scores["cross_enc"] = float(cross_enc_score)

        # Weighted sum
        semantic_score = sum(self.axis_weights[axis] * axis_scores[axis] for axis in self.axis_weights)

        # Clip to [0, 1]
        semantic_score = max(0.0, min(1.0, semantic_score))

        return float(semantic_score), axis_scores


# ─── Rerank orchestrator ──────────────────────────────────────────────────
class Reranker:
    """
    Orchestrates cross-encoder rerank + multi-axis semantic scoring.

    Usage:
        reranker = Reranker.from_intent(intent)
        results = reranker.rerank(candidates, candidate_embeddings, query_text, top_k=500)
    """

    def __init__(
        self,
        cross_encoder: CrossEncoderBackend,
        semantic_scorer: MultiAxisSemanticScorer,
        skill_resolver,
        jd_required_skills: list[str],
    ):
        self.cross_encoder = cross_encoder
        self.semantic_scorer = semantic_scorer
        self.skill_resolver = skill_resolver
        self.jd_required_skills = jd_required_skills

    @classmethod
    def from_intent(cls, intent: Intent) -> "Reranker":
        """Build reranker from intent (loads cross-encoder + skill resolver)."""
        cross_encoder = CrossEncoderBackend(backend_type="auto")
        semantic_scorer = MultiAxisSemanticScorer(intent=intent)
        from .features import SkillTaxonomyResolver
        from .intent import get_jd_required_canonical_skills
        skill_resolver = SkillTaxonomyResolver.from_config()
        jd_required = get_jd_required_canonical_skills(intent.schema)
        return cls(cross_encoder, semantic_scorer, skill_resolver, jd_required)

    def _build_query_text(self, intent: Intent) -> str:
        """Build query text for cross-encoder from intent."""
        parts = [intent.schema.get("jd_title", "Senior AI Engineer")]
        # Top must-have requirement texts
        for req in intent.schema.get("explicit_positive_requirements", []):
            if req.get("must_have") and req.get("embedding_query_text"):
                parts.append(req["embedding_query_text"])
        return " ".join(parts[:3])  # cap to keep query focused

    def _build_candidate_doc(self, candidate: Candidate) -> str:
        """Build document text for cross-encoder from candidate."""
        parts = [
            candidate.title or "",
            candidate.headline or "",
            candidate.summary or "",
        ]
        # Top 2 career descriptions
        for job in candidate.career_history[:2]:
            desc = job.get("description", "")
            if desc:
                parts.append(desc[:500])  # cap each description
        return " ".join(parts)

    def rerank(
        self,
        candidates: list[Candidate],
        candidate_embeddings: dict[str, dict[str, np.ndarray]],
        intent: Intent,
        top_k: int = 500,
    ) -> list[dict]:
        """
        Rerank top candidates with cross-encoder + multi-axis semantic scoring.

        Args:
            candidates: list of Candidate objects (typically top-1000 from coarse recall)
            candidate_embeddings: {candidate_id: {field: np.ndarray}}
            intent: Intent object with axis embeddings
            top_k: number of candidates to rerank (top-K from input)

        Returns:
            list of dicts with candidate_id, semantic_score, axis_scores (sorted desc)
        """
        t0 = time.time()
        candidates = candidates[:top_k]
        log.info(f"Reranking {len(candidates)} candidates with {self.cross_encoder.backend_type} backend")

        # Stage B: Cross-encoder scores
        query_text = self._build_query_text(intent)
        documents = [self._build_candidate_doc(c) for c in candidates]
        cross_enc_scores = self.cross_encoder.score_pairs(query_text, documents)
        log.info(f"Cross-encoder scored {len(candidates)} pairs in {time.time() - t0:.1f}s")

        # Stage C: Multi-axis semantic scores
        results = []
        for i, c in enumerate(candidates):
            embeddings = candidate_embeddings.get(c.candidate_id, {})
            if not embeddings:
                log.warning(f"No embeddings for {c.candidate_id}, using zeros")
                embeddings = {field: np.zeros(self.cross_encoder.dim, dtype=np.float32)
                              for field in ("title", "summary", "career", "skills", "combined")}

            # ESCO coverage for this candidate
            matched_skills = self.skill_resolver.resolve(c)
            esco_score, _ = self.skill_resolver.coverage_score(matched_skills, self.jd_required_skills)

            semantic_score, axis_scores = self.semantic_scorer.score(
                candidate=c,
                candidate_embeddings=embeddings,
                cross_enc_score=float(cross_enc_scores[i]),
                esco_coverage_score=esco_score,
            )

            results.append({
                "candidate_id": c.candidate_id,
                "semantic_score": semantic_score,
                "axis_scores": axis_scores,
                "cross_enc_score": float(cross_enc_scores[i]),
                "esco_coverage": esco_score,
                "matched_canonical_skills": sorted(matched_skills),
            })

        # Sort by semantic_score descending
        results.sort(key=lambda r: -r["semantic_score"])

        log.info(f"Rerank complete in {time.time() - t0:.1f}s")
        return results
