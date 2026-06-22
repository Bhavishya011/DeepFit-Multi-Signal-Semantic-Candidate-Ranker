"""
Module 2: Multi-field Candidate Encoder
============================================================================

For each candidate, produces 5 separate embeddings (all L2-normalized):
    1. title_emb    — profile.current_title + headline
    2. summary_emb  — profile.summary
    3. career_emb   — concatenated career_history[].description
    4. skills_emb   — weighted skills text (proficiency × log(endorsements+1))
    5. combined_emb — L2-normalized weighted sum of the above (for FAISS coarse recall)

Encoder backend selection (auto):
    1. Try `sentence-transformers` with `bge-small-en-v1.5` (preferred, ~384-dim)
    2. If unavailable, fall back to `sklearn TfidfVectorizer + TruncatedSVD`
       (deterministic, no model download, also 384-dim, lower quality but
       works end-to-end on dev set without network access)

The fallback is clearly logged so we never silently ship a degraded encoder
to production.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBED_DIM = 384
SKILL_PROFICIENCY_WEIGHTS = {
    "beginner": 0.5,
    "intermediate": 1.0,
    "advanced": 1.5,
    "expert": 2.0,
}


# ─── Encoder interface ────────────────────────────────────────────────────
class EncoderBackend:
    """Abstract encoder backend — produces L2-normalized embeddings."""

    name: str = "abstract"
    dim: int = DEFAULT_EMBED_DIM

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Default impl calls encode() in a loop; subclasses may override."""
        return np.stack([self.encode(t) for t in texts])


# ─── Backend 1: sentence-transformers (preferred) ─────────────────────────
class SentenceTransformerBackend(EncoderBackend):
    name = "sentence-transformers"
    dim = DEFAULT_EMBED_DIM

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, cache_dir: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from e
        log.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info(f"Loaded. Embedding dim = {self.dim}")

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        # Replace empty strings with a single space to avoid model warnings
        cleaned = [t if t and t.strip() else " " for t in texts]
        vecs = self.model.encode(cleaned, normalize_embeddings=True,
                                  show_progress_bar=False, batch_size=64)
        return vecs.astype(np.float32)


# ─── Backend 2: TF-IDF + SVD fallback (deterministic, no network) ─────────
class TfidfSvdBackend(EncoderBackend):
    """
    Deterministic TF-IDF + TruncatedSVD encoder.

    Produces 384-dim L2-normalized vectors. Quality is lower than BGE but
    the encoder works end-to-end on the dev set without any model download.
    Used for development and CI tests where sentence-transformers is unavailable.

    The vocabulary is fit lazily on first batch of texts (or on a corpus
    passed to fit()). For reproducibility, we use a fixed random_state.
    """

    name = "tfidf-svd"
    dim = DEFAULT_EMBED_DIM

    def __init__(self, dim: int = DEFAULT_EMBED_DIM):
        log.warning(
            "Using TF-IDF + SVD fallback encoder. Quality is lower than BGE-small. "
            "For production: pip install sentence-transformers and re-run precompute."
        )
        self._init_vectorizer()
        self._init_svd(dim)
        self._fitted = False

    def _init_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            stop_words="english",
            sublinear_tf=True,
        )

    def _init_svd(self, dim: int):
        from sklearn.decomposition import TruncatedSVD
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.dim = dim

    def fit(self, corpus: list[str]):
        """Fit the TF-IDF vocabulary + SVD on a corpus."""
        log.info(f"Fitting TF-IDF + SVD on {len(corpus)} documents...")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        # TruncatedSVD requires n_components < min(n_samples, n_features)
        # If corpus is small, cap at min(n_features, n_samples) - 1
        max_components = min(tfidf_matrix.shape[0] - 1,
                            tfidf_matrix.shape[1],
                            self.dim)
        if max_components < self.dim:
            log.warning(
                f"Corpus too small for {self.dim}-dim SVD. "
                f"Reducing to {max_components} components (quality will be lower). "
                f"For full {self.dim}-dim embeddings, use sentence-transformers backend."
            )
            self._init_svd(max_components)
        self.svd.fit(tfidf_matrix)
        self._fitted = True
        log.info(f"Fit complete. Vocab size = {len(self.vectorizer.vocabulary_)}, "
                 f"SVD components = {self.svd.n_components}")

    def encode(self, text: str) -> np.ndarray:
        if not self._fitted:
            # If not fitted, fit on the single text (degenerate but won't crash)
            self.fit([text] if text else ["empty"])
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        tfidf_vec = self.vectorizer.transform([text])
        svd_vec = self.svd.transform(tfidf_vec)[0]
        # L2-normalize
        norm = np.linalg.norm(svd_vec)
        if norm > 0:
            svd_vec = svd_vec / norm
        return svd_vec.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            # Fit on the batch itself (degenerate but works for smoke tests)
            self.fit(texts if texts else ["empty"])
        cleaned = [t if t and t.strip() else " " for t in texts]
        tfidf_matrix = self.vectorizer.transform(cleaned)
        svd_matrix = self.svd.transform(tfidf_matrix)
        # L2-normalize each row
        norms = np.linalg.norm(svd_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        svd_matrix = svd_matrix / norms
        return svd_matrix.astype(np.float32)


# ─── Backend factory ──────────────────────────────────────────────────────
def get_encoder(
    preferred: str = "auto",
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = None,
    svd_dim: int = DEFAULT_EMBED_DIM,
) -> EncoderBackend:
    """
    Auto-select encoder backend.

    Args:
        preferred: "sentence-transformers" | "tfidf-svd" | "auto"
            "auto" tries sentence-transformers first, falls back to tfidf-svd
        model_name: HuggingFace model name for sentence-transformers
        cache_dir: optional cache dir for model download
        svd_dim: dim for TF-IDF fallback

    Returns:
        EncoderBackend instance
    """
    if preferred == "sentence-transformers" or preferred == "auto":
        try:
            return SentenceTransformerBackend(model_name=model_name, cache_dir=cache_dir)
        except ImportError:
            if preferred == "sentence-transformers":
                raise
            log.info("sentence-transformers not available, falling back to TF-IDF + SVD")
    return TfidfSvdBackend(dim=svd_dim)


# ─── Field-text extractors ────────────────────────────────────────────────
def extract_title_text(candidate) -> str:
    """Title + headline — highest weight field."""
    title = candidate.title or ""
    headline = candidate.headline or ""
    return f"{title}. {headline}".strip()


def extract_summary_text(candidate) -> str:
    return candidate.summary or ""


def extract_career_text(candidate) -> str:
    """Concatenated career_history descriptions — production evidence lives here."""
    parts = []
    for job in candidate.career_history:
        desc = job.get("description", "")
        if desc:
            parts.append(desc)
    return " ".join(parts)


def extract_skills_text(candidate) -> str:
    """
    Weighted skills text — each skill repeated N times based on proficiency
    and log(endorsements+1). This makes high-endorsement expert skills
    contribute more to the embedding than low-endorsement beginner ones.
    """
    parts = []
    for skill in candidate.skills:
        name = skill.get("name", "")
        if not name:
            continue
        proficiency = skill.get("proficiency", "intermediate")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        weight = SKILL_PROFICIENCY_WEIGHTS.get(proficiency, 1.0)
        # log(endorsements + 1) capped at log(101) ≈ 4.6
        endorsement_factor = 1.0 + math.log(min(endorsements + 1, 100) + 1)
        # Duration factor: 1.0 + min(duration / 24, 2.0) → 1.0 to 3.0
        duration_factor = 1.0 + min(duration / 24.0, 2.0)

        repeat = max(1, int(weight * endorsement_factor * duration_factor / 3))
        parts.append(" ".join([name] * repeat))
    return " ".join(parts)


# ─── Multi-field encoder ──────────────────────────────────────────────────
class MultiFieldEncoder:
    """
    Encodes candidates into 5 fields: title, summary, career, skills, combined.

    Usage:
        encoder = MultiFieldEncoder.from_auto()
        embeddings = encoder.encode_candidates(candidates)
        # embeddings is a dict: {candidate_id: {field: np.ndarray}}
    """

    FIELDS = ("title", "summary", "career", "skills", "combined")

    def __init__(
        self,
        backend: EncoderBackend,
        field_weights: Optional[dict] = None,
    ):
        self.backend = backend
        # Default field weights from config/field_weights.yaml
        if field_weights is None:
            field_weights = self._default_field_weights()
        # Normalize weights to sum to 1.0
        total = sum(field_weights.values())
        self.field_weights = {k: v / total for k, v in field_weights.items()}
        log.info(f"MultiFieldEncoder initialized. Backend: {backend.name}, "
                 f"dim: {backend.dim}, field_weights: {self.field_weights}")

    @staticmethod
    def _default_field_weights() -> dict:
        """Load from config/field_weights.yaml, fall back to hardcoded defaults."""
        config_path = Path(__file__).parent.parent / "config" / "field_weights.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("combined_embedding_weights", {
                "title": 3.0, "summary": 1.5, "career": 2.0, "skills": 1.0,
            })
        return {"title": 3.0, "summary": 1.5, "career": 2.0, "skills": 1.0}

    @classmethod
    def from_auto(
        cls,
        preferred: str = "auto",
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[str] = None,
        field_weights: Optional[dict] = None,
    ) -> "MultiFieldEncoder":
        backend = get_encoder(preferred=preferred, model_name=model_name, cache_dir=cache_dir)
        return cls(backend=backend, field_weights=field_weights)

    # ─── Single-candidate encoding ────────────────────────────────────────
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string (used by intent axis encoding)."""
        return self.backend.encode(text)

    def encode_candidate(self, candidate) -> dict[str, np.ndarray]:
        """
        Encode one candidate into 5 field embeddings.

        Returns:
            dict with keys: title, summary, career, skills, combined
            Each value is an L2-normalized np.ndarray of shape (dim,)
        """
        texts = {
            "title":   extract_title_text(candidate),
            "summary": extract_summary_text(candidate),
            "career":  extract_career_text(candidate),
            "skills":  extract_skills_text(candidate),
        }
        # Batch encode for efficiency
        field_names = list(texts.keys())
        field_text_list = [texts[f] for f in field_names]
        field_vectors = self.backend.encode_batch(field_text_list)

        embeddings = {}
        for i, fname in enumerate(field_names):
            vec = field_vectors[i]
            # Ensure L2-normalized
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings[fname] = vec.astype(np.float32)

        # Build combined embedding as weighted sum of L2-normalized field vectors
        combined = np.zeros(self.backend.dim, dtype=np.float32)
        for fname in field_names:
            combined += self.field_weights[fname] * embeddings[fname]
        # L2-normalize combined
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        embeddings["combined"] = combined.astype(np.float32)

        return embeddings

    # ─── Batch encoding ───────────────────────────────────────────────────
    def encode_candidates(self, candidates, show_progress: bool = True) -> dict[str, dict[str, np.ndarray]]:
        """
        Encode a list of candidates. Returns:
            {candidate_id: {field: np.ndarray}}
        """
        # If using TF-IDF backend, fit on the full corpus first (one-time)
        if isinstance(self.backend, TfidfSvdBackend) and not self.backend._fitted:
            log.info("Fitting TF-IDF + SVD backend on full candidate corpus...")
            corpus = []
            for c in candidates:
                corpus.append(extract_title_text(c))
                corpus.append(extract_summary_text(c))
                corpus.append(extract_career_text(c))
                corpus.append(extract_skills_text(c))
            # Add JD text to corpus so SVD captures JD-relevant dimensions
            jd_path = Path(__file__).parent.parent / "job_description.md"
            if jd_path.exists():
                with open(jd_path) as f:
                    corpus.append(f.read())
            self.backend.fit(corpus)

        results = {}
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(candidates, desc="Encoding candidates")
            except ImportError:
                iterator = candidates
        else:
            iterator = candidates

        for c in iterator:
            results[c.candidate_id] = self.encode_candidate(c)
        return results


# ─── Persistence helpers ──────────────────────────────────────────────────
def save_embeddings(
    embeddings: dict[str, dict[str, np.ndarray]],
    path: Path,
    dtype: str = "float16",
):
    """
    Save embeddings to a .npz file.

    Layout:
        candidate_ids: array of candidate_id strings (length N)
        title:     (N, dim) array
        summary:   (N, dim) array
        career:    (N, dim) array
        skills:    (N, dim) array
        combined:  (N, dim) array

    fp16 cuts disk usage in half vs fp32 with negligible quality loss for
    cosine similarity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_ids = list(embeddings.keys())
    arrays = {"candidate_ids": np.array(candidate_ids, dtype=object)}

    cast = (lambda x: x.astype(np.float16)) if dtype == "float16" else (lambda x: x.astype(np.float32))
    for field in MultiFieldEncoder.FIELDS:
        matrix = np.stack([embeddings[cid][field] for cid in candidate_ids])
        arrays[field] = cast(matrix)

    np.savez(path, **arrays)
    log.info(f"Saved {len(candidate_ids)} candidate embeddings to {path} ({dtype})")


def load_embeddings(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load embeddings saved by save_embeddings()."""
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    candidate_ids = list(data["candidate_ids"])
    embeddings = {}
    for i, cid in enumerate(candidate_ids):
        embeddings[cid] = {
            field: data[field][i].astype(np.float32) for field in MultiFieldEncoder.FIELDS
        }
    return embeddings
