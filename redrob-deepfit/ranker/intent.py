"""
Module 1: JD-Intent Loader
============================================================================

Loads config/intent_schema.json and computes (or loads pre-computed) intent
axis embeddings. Each axis (explicit_positive_requirements, nice_to_haves,
skill_taxonomy_groups) gets its own embedding vector.

Used by:
    - ranker/rerank.py (multi-axis semantic scoring)
    - ranker/features.py (ESCO coverage against jd_required canonical skills)

The schema is hand-crafted v1 (no LLM dependency). It can be regenerated
by precompute/01_decode_jd_intent.py with an LLM call if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .types import Intent


# ─── Paths ────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
INTENT_SCHEMA_PATH = CONFIG_DIR / "intent_schema.json"
SKILL_ALIASES_PATH = CONFIG_DIR / "skill_aliases.yaml"
# NOTE: extension must be .npz to match np.savez() output
INTENT_EMBEDDINGS_PATH = Path(__file__).parent.parent / "artifacts" / "intent_embeddings.npz"


def load_intent_schema(path: Path = INTENT_SCHEMA_PATH) -> dict:
    """Load the hand-crafted intent_schema.json."""
    with open(path) as f:
        return json.load(f)


def load_skill_aliases(path: Path = SKILL_ALIASES_PATH) -> dict:
    """Load the skill_aliases.yaml file (ESCO + custom AI terminology)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("canonical_skills", {})


def get_jd_required_canonical_skills(schema: dict) -> list[str]:
    """
    Return the list of canonical skills the JD requires (must_have=True).
    Used by Module 6.5 (ESCO coverage scorer).
    """
    required = []
    for group_name, group in schema.get("skill_taxonomy_groups", {}).items():
        # All skills in core_retrieval_ir, core_ranking_eval, core_python_eng are must-have
        # (these map to JD's "things you absolutely need")
        if group_name in ("core_retrieval_ir", "core_ranking_eval", "core_python_eng"):
            required.extend(group.get("canonical_skills", []))
    # Dedupe while preserving order
    seen = set()
    out = []
    for s in required:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def get_all_axis_query_texts(schema: dict) -> dict[str, str]:
    """
    Return a dict {axis_name: embedding_query_text} for every axis in the schema.
    Used by the encoder to produce one embedding per axis.
    """
    axes = {}
    for req in schema.get("explicit_positive_requirements", []):
        if req.get("embedding_query_text"):
            axes[req["axis"]] = req["embedding_query_text"]
    for req in schema.get("nice_to_haves", []):
        if req.get("embedding_query_text"):
            axes[req["axis"]] = req["embedding_query_text"]
    for group_name, group in schema.get("skill_taxonomy_groups", {}).items():
        if group.get("embedding_query_text"):
            axes[group_name] = group["embedding_query_text"]
    # Also add the title archetype for title-fit scoring
    title_match = schema.get("title_archetype_match", {})
    ideal_titles = title_match.get("ideal_titles", [])
    if ideal_titles:
        axes["title_archetype"] = " ".join(ideal_titles[:5])
    return axes


def compute_intent_axis_embeddings(
    schema: dict,
    encoder,
    save_path: Optional[Path] = INTENT_EMBEDDINGS_PATH,
) -> dict[str, np.ndarray]:
    """
    Compute embeddings for every intent axis using the provided encoder.

    Args:
        schema: output of load_intent_schema()
        encoder: object with .encode(text) -> np.ndarray method
                 (ranker.encoder.MultiFieldEncoder or similar)
        save_path: if set, save the embeddings dict as a .npy pickle for reuse

    Returns:
        dict {axis_name: embedding_vector (np.ndarray, L2-normalized)}
    """
    axis_texts = get_all_axis_query_texts(schema)

    # If using TF-IDF backend (not yet fitted), fit it on a corpus made of
    # all axis texts + JD text so the SVD has enough documents.
    backend = getattr(encoder, "backend", None)
    if backend is not None and hasattr(backend, "_fitted") and not backend._fitted:
        corpus = list(axis_texts.values())
        # Add the JD text and skill aliases to enrich the corpus
        jd_path = Path(__file__).parent.parent / "job_description.md"
        if jd_path.exists():
            with open(jd_path) as f:
                corpus.append(f.read())
        aliases_path = Path(__file__).parent.parent / "config" / "skill_aliases.yaml"
        if aliases_path.exists():
            with open(aliases_path) as f:
                corpus.append(f.read())
        # Pad with a few extra copies to exceed min SVD components
        # (TruncatedSVD needs n_components < n_samples)
        while len(corpus) < 50:
            corpus.append(" ".join(axis_texts.values()))
        backend.fit(corpus)

    embeddings = {}
    for axis_name, text in axis_texts.items():
        vec = encoder.encode(text)
        # L2-normalize for cosine similarity later
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings[axis_name] = vec.astype(np.float32)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as a 2D array + axis names list (more portable than pickling a dict)
        axis_names = list(embeddings.keys())
        matrix = np.stack([embeddings[name] for name in axis_names])
        np.savez(save_path, axis_names=np.array(axis_names, dtype=object),
                 embeddings=matrix)
    return embeddings


def load_intent_axis_embeddings(
    path: Path = INTENT_EMBEDDINGS_PATH,
) -> Optional[dict[str, np.ndarray]]:
    """Load pre-computed intent axis embeddings. Returns None if file missing."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    axis_names = list(data["axis_names"])
    matrix = data["embeddings"]
    return {name: matrix[i] for i, name in enumerate(axis_names)}


def load_intent(
    schema_path: Path = INTENT_SCHEMA_PATH,
    embeddings_path: Path = INTENT_EMBEDDINGS_PATH,
    encoder=None,
) -> Intent:
    """
    Load the full Intent object: schema + axis embeddings.

    If embeddings_path exists, load from disk. Otherwise, if encoder is
    provided, compute and save. Otherwise, return Intent with empty embeddings
    (downstream modules must handle this gracefully).
    """
    schema = load_intent_schema(schema_path)

    # Try loading pre-computed embeddings
    axis_embeddings = load_intent_axis_embeddings(embeddings_path)

    # If not found and encoder provided, compute them now
    if axis_embeddings is None and encoder is not None:
        axis_embeddings = compute_intent_axis_embeddings(
            schema, encoder, save_path=embeddings_path
        )
    elif axis_embeddings is None:
        axis_embeddings = {}

    return Intent(schema=schema, axis_embeddings=axis_embeddings)


def get_title_archetype(schema: dict) -> dict:
    """Return the title archetype match config (ideal / acceptable / mismatched / disqualifying)."""
    return schema.get("title_archetype_match", {})


def get_disqualifier_detection_hints(schema: dict) -> list[dict]:
    """Return the list of hard_disqualifiers with their detection_hints."""
    return schema.get("hard_disqualifiers", [])
