#!/usr/bin/env python3
"""
Pre-compute script: encode all candidates into multi-field embeddings.

Usage:
    python precompute/02_encode_candidates.py --candidates tests/dev_set/dev_candidates.json
    python precompute/02_encode_candidates.py --candidates candidates.jsonl --output artifacts/candidate_embeddings.npz

For .jsonl input (full 100K pool):
    python precompute/02_encode_candidates.py --candidates candidates.jsonl.gz

For dev set testing:
    python precompute/02_encode_candidates.py --candidates tests/dev_set/dev_candidates.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

# Make ranker importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranker.encoder import MultiFieldEncoder, save_embeddings
from ranker.types import Candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_candidates(path: Path) -> list[Candidate]:
    """Load candidates from .json (list) or .jsonl(.gz) (one per line)."""
    if path.suffix == ".json":
        with open(path) as f:
            records = json.load(f)
        log.info(f"Loaded {len(records)} candidates from JSON ({path})")
    elif path.suffix in (".jsonl", ".gz"):
        opener = gzip.open if path.suffix == ".gz" else open
        records = []
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        log.info(f"Loaded {len(records)} candidates from JSONL ({path})")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return [Candidate.from_dict(r) for r in records]


def main():
    parser = argparse.ArgumentParser(description="Pre-compute multi-field candidate embeddings")
    parser.add_argument(
        "--candidates", type=Path, required=True,
        help="Path to candidates file (.json, .jsonl, or .jsonl.gz)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent.parent / "artifacts" / "candidate_embeddings.npz",
        help="Output .npz file path"
    )
    parser.add_argument(
        "--backend", choices=["auto", "sentence-transformers", "tfidf-svd"],
        default="auto",
        help="Encoder backend to use (default: auto = try sentence-transformers, fall back to tfidf-svd)"
    )
    parser.add_argument(
        "--model", type=str, default="BAAI/bge-small-en-v1.5",
        help="HuggingFace model name for sentence-transformers backend"
    )
    parser.add_argument(
        "--dtype", choices=["float16", "float32"], default="float16",
        help="Storage dtype (float16 halves disk usage, negligible quality loss)"
    )
    args = parser.parse_args()

    t0 = time.time()
    candidates = load_candidates(args.candidates)
    t_load = time.time() - t0
    log.info(f"Loaded {len(candidates)} candidates in {t_load:.1f}s")

    t1 = time.time()
    encoder = MultiFieldEncoder.from_auto(preferred=args.backend, model_name=args.model)
    t_init = time.time() - t1
    log.info(f"Encoder initialized in {t_init:.1f}s (backend: {encoder.backend.name})")

    t2 = time.time()
    embeddings = encoder.encode_candidates(candidates, show_progress=True)
    t_encode = time.time() - t2
    log.info(f"Encoded {len(embeddings)} candidates in {t_encode:.1f}s "
             f"({len(embeddings) / max(t_encode, 0.01):.1f} candidates/sec)")

    t3 = time.time()
    save_embeddings(embeddings, args.output, dtype=args.dtype)
    t_save = time.time() - t3
    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    log.info(f"Saved to {args.output} in {t_save:.1f}s ({file_size_mb:.1f} MB)")

    # ─── Also compute intent axis embeddings using the SAME fitted encoder ───
    # This ensures dimension match between candidate embeddings and intent axes.
    # CRITICAL: if dimensions don't match, cosine similarity in rerank.py fails.
    t4 = time.time()
    from ranker.intent import load_intent_schema, compute_intent_axis_embeddings, INTENT_EMBEDDINGS_PATH
    schema = load_intent_schema()
    intent_embeddings = compute_intent_axis_embeddings(
        schema, encoder, save_path=INTENT_EMBEDDINGS_PATH
    )
    t_intent = time.time() - t4
    log.info(f"Computed {len(intent_embeddings)} intent axis embeddings in {t_intent:.1f}s "
             f"(dim: {encoder.backend.dim})")

    total = time.time() - t0
    log.info(f"\nTotal: {total:.1f}s for {len(candidates)} candidates")
    log.info(f"  Load:     {t_load:.1f}s")
    log.info(f"  Init:     {t_init:.1f}s")
    log.info(f"  Encode:   {t_encode:.1f}s")
    log.info(f"  Save:     {t_save:.1f}s")
    log.info(f"  Intent:   {t_intent:.1f}s")

    # Verify dimension match
    sample_candidate_emb = next(iter(embeddings.values()))["combined"]
    sample_intent_emb = next(iter(intent_embeddings.values()))
    assert sample_candidate_emb.shape == sample_intent_emb.shape, (
        f"CRITICAL: dimension mismatch! "
        f"candidate={sample_candidate_emb.shape}, intent={sample_intent_emb.shape}"
    )
    log.info(f"✓ Dimension match verified: {sample_candidate_emb.shape}")

    # Estimate full 100K runtime
    if len(candidates) < 100000:
        est_100k = t_encode * (100000 / len(candidates))
        log.info(f"\nEstimated encode time for 100K candidates: {est_100k:.0f}s "
                 f"({est_100k / 60:.1f} min)")


if __name__ == "__main__":
    main()
