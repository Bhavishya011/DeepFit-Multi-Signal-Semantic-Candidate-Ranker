#!/usr/bin/env python3
"""
DeepFit — Redrob Intelligent Candidate Discovery Ranker
============================================================================

ENTRY POINT for the Redrob Hackathon submission.

Single command (per spec Section 10.3):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This must produce submission.csv from candidates.jsonl within 5 minutes on
a 16GB CPU-only machine with no network access.

Pre-computed artifacts in artifacts/ are assumed to be present (regenerated
by precompute/02_encode_candidates.py if missing).

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
    python rank.py --candidates ./tests/dev_set/dev_candidates.json --out ./dev_submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make ranker importable
sys.path.insert(0, str(Path(__file__).parent))

from ranker.pipeline import Pipeline, load_candidates_from_file

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="DeepFit — Redrob Intelligent Candidate Discovery Ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full 100K pool
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

    # Gzipped input
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

    # Dev set (50 candidates)
    python rank.py --candidates ./tests/dev_set/dev_candidates.json --out ./dev_submission.csv
        """,
    )
    parser.add_argument(
        "--candidates", type=Path, required=True,
        help="Path to candidates file (.json, .jsonl, or .jsonl.gz)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("./submission.csv"),
        help="Output CSV path (default: ./submission.csv)",
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of candidates to rank (default: 100, per spec)",
    )
    parser.add_argument(
        "--coarse-recall-k", type=int, default=1000,
        help="Top-K from coarse recall (default: 1000)",
    )
    parser.add_argument(
        "--rerank-k", type=int, default=500,
        help="Top-K to rerank with cross-encoder (default: 500)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-stage timing output",
    )
    args = parser.parse_args()

    t_start = time.time()

    # ─── Load candidates ──────────────────────────────────────────────────
    log.info(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates_from_file(args.candidates)
    log.info(f"Loaded {len(candidates)} candidates in {time.time() - t_start:.1f}s")

    if len(candidates) == 0:
        log.error("No candidates loaded. Exiting.")
        sys.exit(1)

    # ─── Build + run pipeline ─────────────────────────────────────────────
    pipeline = Pipeline.build()
    results = pipeline.run(
        candidates=candidates,
        top_n=args.top_n,
        coarse_recall_k=args.coarse_recall_k,
        rerank_k=args.rerank_k,
        verbose=not args.quiet,
    )

    # ─── Write CSV ────────────────────────────────────────────────────────
    pipeline.write_csv(results, args.out)

    total_time = time.time() - t_start
    log.info(f"\n✓ Complete in {total_time:.1f}s")
    log.info(f"  Output: {args.out}")
    log.info(f"  Rows: {len(results)} (header + {len(results)} data rows)")

    # ─── Verify output format ─────────────────────────────────────────────
    if len(results) != args.top_n:
        log.warning(f"Expected {args.top_n} results, got {len(results)}")
    if results:
        scores = [r.score for r in results]
        if not all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
            log.error("Scores are not non-increasing by rank! This will fail validation.")
            sys.exit(1)
        log.info(f"  Top score: {results[0].score:.4f} (rank 1: {results[0].candidate_id})")
        log.info(f"  Bottom score: {results[-1].score:.4f} (rank {len(results)}: {results[-1].candidate_id})")

    # ─── Runtime budget check ─────────────────────────────────────────────
    if total_time > 300:
        log.warning(f"⚠ Runtime {total_time:.1f}s exceeds 5-minute budget!")
    else:
        log.info(f"  Runtime within 5-minute budget ({total_time:.1f}s / 300s)")


if __name__ == "__main__":
    main()
