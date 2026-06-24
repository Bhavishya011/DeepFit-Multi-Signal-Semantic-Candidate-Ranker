#!/usr/bin/env python3
"""
Final audit — Modules 9 (combiner) + 10 (reasoning) + end-to-end pipeline.

Tests:
  P. combiner.py — equation integrity, weight sum, edge cases
  Q. reasoning.py — template variation, anti-hallucination, rank-aware tone
  R. pipeline.py — end-to-end on dev set
  S. CSV format — validates against official spec
"""

from __future__ import annotations

import csv
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ranker.types import Candidate, ScoreComponents, FilterVerdict, RankedCandidate
from ranker.combiner import FinalCombiner, DEFAULT_BASE_WEIGHTS
from ranker.reasoning import ReasoningGenerator, tone_for_rank, TEMPLATES
from ranker.pipeline import Pipeline, load_candidates_from_file

# ─── Test framework ──────────────────────────────────────────────────────
PASS = 0
FAIL = 0
ERRORS = []

def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        ERRORS.append(msg)
        print(f"  ✗ {msg}")

def section(name):
    print(f"\n{'─'*70}\n  {name}\n{'─'*70}")


# ═══════════════════════════════════════════════════════════════════════
# P. Combiner
# ═══════════════════════════════════════════════════════════════════════
def test_combiner():
    section("P. Combiner — equation integrity")

    combiner = FinalCombiner.from_config()

    # P1. Base weights sum to 1.0
    total = sum(combiner.base_weights.values())
    check(abs(total - 1.0) < 1e-6, f"Base weights sum to 1.0 (got {total:.6f})")

    # P2. All required weights present
    for w in ["semantic", "structural", "production", "esco_coverage", "seniority"]:
        check(w in combiner.base_weights, f"base_weights has '{w}'")

    # P3. Honeypot penalty is binary
    check(combiner.honeypot_penalty_cfg["clean"] == 1.0, "honeypot clean = 1.0")
    check(combiner.honeypot_penalty_cfg["flagged"] == 0.0, "honeypot flagged = 0.0")

    # P4. Trap penalty in [0, 1]
    check(combiner.trap_penalty_cfg["clean"] == 1.0, "trap clean = 1.0")
    check(0.0 <= combiner.trap_penalty_cfg["flagged"] <= 1.0,
          f"trap flagged in [0,1] (got {combiner.trap_penalty_cfg['flagged']})")

    # P5. Availability range valid
    check(combiner.availability_range["min"] < combiner.availability_range["max"],
          f"availability min < max ({combiner.availability_range['min']} < {combiner.availability_range['max']})")
    check(combiner.availability_range["min"] == 0.30, "availability min = 0.30")
    check(combiner.availability_range["max"] == 1.20, "availability max = 1.20")

    # P6. Equation integrity checks
    checks = combiner.verify_equation()
    for check_name, passed in checks.items():
        check(passed, f"Equation check: {check_name}")

    # P7. Worked examples from the plan
    # Ideal candidate: all 1.0, no filters, max availability
    ideal = ScoreComponents(
        semantic_score=0.92, structural_score=0.90, production_evidence=0.85,
        esco_coverage=0.80, seniority_match=0.95,
        honeypot_penalty=1.0, trap_penalty=1.0, availability_mult=1.15,
    )
    ideal_score, ideal_bd = combiner.combine(ideal)
    expected_ideal_base = (
        0.55 * 0.92 + 0.20 * 0.90 + 0.10 * 0.85 + 0.10 * 0.80 + 0.05 * 0.95
    )
    expected_ideal = expected_ideal_base * 1.0 * 1.0 * 1.15
    check(abs(ideal_score - expected_ideal) < 1e-6,
          f"Ideal candidate score matches formula (expected {expected_ideal:.4f}, got {ideal_score:.4f})")

    # Honeypot: any values, honeypot fires → score = 0
    honeypot = ScoreComponents(
        semantic_score=0.95, structural_score=0.85, production_evidence=0.50,
        esco_coverage=0.70, seniority_match=0.80,
        honeypot_penalty=0.0, trap_penalty=1.0, availability_mult=1.10,
    )
    honeypot_score, _ = combiner.combine(honeypot)
    check(honeypot_score == 0.0, f"Honeypot candidate score = 0 (got {honeypot_score})")

    # Trap: any values, trap fires → score = base * 0.3
    trap = ScoreComponents(
        semantic_score=0.78, structural_score=0.50, production_evidence=0.10,
        esco_coverage=0.40, seniority_match=0.30,
        honeypot_penalty=1.0, trap_penalty=0.3, availability_mult=0.80,
    )
    trap_score, _ = combiner.combine(trap)
    expected_trap_base = (
        0.55 * 0.78 + 0.20 * 0.50 + 0.10 * 0.10 + 0.10 * 0.40 + 0.05 * 0.30
    )
    expected_trap = expected_trap_base * 0.3 * 0.80
    check(abs(trap_score - expected_trap) < 1e-6,
          f"Trap candidate score = base * 0.3 * avail (expected {expected_trap:.4f}, got {trap_score:.4f})")

    # Dead candidate: perfect on paper but availability floored
    dead = ScoreComponents(
        semantic_score=1.0, structural_score=1.0, production_evidence=1.0,
        esco_coverage=1.0, seniority_match=1.0,
        honeypot_penalty=1.0, trap_penalty=1.0, availability_mult=0.30,
    )
    dead_score, _ = combiner.combine(dead)
    # Dead: base=1.0 * 1.0 * 1.0 * 0.30 = 0.30
    check(0.29 <= dead_score <= 0.31,
          f"Dead candidate score ≈ 0.30 (got {dead_score:.4f}) — sinks below top-100 cutoff")


# ═══════════════════════════════════════════════════════════════════════
# Q. Reasoning
# ═══════════════════════════════════════════════════════════════════════
def test_reasoning():
    section("Q. Reasoning — variation, anti-hallucination, rank-aware tone")

    generator = ReasoningGenerator.from_config()

    # Q1. Tone varies by rank
    check(tone_for_rank(1)["tone"] == "strong fit", "Rank 1 tone = 'strong fit'")
    check(tone_for_rank(15)["tone"] == "solid match", "Rank 15 tone = 'solid match'")
    check(tone_for_rank(45)["tone"] == "reasonable match", "Rank 45 tone = 'reasonable match'")
    check(tone_for_rank(80)["tone"] == "adjacent candidate", "Rank 80 tone = 'adjacent candidate'")

    # Q2. At least 6 templates for variation
    check(len(TEMPLATES) >= 6, f"At least 6 templates (got {len(TEMPLATES)})")

    # Q3. Generate reasoning for a real candidate
    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        samples = json.load(f)
    c = Candidate.from_dict(samples[0])

    # Build fake components
    components = ScoreComponents(
        semantic_score=0.7, structural_score=0.8, production_evidence=0.6,
        esco_coverage=0.5, seniority_match=0.7,
        honeypot_penalty=1.0, trap_penalty=1.0, availability_mult=1.0,
        axis_scores={"title_fit": 0.6, "career_retrieval": 0.5, "cross_enc": 0.8},
        availability_breakdown={"recency": {"days_inactive": 30}},
        structural_breakdown={"yoe_band": {"label": "in_band"}, "location": {"label": "tier_1"}},
        matched_canonical_skills=["Python", "PyTorch"],
    )
    breakdown = {
        "base_components": {
            "semantic": {"value": 0.7, "weight": 0.55},
            "structural": {"value": 0.8, "weight": 0.20},
            "production": {"value": 0.6, "weight": 0.10},
            "esco_coverage": {"value": 0.5, "weight": 0.10},
            "seniority": {"value": 0.7, "weight": 0.05},
        }
    }

    reasoning = generator.generate(c, rank=5, components=components, breakdown=breakdown)
    check(reasoning and len(reasoning) > 20, f"Reasoning generated (len={len(reasoning)})")
    check(len(reasoning) <= 250, f"Reasoning ≤ 250 chars (got {len(reasoning)})")
    check(reasoning.endswith("."), "Reasoning ends with period")

    # Q4. Reasoning should mention specific facts from candidate OR be semantically relevant
    # CAND_0000001 is "Backend Engineer" with 6.9 YOE — depending on template selected,
    # reasoning may lead with title, YOE, or semantic match. All are valid.
    facts_present = any(fact in reasoning.lower() for fact in [
        "6.9", "backend", "engineer", "semantic match", "strong fit", "experience"
    ])
    check(facts_present, f"Reasoning cites candidate facts or fit: '{reasoning[:100]}...'")

    # Q5. Anti-hallucination check
    passes, violations = generator.verify_no_hallucination(c, reasoning)
    check(passes, f"Anti-hallucination check passes (violations: {violations})")

    # Q6. Generate reasoning for 10 candidates — check variation
    reasonings = []
    for i in range(10):
        c = Candidate.from_dict(samples[i])
        # Vary the rank to get different tones
        rank = (i + 1) * 10
        r = generator.generate(c, rank=rank, components=components, breakdown=breakdown)
        reasonings.append(r)
    unique = len(set(reasonings))
    check(unique >= 7, f"At least 7 unique reasonings out of 10 (got {unique})")

    # Q7. Rank-aware tone appears in reasoning
    top_reasoning = generator.generate(c, rank=1, components=components, breakdown=breakdown)
    mid_reasoning = generator.generate(c, rank=50, components=components, breakdown=breakdown)
    bottom_reasoning = generator.generate(c, rank=95, components=components, breakdown=breakdown)
    # Top should have "strong fit" or "Excellent match"
    check("strong fit" in top_reasoning.lower() or "excellent" in top_reasoning.lower(),
          f"Top-rank reasoning has strong tone: '{top_reasoning[:80]}...'")
    # Bottom should have weak tone — could be "adjacent", "borderline", "concern",
    # OR start with a concern-led clause (e.g., "120d notice period ...")
    bottom_lower = bottom_reasoning.lower()
    has_weak_tone = any(kw in bottom_lower for kw in [
        "adjacent", "borderline", "concern", "note:", "inactive",
        "notice period", "below", "limited", "low response"
    ])
    check(has_weak_tone, f"Bottom-rank reasoning has weak tone: '{bottom_reasoning[:80]}...'")

    # Q8. Concern acknowledgment when present
    # Build components with a clear concern (low production evidence)
    concerned_components = ScoreComponents(
        semantic_score=0.7, structural_score=0.8, production_evidence=0.1,  # LOW
        esco_coverage=0.1, seniority_match=0.7,  # LOW esco
        honeypot_penalty=1.0, trap_penalty=1.0, availability_mult=0.4,  # LOW availability
        availability_breakdown={"recency": {"days_inactive": 200}},  # INACTIVE
        structural_breakdown={"yoe_band": {"label": "below_band_3.0yr"}},
    )
    concerned_reasoning = generator.generate(c, rank=20, components=concerned_components, breakdown=breakdown)
    # Should mention at least one concern
    has_concern = any(kw in concerned_reasoning.lower() for kw in [
        "inactive", "concern", "note:", "below", "low", "limited", "200d"
    ])
    check(has_concern, f"Reasoning acknowledges concern: '{concerned_reasoning[:100]}...'")


# ═══════════════════════════════════════════════════════════════════════
# R. Pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════
def test_pipeline():
    section("R. Pipeline end-to-end on dev set")

    # R1. Load candidates
    candidates = load_candidates_from_file(
        Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json"
    )
    check(len(candidates) == 50, f"Loaded 50 candidates (got {len(candidates)})")

    # R2. Build pipeline
    pipeline = Pipeline.build()
    check(pipeline is not None, "Pipeline built successfully")
    check(pipeline.intent is not None, "Pipeline has intent")
    check(pipeline.combiner is not None, "Pipeline has combiner")
    check(pipeline.reasoning_generator is not None, "Pipeline has reasoning generator")

    # R3. Run pipeline
    import time
    t0 = time.time()
    results = pipeline.run(candidates, top_n=100, verbose=False)
    runtime = time.time() - t0
    check(runtime < 60, f"Pipeline runs in <60s on dev set (got {runtime:.1f}s)")
    check(len(results) > 0, f"Pipeline returns results (got {len(results)})")
    check(len(results) <= 100, f"Pipeline returns ≤100 results (got {len(results)})")

    # R4. Results are sorted by score descending
    scores = [r.score for r in results]
    check(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
          "Results sorted by score descending")

    # R5. Ranks are 1..N with no gaps
    ranks = [r.rank for r in results]
    check(ranks == list(range(1, len(results) + 1)),
          f"Ranks are 1..{len(results)} with no gaps")

    # R6. All candidate_ids are valid format
    cid_pattern = re.compile(r"^CAND_[0-9]{7}$")
    check(all(cid_pattern.match(r.candidate_id) for r in results),
          "All candidate_ids match CAND_XXXXXXX pattern")

    # R7. All reasoning non-empty and unique
    reasonings = [r.reasoning for r in results]
    check(all(r.strip() for r in reasonings), "All reasoning non-empty")
    check(len(set(reasonings)) == len(reasonings), "All reasoning unique (no template monotony)")

    # R8. Reasoning length reasonable (1-2 sentences, ≤250 chars)
    lengths = [len(r) for r in reasonings]
    check(all(20 <= l <= 250 for l in lengths),
          f"All reasoning length in [20, 250] (min={min(lengths)}, max={max(lengths)})")

    # R9. Write CSV and validate format
    output_path = Path(__file__).parent.parent / "dev_submission.csv"
    pipeline.write_csv(results, output_path)
    check(output_path.exists(), f"CSV written to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# S. CSV format validation (against official spec)
# ═══════════════════════════════════════════════════════════════════════
def test_csv_format():
    section("S. CSV format validation (official spec)")

    csv_path = Path(__file__).parent.parent / "dev_submission.csv"
    if not csv_path.exists():
        check(False, "dev_submission.csv not found — run pipeline test first")
        return

    REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
    CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # S1. Header matches spec exactly
    check(header == REQUIRED_HEADER, f"Header matches spec (got {header})")

    # S2. candidate_id format valid
    bad_ids = [r[0] for r in rows if not CANDIDATE_ID_PATTERN.match(r[0])]
    check(len(bad_ids) == 0, f"All candidate_ids valid format (bad: {bad_ids})")

    # S3. Ranks start at 1, no duplicates
    ranks = [int(r[1]) for r in rows]
    check(min(ranks) == 1, f"Ranks start at 1 (min={min(ranks)})")
    check(len(set(ranks)) == len(ranks), "No duplicate ranks")

    # S4. Scores are floats, non-increasing
    scores = [float(r[2]) for r in rows]
    non_increasing = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    check(non_increasing, "Scores non-increasing by rank")

    # S5. Reasoning column non-empty
    reasoning = [r[3] for r in rows]
    check(all(r.strip() for r in reasoning), "All reasoning non-empty")

    # S6. Reasoning unique (no template monotony — Stage 4 criteria)
    check(len(set(reasoning)) == len(reasoning), "All reasoning unique (no monotony)")

    # S7. Reasoning length reasonable
    lengths = [len(r) for r in reasoning]
    check(all(l <= 250 for l in lengths), f"All reasoning ≤250 chars (max={max(lengths)})")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("  DeepFit Final Audit — Modules 9 (combiner) + 10 (reasoning) + Pipeline")
    print("="*70)

    tests = [
        ("combiner", test_combiner),
        ("reasoning", test_reasoning),
        ("pipeline", test_pipeline),
        ("csv_format", test_csv_format),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            global FAIL
            FAIL += 1
            ERRORS.append(f"{name}: unhandled exception - {e}")
            print(f"\n  ✗✗✗ UNHANDLED EXCEPTION in {name}: {e}")
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"  FINAL AUDIT SUMMARY")
    print("="*70)
    print(f"  Passed: {PASS}")
    print(f"  Failed: {FAIL}")
    if ERRORS:
        print(f"\n  Failures:")
        for e in ERRORS[:20]:
            print(f"    - {e}")
    print()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
