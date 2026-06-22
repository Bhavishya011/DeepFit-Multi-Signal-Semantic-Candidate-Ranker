#!/usr/bin/env python3
"""
Test: run HoneypotScorer on dev set and compare against heuristic labels.

This is the validation gate for Module 3. If the scorer's verdicts match
the heuristic labels closely, we have confidence the rules are correct.
Mismatches reveal either:
  - False positives (scorer kills a candidate the heuristic kept)
  - False negatives (scorer misses a candidate the heuristic killed)
  - Heuristic errors (we trusted the heuristic but it was wrong)

Usage:
    python scripts/test_filters_on_dev.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ranker.filters import HoneypotScorer
from ranker.types import Candidate

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


def load_dev_candidates() -> list[Candidate]:
    path = Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json"
    with open(path) as f:
        records = json.load(f)
    return [Candidate.from_dict(r) for r in records]


def load_dev_labels() -> dict[str, int]:
    path = Path(__file__).parent.parent / "tests" / "dev_set" / "dev_labels.json"
    with open(path) as f:
        data = json.load(f)
    return {cid: v.get("tier") for cid, v in data.get("labels", {}).items() if v.get("tier") is not None}


def main():
    candidates = load_dev_candidates()
    labels = load_dev_labels()
    scorer = HoneypotScorer.from_config()

    print(f"\n{'='*100}")
    print(f"  HoneypotScorer validation on {len(candidates)} dev candidates")
    print(f"  (Comparing scorer verdicts against heuristic labels)")
    print(f"{'='*100}")

    results = []
    for c in candidates:
        verdict = scorer.score(c)
        label_tier = labels.get(c.candidate_id, None)
        # Tier 0 = irrelevant (honeypot/trap/disqualifier) → should be filtered
        # Tier 1 = marginal (adjacent skills, missing key req) → should NOT be filtered
        # Tier 2/3 = relevant → should NOT be filtered
        # Scorer filtering happens when: honeypot fires (hard kill) OR trap fires (soft penalty)
        # NOTE: traps are not "filtered" in the hard sense — they're penalized but may still
        # appear in top-100 if their semantic score is high enough.
        # For this validation, we compare against tier==0 (hard filter expected).
        expected_hard_filtered = label_tier == 0
        actual_hard_filtered = verdict.is_honeypot  # only hard honeypots = filtered
        results.append({
            "candidate_id": c.candidate_id,
            "title": c.title,
            "label_tier": label_tier,
            "is_honeypot": verdict.is_honeypot,
            "is_trap": verdict.is_trap,
            "actual_hard_filtered": actual_hard_filtered,
            "expected_hard_filtered": expected_hard_filtered,
            "fired_honeypot": verdict.fired_honeypot_rules,
            "fired_trap": verdict.fired_trap_rules,
            "matches": actual_hard_filtered == expected_hard_filtered,
        })

    # ─── Print per-candidate results ─────────────────────────────────────
    print(f"\n{'ID':<15} {'Title':<32} {'Label':>5} {'HP':>3} {'TRAP':>5} {'HP?':>4} {'Exp?':>4} {'✓':>2}  Rules fired")
    print("─" * 120)
    for r in results:
        match_str = "✓" if r["matches"] else "✗"
        rules = r["fired_honeypot"] + [f"trap:{t}" for t in r["fired_trap"]]
        rules_str = ", ".join(rules)[:50] if rules else "(none)"
        print(f"{r['candidate_id']:<15} {r['title'][:31]:<32} "
              f"{str(r['label_tier']):>5} "
              f"{'HP' if r['is_honeypot'] else '.':>3} "
              f"{'TRAP' if r['is_trap'] else '.':>5} "
              f"{'YES' if r['actual_hard_filtered'] else 'no':>4} "
              f"{'YES' if r['expected_hard_filtered'] else 'no':>4} "
              f"{match_str:>2}  {rules_str}")

    # ─── Confusion matrix ────────────────────────────────────────────────
    tp = sum(1 for r in results if r["actual_hard_filtered"] and r["expected_hard_filtered"])
    fp = sum(1 for r in results if r["actual_hard_filtered"] and not r["expected_hard_filtered"])
    fn = sum(1 for r in results if not r["actual_hard_filtered"] and r["expected_hard_filtered"])
    tn = sum(1 for r in results if not r["actual_hard_filtered"] and not r["expected_hard_filtered"])

    print(f"\n{'='*80}")
    print(f"  Confusion Matrix (hard honeypot filter = predicted positive)")
    print(f"  (Traps are NOT counted as filtered — they're soft penalties)")
    print(f"{'='*80}")
    print(f"                   Expected filtered    Expected kept")
    print(f"  Predicted filtered  TP = {tp:<3}            FP = {fp:<3}")
    print(f"  Predicted kept       FN = {fn:<3}            TN = {tn:<3}")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / len(results)

    print(f"\n  Precision: {precision:.3f}  (of those we filtered, how many were correctly filtered)")
    print(f"  Recall:    {recall:.3f}  (of those that should be filtered, how many we caught)")
    print(f"  F1:        {f1:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")

    # ─── Rule fire counts ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Rule fire counts")
    print(f"{'='*80}")
    all_rules = Counter()
    for r in results:
        for rule in r["fired_honeypot"]:
            all_rules[f"HARD: {rule}"] += 1
        for rule in r["fired_trap"]:
            all_rules[f"SOFT: {rule}"] += 1
    for rule, count in all_rules.most_common():
        print(f"  {rule:<55} {count:>3} fires")

    # ─── Mismatches detail ──────────────────────────────────────────────
    mismatches = [r for r in results if not r["matches"]]
    if mismatches:
        print(f"\n{'='*80}")
        print(f"  Mismatches ({len(mismatches)}) — analysis")
        print(f"{'='*80}")
        print(f"\n  Two categories of mismatches:")
        print(f"  (A) Trap-but-not-honeypot: heuristic labeled tier 0 (filtered) but scorer")
        print(f"      correctly classified as a TRAP (soft penalty), not a hard honeypot.")
        print(f"      These are NOT bugs — the scorer is being more precise than the heuristic.")
        print(f"      The candidate will still sink below top-100 via the 0.3 trap_penalty.")
        print(f"  (B) Scorer-caught-honeypot: scorer found a real impossibility the heuristic missed.")
        print(f"      These ARE wins for the scorer — manual review needed to confirm.")

        cat_a = [r for r in mismatches if r["expected_hard_filtered"] and r["is_trap"] and not r["is_honeypot"]]
        cat_b = [r for r in mismatches if not r["expected_hard_filtered"] and r["is_honeypot"]]
        cat_other = [r for r in mismatches if r not in cat_a and r not in cat_b]

        if cat_a:
            print(f"\n  ── Category A: Trap-not-honeypot ({len(cat_a)} cases) ──")
            for r in cat_a:
                print(f"    {r['candidate_id']} ({r['title']}) — trap: {r['fired_trap']}")

        if cat_b:
            print(f"\n  ── Category B: Scorer-caught-honeypot ({len(cat_b)} cases) ──")
            for r in cat_b:
                print(f"    {r['candidate_id']} ({r['title']}) — heuristic tier {r['label_tier']}, "
                      f"scorer hard-killed: {r['fired_honeypot']}")

        if cat_other:
            print(f"\n  ── Other mismatches ({len(cat_other)} cases) ──")
            for r in cat_other:
                print(f"    {r['candidate_id']} ({r['title']})")
                print(f"      Label tier: {r['label_tier']} (expected_hard_filtered={r['expected_hard_filtered']})")
                print(f"      Scorer: is_honeypot={r['is_honeypot']}, is_trap={r['is_trap']}")
                print(f"      Fired honeypot rules: {r['fired_honeypot']}")
                print(f"      Fired trap rules:     {r['fired_trap']}")

        print(f"\n  Summary: {len(cat_a)} category-A (scorer more precise), "
              f"{len(cat_b)} category-B (scorer caught real honeypot), "
              f"{len(cat_other)} other (needs review)")
    else:
        print(f"\n  ✓ No mismatches — scorer perfectly matches heuristic labels.")

    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
