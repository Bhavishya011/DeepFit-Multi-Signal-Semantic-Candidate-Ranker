#!/usr/bin/env python3
"""
CI tests — run before every submission to catch format/runtime/reasoning issues.

These tests are designed to be fast (<60s total) and catch the most common
failure modes that would get a submission auto-rejected at Stage 1 or fail
reproduction at Stage 3.

Usage:
    pytest tests/test_ci.py -v
    # or
    python tests/test_ci.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ─── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DEV_CANDIDATES = ROOT / "tests" / "dev_set" / "dev_candidates.json"
SUBMISSION_CSV = ROOT / "submission.csv"
RANK_PY = ROOT / "rank.py"

# ─── Spec constants ───────────────────────────────────────────────────────
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
MIN_DATA_ROWS = 1  # dev set only has 31 survivors; relax for dev
MAX_DATA_ROWS = 100
MAX_RUNTIME_SECONDS = 300  # 5 min hard limit
MAX_REASONING_CHARS = 250


# ═══════════════════════════════════════════════════════════════════════
# Test 1: rank.py runs end-to-end and produces valid CSV
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineEndToEnd:
    """Tests that rank.py runs end-to-end and produces a valid CSV."""

    @pytest.fixture(scope="class")
    def submission_csv(self):
        """Run rank.py once and cache the output for all tests in this class."""
        if SUBMISSION_CSV.exists():
            # Already exists from a previous run — use it
            return SUBMISSION_CSV

        # Run rank.py on dev set
        result = subprocess.run(
            [sys.executable, str(RANK_PY),
             "--candidates", str(DEV_CANDIDATES),
             "--out", str(SUBMISSION_CSV)],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        assert result.returncode == 0, f"rank.py failed:\n{result.stderr}"
        assert SUBMISSION_CSV.exists(), "submission.csv not created"
        return SUBMISSION_CSV

    def test_csv_exists(self, submission_csv):
        """CSV file exists after rank.py runs."""
        assert submission_csv.exists()

    def test_header_matches_spec(self, submission_csv):
        """Header is exactly: candidate_id,rank,score,reasoning"""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == REQUIRED_HEADER, f"Header mismatch: {header}"

    def test_row_count_in_range(self, submission_csv):
        """Row count is between 1 and 100 (dev set may have fewer than 100)."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        assert MIN_DATA_ROWS <= len(rows) <= MAX_DATA_ROWS, f"Row count {len(rows)} not in [{MIN_DATA_ROWS}, {MAX_DATA_ROWS}]"

    def test_candidate_ids_valid_format(self, submission_csv):
        """All candidate_ids match CAND_XXXXXXX pattern."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue
                cid = row[0].strip()
                assert CANDIDATE_ID_PATTERN.match(cid), f"Row {i}: invalid candidate_id '{cid}'"

    def test_no_duplicate_candidate_ids(self, submission_csv):
        """No duplicate candidate_ids."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            ids = [row[0] for row in reader if any(cell.strip() for cell in row)]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {len(ids)} total, {len(set(ids))} unique"

    def test_ranks_start_at_1(self, submission_csv):
        """Ranks start at 1."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            ranks = [int(row[1]) for row in reader if any(cell.strip() for cell in row)]
        assert min(ranks) == 1, f"Min rank is {min(ranks)}, expected 1"

    def test_no_duplicate_ranks(self, submission_csv):
        """No duplicate ranks."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            ranks = [int(row[1]) for row in reader if any(cell.strip() for cell in row)]
        assert len(ranks) == len(set(ranks)), f"Duplicate ranks: {ranks}"

    def test_ranks_contiguous(self, submission_csv):
        """Ranks are 1, 2, 3, ... with no gaps."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            ranks = sorted([int(row[1]) for row in reader if any(cell.strip() for cell in row)])
        expected = list(range(1, len(ranks) + 1))
        assert ranks == expected, f"Ranks not contiguous: {ranks}"

    def test_scores_are_floats(self, submission_csv):
        """All scores are valid floats."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue
                try:
                    float(row[2])
                except ValueError:
                    pytest.fail(f"Row {i}: score '{row[2]}' is not a float")

    def test_scores_non_increasing(self, submission_csv):
        """Scores are non-increasing by rank."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        scores = [float(row[2]) for row in rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score increased at rank {i+1}: {scores[i]} < {scores[i+1]}"

    def test_reasoning_non_empty(self, submission_csv):
        """All reasoning fields are non-empty."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue
                assert row[3].strip(), f"Row {i}: empty reasoning"

    def test_reasoning_unique(self, submission_csv):
        """All reasoning strings are unique (no template monotony)."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            reasoning = [row[3] for row in reader if any(cell.strip() for cell in row)]
        assert len(reasoning) == len(set(reasoning)), f"Duplicate reasoning found ({len(reasoning)} total, {len(set(reasoning))} unique)"

    def test_reasoning_length_within_limit(self, submission_csv):
        """All reasoning ≤250 chars."""
        with open(submission_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue
                assert len(row[3]) <= MAX_REASONING_CHARS, f"Row {i}: reasoning {len(row[3])} chars > {MAX_REASONING_CHARS}"


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Runtime budget
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeBudget:
    """Tests that the pipeline runs within the 5-minute budget."""

    def test_dev_set_runtime_under_60s(self):
        """Dev set (50 candidates) should rank in <60s."""
        # The dev set is small; this is a sanity check
        # The real 100K test happens in Stage 3 sandbox
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(RANK_PY),
             "--candidates", str(DEV_CANDIDATES),
             "--out", str(ROOT / "_test_runtime.csv"),
             "--quiet"],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        runtime = time.time() - t0

        # Cleanup
        (ROOT / "_test_runtime.csv").unlink(missing_ok=True)

        assert result.returncode == 0, f"rank.py failed:\n{result.stderr}"
        assert runtime < 60, f"Dev set runtime {runtime:.1f}s > 60s budget"


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Honeypot detection (no honeypots in top-10)
# ═══════════════════════════════════════════════════════════════════════

class TestHoneypotDetection:
    """Tests that honeypots are properly filtered (spec Section 7)."""

    def test_no_honeypots_in_top_10(self):
        """Top-10 must not contain honeypots (would disqualify at Stage 3)."""
        # Load submission
        if not SUBMISSION_CSV.exists():
            pytest.skip("submission.csv not found — run pipeline first")

        with open(SUBMISSION_CSV) as f:
            reader = csv.reader(f)
            next(reader)
            top_10_ids = [row[0] for i, row in enumerate(reader) if i < 10]

        # Load dev candidates to check for known honeypots
        with open(DEV_CANDIDATES) as f:
            candidates = json.load(f)

        # Run honeypot scorer on all candidates
        sys.path.insert(0, str(ROOT))
        from ranker.types import Candidate
        from ranker.filters import HoneypotScorer

        scorer = HoneypotScorer.from_config()
        honeypot_ids = set()
        for record in candidates:
            c = Candidate.from_dict(record)
            verdict = scorer.score(c)
            if verdict.is_honeypot:
                honeypot_ids.add(c.candidate_id)

        # Check none of the top-10 are honeypots
        honeypots_in_top_10 = [cid for cid in top_10_ids if cid in honeypot_ids]
        assert len(honeypots_in_top_10) == 0, \
            f"Honeypots in top-10: {honeypots_in_top_10} (would fail Stage 3)"


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Reasoning anti-hallucination
# ═══════════════════════════════════════════════════════════════════════

class TestReasoningAntiHallucination:
    """Tests that reasoning doesn't hallucinate (spec Section 3 Table 3)."""

    def test_reasoning_claims_traceable_to_profile(self):
        """Every claim in reasoning must be traceable to the candidate's profile."""
        if not SUBMISSION_CSV.exists():
            pytest.skip("submission.csv not found")

        # Load submission
        with open(SUBMISSION_CSV) as f:
            reader = csv.DictReader(f)
            submission_rows = list(reader)

        # Load dev candidates
        with open(DEV_CANDIDATES) as f:
            candidates_data = {c["candidate_id"]: c for c in json.load(f)}

        sys.path.insert(0, str(ROOT))
        from ranker.types import Candidate
        from ranker.reasoning import ReasoningGenerator

        generator = ReasoningGenerator.from_config()
        violations = []

        for row in submission_rows:
            cid = row["candidate_id"]
            reasoning = row["reasoning"]

            if cid not in candidates_data:
                continue  # skip if candidate not in dev set

            candidate = Candidate.from_dict(candidates_data[cid])
            passes, claim_violations = generator.verify_no_hallucination(candidate, reasoning)
            if not passes:
                violations.append({
                    "candidate_id": cid,
                    "reasoning": reasoning,
                    "violations": claim_violations,
                })

        assert len(violations) == 0, \
            f"Anti-hallucination violations in {len(violations)} candidates:\n" + \
            "\n".join(f"  {v['candidate_id']}: {v['violations']}" for v in violations[:5])


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Combiner equation integrity
# ═══════════════════════════════════════════════════════════════════════

class TestCombinerEquation:
    """Tests that the final combiner equation is correct."""

    def test_base_weights_sum_to_1(self):
        """Base weights must sum to 1.0."""
        sys.path.insert(0, str(ROOT))
        from ranker.combiner import FinalCombiner
        combiner = FinalCombiner.from_config()
        total = sum(combiner.base_weights.values())
        assert abs(total - 1.0) < 1e-6, f"Base weights sum to {total}, expected 1.0"

    def test_equation_integrity_checks_pass(self):
        """All verify_equation() checks pass."""
        sys.path.insert(0, str(ROOT))
        from ranker.combiner import FinalCombiner
        combiner = FinalCombiner.from_config()
        checks = combiner.verify_equation()
        failed = [name for name, passed in checks.items() if not passed]
        assert not failed, f"Equation checks failed: {failed}"


# ═══════════════════════════════════════════════════════════════════════
# Main (for running without pytest)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run tests without pytest
    print("\n" + "="*70)
    print("  DeepFit CI Tests")
    print("="*70 + "\n")

    # First, ensure submission.csv exists
    if not SUBMISSION_CSV.exists():
        print("  Running rank.py to generate submission.csv...")
        result = subprocess.run(
            [sys.executable, str(RANK_PY),
             "--candidates", str(DEV_CANDIDATES),
             "--out", str(SUBMISSION_CSV)],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"  ✗ rank.py failed:\n{result.stderr}")
            sys.exit(1)

    # Run all test classes
    test_classes = [
        TestPipelineEndToEnd(),
        TestRuntimeBudget(),
        TestHoneypotDetection(),
        TestReasoningAntiHallucination(),
        TestCombinerEquation(),
    ]

    total_pass = 0
    total_fail = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n  {class_name}")
        print(f"  {'─'*60}")

        methods = [m for m in dir(test_class) if m.startswith("test_")]
        for method_name in methods:
            try:
                # Skip fixtures
                if method_name == "submission_csv":
                    continue
                method = getattr(test_class, method_name)
                # Handle fixtures by providing the file path
                if "submission_csv" in method.__code__.co_varnames:
                    method(submission_csv=SUBMISSION_CSV)
                else:
                    method()
                print(f"    ✓ {method_name}")
                total_pass += 1
            except Exception as e:
                print(f"    ✗ {method_name}: {e}")
                total_fail += 1

    print(f"\n{'='*70}")
    print(f"  CI Tests Summary")
    print(f"{'='*70}")
    print(f"  Passed: {total_pass}")
    print(f"  Failed: {total_fail}")
    print()
    sys.exit(0 if total_fail == 0 else 1)
