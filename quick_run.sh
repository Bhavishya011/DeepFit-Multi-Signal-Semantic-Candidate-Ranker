#!/bin/bash
# Quick run script — does everything from setup to submission in one command.
#
# Usage:
#   bash quick_run.sh candidates.jsonl
#   bash quick_run.sh candidates.jsonl.gz
#
# What it does:
#   1. Verifies dependencies are installed
#   2. Pre-computes embeddings (if not already done)
#   3. Runs the ranker
#   4. Validates the output
#
# Total time: ~20-25 min for 100K candidates on a 16GB CPU machine.

set -e

cd "$(dirname "$0")"

CANDIDATES="${1:?Usage: bash quick_run.sh candidates.jsonl}"
OUTPUT="submission.csv"

echo "============================================================"
echo "  DeepFit — Quick Run"
echo "============================================================"
echo "  Candidates: $CANDIDATES"
echo "  Output:     $OUTPUT"
echo "============================================================"
echo ""

# ─── Step 1: Check dependencies ──────────────────────────────────────────
echo "[1/4] Checking dependencies..."
python3 -c "
import importlib
deps = ['numpy', 'pandas', 'yaml', 'sklearn', 'faiss', 'rank_bm25', 'sentence_transformers']
missing = []
for dep in deps:
    try:
        importlib.import_module(dep)
    except ImportError:
        missing.append(dep)
if missing:
    print(f'  Missing: {missing}')
    print(f'  Run: pip install -r requirements.txt')
    exit(1)
print('  All dependencies installed.')
"

# ─── Step 2: Pre-compute embeddings (skip if already done) ───────────────
EMB_PATH="artifacts/candidate_embeddings.npz"
INTENT_PATH="artifacts/intent_embeddings.npz"

if [ -f "$EMB_PATH" ] && [ -f "$INTENT_PATH" ]; then
    echo ""
    echo "[2/4] Pre-computed embeddings already exist. Skipping."
    echo "  (Delete $EMB_PATH to force re-compute)"
else
    echo ""
    echo "[2/4] Pre-computing embeddings (one-time, ~15-20 min for 100K)..."
    bash precompute/build_all.sh "$CANDIDATES" "$EMB_PATH"
fi

# ─── Step 3: Run the ranker ──────────────────────────────────────────────
echo ""
echo "[3/4] Running ranker (must complete in ≤5 min)..."
START_TIME=$(date +%s)
python3 rank.py --candidates "$CANDIDATES" --out "$OUTPUT"
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "  Runtime: ${RUNTIME}s"
if [ $RUNTIME -gt 300 ]; then
    echo "  ⚠ WARNING: Runtime exceeds 5-min budget!"
    echo "  Try: python rank.py --candidates $CANDIDATES --out $OUTPUT --rerank-k 300"
    exit 1
fi

# ─── Step 4: Validate ────────────────────────────────────────────────────
echo ""
echo "[4/4] Validating output..."

# Row count
ROW_COUNT=$(($(wc -l < "$OUTPUT") - 1))
if [ $ROW_COUNT -ne 100 ]; then
    echo "  ⚠ WARNING: Expected 100 data rows, got $ROW_COUNT"
else
    echo "  ✓ Row count: 100 (correct)"
fi

# Run CI tests
echo ""
echo "  Running CI tests..."
python3 tests/test_ci.py 2>&1 | tail -5

echo ""
echo "============================================================"
echo "  ✅ Done!"
echo "============================================================"
echo "  Output: $OUTPUT"
echo "  Runtime: ${RUNTIME}s"
echo ""
echo "  Next steps:"
echo "  1. Inspect top-10: head -11 $OUTPUT"
echo "  2. Run official validator: python validate_submission.py $OUTPUT"
echo "  3. Fill in submission_metadata.yaml"
echo "  4. Push to GitHub + deploy HuggingFace Space"
echo "  5. Submit!"
echo "============================================================"
