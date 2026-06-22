#!/bin/bash
# Build all pre-computed artifacts.
#
# Usage:
#   bash precompute/build_all.sh                              # dev set
#   bash precompute/build_all.sh --candidates candidates.jsonl  # full pool

set -e

cd "$(dirname "$0")/.."

CANDIDATES="${1:-tests/dev_set/dev_candidates.json}"
OUTPUT="${2:-artifacts/candidate_embeddings.npz}"

echo "============================================================"
echo "  DeepFit Pre-compute Build"
echo "============================================================"
echo "  Candidates: $CANDIDATES"
echo "  Output:     $OUTPUT"
echo "============================================================"
echo ""

# Step 1: Encode candidates + compute intent axis embeddings
echo "[1/2] Encoding candidates + intent axes..."
python3 precompute/02_encode_candidates.py \
    --candidates "$CANDIDATES" \
    --output "$OUTPUT" \
    --backend auto

echo ""
echo "[2/2] Build complete!"
echo ""
echo "Artifacts produced:"
ls -lh artifacts/*.npz 2>/dev/null || echo "  (none)"
echo ""
echo "Next: run the ranker"
echo "  python rank.py --candidates $CANDIDATES --out submission.csv"
