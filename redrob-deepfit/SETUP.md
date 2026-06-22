# DeepFit — Setup & Submission Guide

> Read this before running anything.

## 1. Prerequisites

- Python 3.10+ (tested on 3.11, 3.12)
- 16GB+ RAM
- ~2GB free disk space (for model weights + embeddings)
- The `candidates.jsonl` or `candidates.jsonl.gz` file from the hackathon bundle

## 2. Quick start (3 commands)

```bash
# (1) Install dependencies
pip install -r requirements.txt

# (2) Pre-compute embeddings (one-time, ~15-20 min for 100K candidates)
bash precompute/build_all.sh candidates.jsonl artifacts/candidate_embeddings.npz

# (3) Run the ranker (must complete in ≤5 min)
python rank.py --candidates candidates.jsonl --out submission.csv
```

## 3. Validate before submitting

```bash
# Our CI tests
python tests/test_ci.py

# Official validator (from hackathon bundle)
python validate_submission.py submission.csv
```

Both should report 0 failures.

## 4. If the file is gzipped

If your candidates file is `candidates.jsonl.gz`:

```bash
# Pre-compute (use .gz path)
bash precompute/build_all.sh candidates.jsonl.gz artifacts/candidate_embeddings.npz

# Rank (use .gz path)
python rank.py --candidates candidates.jsonl.gz --out submission.csv
```

The pipeline auto-detects `.gz` and decompresses on the fly.

## 5. Runtime budget checklist

Before submitting, verify on your machine:

| Stage | Expected time | Hard limit |
|---|---|---|
| Pre-compute (one-time, offline) | 15-20 min | None (allowed to exceed) |
| **Ranking step** (`rank.py`) | 60-90s | **5 min (HARD)** |
| Validation | 5s | None |

If `rank.py` takes >4 min on your machine, reduce `--rerank-k`:
```bash
python rank.py --candidates candidates.jsonl --out submission.csv --rerank-k 300
```

## 6. Project structure

```
DeepFit-Multi-Signal-Semantic-Candidate-Ranker/
├── rank.py                    ← ENTRY POINT
├── app.py                     ← HuggingFace Spaces sandbox
├── requirements.txt           ← Python deps
├── sandbox_requirements.txt   ← HF Spaces deps (rename to requirements.txt in Space)
├── submission_metadata.yaml   ← Fill in before submitting
├── README.md                  ← Full docs
├── SETUP.md                   ← This file
├── job_description.md         ← JD text
│
├── config/                    ← Single source of truth (5 files)
│   ├── intent_schema.json     ← JD intent (16 axes)
│   ├── skill_aliases.yaml     ← ESCO + 200+ AI aliases
│   ├── field_weights.yaml     ← BM25 + embedding weights
│   ├── honeypot_rules.yaml    ← 8 hard + 5 soft rules
│   └── combiner_weights.yaml  ← THE equation
│
├── ranker/                    ← 10 modules
│   ├── types.py               ← Module 0
│   ├── intent.py              ← Module 1
│   ├── encoder.py             ← Module 2
│   ├── filters.py             ← Module 3
│   ├── features.py            ← Modules 4, 6, 6.5
│   ├── availability.py        ← Module 5
│   ├── recall.py              ← Module 7
│   ├── rerank.py              ← Module 8
│   ├── combiner.py            ← Module 9
│   ├── reasoning.py           ← Module 10
│   └── pipeline.py            ← Orchestrator
│
├── precompute/                ← Offline scripts
│   ├── 02_encode_candidates.py
│   └── build_all.sh
│
├── scripts/                   ← Audit + label tools
│   ├── audit_all.py
│   ├── audit_features.py
│   ├── audit_recall_rerank.py
│   ├── audit_final.py
│   ├── label_dev_set.py
│   └── test_filters_on_dev.py
│
├── tests/
│   ├── test_ci.py             ← CI tests (run before submit)
│   └── dev_set/
│       ├── dev_candidates.json  ← 50 sample candidates
│       └── dev_labels.json      ← Heuristic labels
│
└── artifacts/                 ← Pre-computed (gitignored, regenerated)
    └── .gitkeep
```

## 7. Pre-submission checklist

Before you submit, verify each item:

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `bash precompute/build_all.sh candidates.jsonl artifacts/candidate_embeddings.npz` completes
- [ ] `artifacts/candidate_embeddings.npz` exists (~385MB)
- [ ] `artifacts/intent_embeddings.npz` exists (~2MB)
- [ ] `python rank.py --candidates candidates.jsonl --out submission.csv` completes in ≤5 min
- [ ] `submission.csv` has exactly 100 data rows + 1 header row
- [ ] `python tests/test_ci.py` reports 0 failures
- [ ] `python validate_submission.py submission.csv` reports "valid"
- [ ] Top-10 of `submission.csv` looks reasonable (ML/AI engineers, not Marketing Managers)
- [ ] `submission_metadata.yaml` filled in (team name, contact, GitHub repo, sandbox URL)
- [ ] GitHub repo is public (or organizer access granted)
- [ ] HuggingFace Space deployed and URL added to `submission_metadata.yaml`

## 8. Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"
→ `pip install -r requirements.txt` didn't complete. Re-run it.

### "ModuleNotFoundError: No module named 'faiss'"
→ `pip install faiss-cpu` (sometimes needs separate install)

### Pre-compute is slow (>30 min)
→ Make sure you're using CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### `rank.py` exceeds 5 min
→ Reduce `--rerank-k 300` (default 500). Quality drops slightly but still passes.

### Top-10 contains honeypots
→ This shouldn't happen. Run `python scripts/test_filters_on_dev.py` to verify filter is working.

### Reasoning is templated/identical
→ Shouldn't happen. The generator uses 6+ templates with random selection. Check `python tests/test_ci.py` passes the uniqueness test.

### CSV format validation fails
→ Run `python validate_submission.py submission.csv` and follow the error messages.

## 9. Need help?

- Check `README.md` for full architecture docs
- Run the audit suites in `scripts/` to diagnose issues
- Email: bhavishyajain011@gmail.com
