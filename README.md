# DeepFit — Redrob Intelligent Candidate Discovery Ranker

> **Status:** ✅ Feature-complete, validated, submission-ready
> **Team:** Bhavishya Jain (bhavishyajain011@gmail.com)
> **Repo:** https://github.com/Bhavishya011/DeepFit-Multi-Signal-Semantic-Candidate-Ranker
> **Goal:** Top-100 ranked candidate shortlist for Redrob's "Senior AI Engineer — Founding Team" JD

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ranker (produces submission.csv in ~30 seconds)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

The pipeline builds TF-IDF embeddings in-memory (no pre-computation needed) and
produces a valid submission.csv in ~30 seconds on a 16GB CPU machine.

### 3. Validate the submission
```bash
python validate_submission.py submission.csv
```

## Architecture overview

```
Load candidates (~10s) → Phase 1: Honeypot filter (~3s) → Phase 2: FAISS recall (~5s)
→ Phase 3-6: Features + Combiner (~5s) → Phase 7: Reasoning (~3s) → Write CSV (~1s)
Total: ~30s (270s safety margin under 5-min budget)
```

### The 10 modules

| Module | File | Purpose | Status |
|---|---|---|---|
| 0 | `ranker/types.py` | Shared dataclasses (Candidate, Intent, ScoreComponents, FilterVerdict) | ✅ |
| 1 | `ranker/intent.py` | JD-Intent loader (16 axes: 5 positive reqs + 5 nice-to-haves + 5 skill groups + title archetype) | ✅ |
| 2 | `ranker/encoder.py` | Multi-field encoder (BGE-small + TF-IDF fallback) | ✅ |
| 3 | `ranker/filters.py` | Honeypot + trap scorer (8 hard rules + 5 soft rules) | ✅ |
| 4 | `ranker/features.py` | Production-evidence extractor (30+ regex patterns) | ✅ |
| 5 | `ranker/availability.py` | Behavioral availability multiplier [0.30, 1.20] | ✅ |
| 6 | `ranker/features.py` | Structural score (YOE, location, education, work_mode, seniority) | ✅ |
| 6.5 | `ranker/features.py` | Skill taxonomy resolver (ESCO + 200+ custom AI aliases) | ✅ |
| 7 | `ranker/recall.py` | FAISS-only coarse recall (BM25 skipped for speed) | ✅ |
| 8 | `ranker/rerank.py` | Multi-axis semantic scoring (cosine only, no cross-encoder) | ✅ |
| 9 | `ranker/combiner.py` | Final combiner (THE equation) | ✅ |
| 10 | `ranker/reasoning.py` | Evidence-grounded reasoning generator | ✅ |

### The final score equation

```
final_score = base_score × honeypot_penalty × trap_penalty × availability_mult

where:
    base_score = 0.55×semantic + 0.20×structural + 0.10×production
               + 0.10×esco_coverage + 0.05×seniority
    (weights sum to 1.0, asserted at runtime)

    honeypot_penalty ∈ {0.0, 1.0}    # Binary kill switch
    trap_penalty ∈ {0.3, 1.0}        # Soft kill
    availability_mult ∈ [0.30, 1.20] # Continuous modulation
```

## Reproduce command (Stage 3 requirement)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This single command must produce `submission.csv` from `candidates.jsonl` within 5 minutes on a 16GB CPU-only machine with no network access.

**No pre-computation required** — the pipeline builds TF-IDF embeddings in-memory during the ranking step. If you have pre-computed BGE-small embeddings in `artifacts/candidate_embeddings.npz`, the pipeline will use them instead (higher quality).

## Optional: Pre-compute BGE-small embeddings (higher quality)

For better semantic matching, you can pre-compute BGE-small embeddings (one-time, offline):

```bash
# Install sentence-transformers first
pip install sentence-transformers

# Pre-compute (takes ~40 min on GPU, ~3 hrs on CPU)
python precompute/02_encode_candidates.py \
    --candidates candidates.jsonl \
    --output artifacts/candidate_embeddings.npz \
    --backend sentence-transformers
```

The pipeline auto-detects pre-computed embeddings and uses them. If not found, falls back to in-memory TF-IDF.

## Testing

```bash
# CI tests (format, runtime, honeypot, reasoning anti-hallucination)
python tests/test_ci.py

# Audit suites
python scripts/audit_all.py           # Modules 0-3
python scripts/audit_features.py      # Modules 4-6
python scripts/audit_recall_rerank.py # Modules 7-8
python scripts/audit_final.py         # Modules 9-10 + Pipeline
```

## Config files (single source of truth)

| File | Purpose |
|---|---|
| `config/intent_schema.json` | JD intent: 5 positive reqs + 7 disqualifiers + 5 skill groups + title archetypes |
| `config/skill_aliases.yaml` | 17 canonical skills with 200+ AI aliases (RAG, LoRA, vLLM, Pinecone, etc.) |
| `config/field_weights.yaml` | BM25 field weights + embedding field weights + RRF k=60 + semantic axis weights |
| `config/honeypot_rules.yaml` | 8 hard honeypot rules + 5 soft trap rules + reference lists |
| `config/combiner_weights.yaml` | THE final score equation (base weights, multipliers, availability sub-components) |

## Key design decisions

1. **JD-Intent Schema** (not just embedding the JD) — structurally decomposes the JD into positive requirements, nice-to-haves, hard disqualifiers, and context signals. Catches the JD's "between-the-lines" signal that keyword matchers miss.

2. **Multi-field embeddings** (not single blob) — 5 separate embeddings per candidate (title, summary, career, skills, combined). Title gets 3× weight — kills the "Marketing Manager with 9 AI skills" trap.

3. **Hard honeypot filter** (not soft signal) — 8 impossibility rules (skill duration > YOE, salary inverted, inactive+responsive mismatch, etc.). Honeypots in top-100 = disqualification at Stage 3. Filters 36.5% of the 100K pool.

4. **Multiplicative availability** (not additive) — a dead candidate (perfect on paper, 200d inactive, 5% response) sinks to 0.30 × base, below rank 100 cutoff. A perfect active candidate gets 1.20 × base boost.

5. **Evidence-grounded reasoning** (no LLM call) — 6+ templates per archetype with slot-filling from profile facts. Anti-hallucination CI check verifies every cited skill/company/signal against the profile.

6. **FAISS-only recall** (skips BM25) — BM25 takes 1-2 hours on CPU for 100K candidates. FAISS-only recall runs in 5 seconds with minimal quality loss (~1-2% NDCG).

## Team

- **Bhavishya Jain** — bhavishyajain011@gmail.com — Team Lead / ML Engineer

## AI tools declaration

- **Claude** — architecture discussion, JD-intent decomposition, code review
- **GitHub Copilot** — autocomplete

No candidate data was fed to any LLM. No LLM calls during the ranking step.
