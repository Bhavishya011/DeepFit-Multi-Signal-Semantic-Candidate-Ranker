# DeepFit

> **An AI recruiter that reads between the lines of a job description.**
>
> Most candidate-matching systems filter. DeepFit *understands* — then ranks.

---

## The Problem

Redrob's Senior AI Engineer JD is a **meta-trap**. It says "5-9 years experience" but means "judgment, not tenure." It lists skills but warns against "framework enthusiasts." It disqualifies pure researchers, LangChain-only newcomers, title-chasers, and services-only careers — all in prose that keyword matchers cannot parse.

A naive embedding ranker will surface ML-research PhDs and LangChain-bootcamp juniors. It will be fooled by a Marketing Manager who stuffed 9 AI skills into their profile. It will rank a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate — because semantic similarity is blind to availability.

The spec is explicit: *"The right answer involves reasoning about the gap between what the JD says and what the JD means."*

DeepFit is built to reason about that gap.

---

## The Insight

Three observations shaped the architecture:

### 1. The JD is structured, not a blob

A job description is not free text — it's a contract with positive requirements, negative disqualifiers, and context signals. Embedding the JD as a single vector collapses this structure. We decompose it instead: 5 positive requirements (with weights), 5 nice-to-haves, 7 hard disqualifiers, and context signals (location, notice period, engagement). Each axis gets its own embedding, so we can score a candidate *per-axis* and explain *why*.

### 2. Impossibilities are not low scores — they are zero

A candidate who claims 88 months of Pinecone experience but has only 72 months of total work experience is not a "weak match." They are a honeypot. A candidate whose salary range is inverted (min > max) is not "marginally relevant." They are noise. Treating these as soft signals lets them contaminate the top-100. DeepFit hard-filters them: 8 impossibility rules remove 36.5% of the 100K pool before ranking even begins.

### 3. Availability is multiplicative, not additive

A perfect semantic match who is 200 days inactive with a 5% response rate is, for hiring purposes, not available. Adding `+0.1` for active candidates doesn't capture this — the dead candidate still ranks high. DeepFit uses a multiplicative availability multiplier in `[0.30, 1.20]`. A dead candidate's score is capped at 30% of their base, sinking them below rank 100. An active, open-to-work candidate with strong response rates gets a 20% boost. The multiplier is the difference between a ranked list and a *hirable* list.

---

## The Architecture

DeepFit is a 10-module pipeline that flows from JD understanding to ranked output. Every module has a specific job; every design choice has a reason.

```
                    ┌─────────────────────────────────────────┐
                    │           JOB DESCRIPTION               │
                    │   "Senior AI Engineer — Founding Team"  │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │   Module 1: JD-Intent Schema            │
                    │   16 axes: 5 reqs + 5 nice-to-haves +   │
                    │   7 disqualifiers + context signals     │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │   Module 2: Multi-Field Encoder         │
                    │   5 embeddings per candidate            │
                    │   title (3x) · career (2x) ·            │
                    │   summary (1.5x) · skills (1x)          │
                    └────────────────┬────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │   Module 3: Honeypot + Trap Filter           │
              │   8 hard rules → score = 0 (excluded)        │
              │   5 soft rules → score × 0.3 (sinks)         │
              │   Result: 36.5% of pool filtered             │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │   Module 7: FAISS Coarse Recall              │
              │   Top-1000 by inner-product similarity       │
              │   to JD intent query embedding               │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │   Module 8: Multi-Axis Semantic Scoring      │
              │   6 cosine axes:                             │
              │   title_fit · career_retrieval ·             │
              │   career_ranking · career_llm ·              │
              │   skills_coverage · cross_enc                │
              └──────────────────────┬──────────────────────┘
                                     │
         ┌───────────────────────────▼───────────────────────────┐
         │   Modules 4, 6, 6.5: Feature Extraction                │
         │   • Production evidence (30+ regex: deployed, shipped, │
         │     A/B test, NDCG, vector search, SLO)                │
         │   • ESCO + 200 custom AI aliases (RAG, LoRA, vLLM,     │
         │     Pinecone, FAISS, LangChain, MLflow)                │
         │   • Structural: YOE band, location tier, education,    │
         │     work mode, seniority                               │
         └───────────────────────────┬───────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │   Module 5: Behavioral Availability          │
              │   Multiplicative [0.30, 1.20]                │
              │   recency × response × open_to_work ×        │
              │   notice × github × demand × verified        │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │   Module 9: Final Combiner                   │
              │                                              │
              │   final = base × honeypot × trap × avail     │
              │                                              │
              │   base = 0.55·sem + 0.20·struct + 0.10·prod  │
              │        + 0.10·esco + 0.05·seniority          │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │   Module 10: Reasoning Generator             │
              │   6 templates · rank-aware tone ·            │
              │   anti-hallucination CI checks ·             │
              │   100/100 unique strings                      │
              └──────────────────────┬──────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │         submission.csv                   │
                    │   100 rows · validated · 37 seconds      │
                    └─────────────────────────────────────────┘
```

---

## The Final Score Equation

One formula. No ambiguity. Every teammate implements the same combiner.

```
final = base × honeypot × trap × availability
```

**`base`** is additive — it captures overall goodness of fit. The weights sum to 1.0, so `base ∈ [0, 1]` reads as "fraction of ideal fit":

| Component | Weight | What it measures |
|---|---|---|
| Semantic | 0.55 | Multi-axis cosine similarity to JD intent |
| Structural | 0.20 | YOE band, location tier, education, work mode |
| Production | 0.10 | Regex evidence of shipped systems in career history |
| ESCO Coverage | 0.10 | Skill taxonomy match (with 200+ AI aliases) |
| Seniority | 0.05 | Title level + archetype cosine match |

**`honeypot`** is binary `{0.0, 1.0}` — a kill switch. If any of 8 impossibility rules fire, the candidate is excluded. No soft landing.

**`trap`** is binary `{0.3, 1.0}` — a soft kill. If any of 5 trap rules fire (title-skill mismatch, services-only career, etc.), the candidate's score is capped at 30% of base. They sink below rank 100 but aren't zeroed, so we can audit borderline cases.

**`availability`** is continuous `[0.30, 1.20]` — the modulation that turns a ranked list into a hirable list. Nine sub-components multiplied together:

```
availability = recency × response × open_to_work × verified ×
               notice × github × demand × linkedin × email
```

A dead candidate (200d inactive, 5% response, not open-to-work, 120d notice) gets `availability ≈ 0.30`. Even with `base = 1.0`, their final score is `0.30` — below the rank-100 cutoff. An active candidate (1d inactive, 95% response, open-to-work, 30d notice, GitHub 80) gets `availability ≈ 1.20`. Their final score is boosted 20% above base.

---

## The Honeypot Filter

This is the highest-ROI module. The spec warns: *"submissions with honeypot rate > 10% in top 100 are disqualified."* DeepFit achieves **0%**.

### 8 Hard Rules (score = 0)

| Rule | What it catches |
|---|---|
| Single skill duration > YOE + 6mo | Pinecone 88mo but YOE only 72mo |
| Skill duration sum > 10× YOE | Massive stuffing (50 skills at 80mo each) |
| Endorsement inflation | 5+ expert skills with 0 endorsements |
| Career date impossibility | Career started >5yr before education ended |
| YOE inflation | Claimed YOE > actual career span + 2yr grace |
| Salary range inverted | min > max — data quality honeypot |
| Inactive + responsive mismatch | 180d+ inactive but 80%+ response rate |
| Interview rate contradiction | Interview >0 but no offer history + no applications |

### 5 Soft Trap Rules (score × 0.3)

| Rule | What it catches |
|---|---|
| Title-skill mismatch | "Marketing Manager with 9 AI skills" — the JD's explicit trap |
| Services-only career | Entire career at TCS/Infosys/Wipro (JD disqualifier) |
| Pure research background | All titles are Researcher/Postdoc (JD disqualifier) |
| Recent LangChain-only | All AI skills <12mo + no pre-LLM ML experience |
| Title-chaser pattern | 3+ roles <18mo + Senior→Staff→Principal progression |

On the 100K pool: **36,507 candidates filtered (36.5%)**, leaving 63,493 for ranking.

---

## The Reasoning Generator

Stage 4 manual review samples 10 random rows and checks reasoning against 6 criteria: specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency. DeepFit's generator is built to pass all 6.

**No LLM calls.** Template-based slot-filling with 6 variants per archetype. Every cited fact is extracted directly from the candidate's profile — the generator cannot hallucinate by construction.

**Anti-hallucination CI checks** verify:
- Every skill mentioned exists in `candidate.skills` or `career_history.description`
- Every response rate / days-inactive / YOE claim matches the profile
- Word-boundary regex prevents false matches ("rag" in "coverage" doesn't trigger RAG)
- All 100 reasoning strings are unique (no template monotony)

**Rank-aware tone:** "strong fit" (rank 1-10), "solid match" (11-30), "reasonable match" (31-60), "adjacent candidate" (61-100). A rank-5 candidate with critical reasoning, or a rank-95 candidate with glowing reasoning, would fail the rank-consistency check.

**Example outputs:**
```
#1  "strong fit: Senior ML Engineer at Genpact AI; active 28d ago, open to work, 88% response rate."
#47 "solid match: 6.7 yrs experience (in JD band); 20% response rate. Note: last active 81d ago."
#99 "adjacent candidate: 2.4 yrs at TCS. Concern: services_only_career pattern detected."
```

---

## The JD-Intent Schema

The JD is hand-decomposed into a structured schema (no LLM dependency — human-reviewed for defensibility at Stage 5 interview):

**5 Positive Requirements (must-have):**
1. Embeddings-based retrieval (production experience)
2. Vector database / hybrid search (production experience)
3. Strong Python (code quality matters)
4. Ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing)
5. 5-9 years experience (band, not hard requirement)

**5 Nice-to-haves:** LoRA/QLoRA/PEFT · Learning-to-rank · HR-tech exposure · Distributed systems · OSS contributions

**7 Hard Disqualifiers:** Pure research (no production) · LangChain-only (<12mo) · No recent code (18mo) · Title-chasers · Services-only career · CV/speech/robotics-only · Closed-source-only without validation

**5 Skill Taxonomy Groups:** Core retrieval/IR · Ranking evaluation · Production ML · LLM engineering · Core Python

Each axis gets its own embedding vector, enabling per-axis scoring. A candidate can score high on "career_retrieval" but low on "career_ranking" — and the reasoning generator can explain exactly which axis drove their rank.

---

## The Skill Taxonomy Resolver

ESCO (the European skills taxonomy) is the standard, but it lags 2-3 years behind industry usage. Real 2024-2026 AI resumes say things like "fine-tuned Mistral-7B with LoRA" or "built RAG with LangChain and Pinecone" — none of which appear in ESCO.

DeepFit layers a custom alias file on top of ESCO: **17 canonical skills with 200+ aliases** covering modern AI terminology:

| Canonical Skill | Sample Aliases |
|---|---|
| Vector Database | Pinecone, Weaviate, Qdrant, Milvus, FAISS, HNSW, pgvector, Chroma |
| LLM Fine-tuning | LoRA, QLoRA, PEFT, RLHF, DPO, fine-tuned Mistral/Llama/Qwen |
| Retrieval Augmented Generation | RAG, retrieval augmented generation, grounded generation |
| LLM Serving | vLLM, TGI, Triton, TensorRT-LLM, SGLang |
| Ranking Evaluation | NDCG, MRR, MAP, Precision@K, A/B testing, offline-to-online |
| Learning to Rank | LambdaMART, XGBoost ranking, LightGBM ranking, LTR |

Without this, a candidate who writes "Built RAG with LangChain and Pinecone, fine-tuned Mistral with QLoRA, served via vLLM" gets **zero** credit from ESCO. With DeepFit's alias resolver, they get credit for 4 canonical skills that map directly to JD requirements.

---

## The Production-Evidence Extractor

The JD explicitly disqualifies "pure research environments without any production deployment." A candidate whose career history mentions only papers, posters, and benchmarks should be deprioritized *even if their semantic match is high*.

DeepFit regex-mines `career_history[].description` for **30+ production signals**:

**Positive (boost):** deployed to production · shipped · served N users · A/B test · monitoring · SLO · canary · rollout · NDCG · MRR · vector search · dense retrieval · embedding drift · retrieval quality regression

**Negative (penalty):** paper · publication · arxiv · neurips · ICML · CVPR · thesis · dissertation · research lab · benchmarks only · no production

The score is a sigmoid of `(production_hits × 0.15) − (research_hits × 0.10)`, with boosts for production-dominant profiles and heavy penalties for research-only.

---

## Results

On the full 100K candidate pool:

| Metric | Value |
|---|---|
| Total runtime | **37 seconds** (5-min budget — 88% safety margin) |
| Candidates processed | 100,000 |
| Honeypots filtered | 36,507 (36.5%) |
| Honeypots in top-100 | **0%** (spec requires <10%) |
| Top score | 0.6213 (Senior ML Engineer @ Genpact AI) |
| Rank-100 score | 0.3135 (clear discrimination) |
| Unique reasoning strings | 100/100 |
| Reasoning length | All ≤250 chars |
| Official validator | **"Submission is valid."** |

**Top-5 candidates:**
1. Senior ML Engineer @ Genpact AI — 88% response, active 28d
2. Senior ML Engineer @ Flipkart — 87% response, active 34d
3. Senior NLP Engineer, 7.8 YOE — 89% response, active 41d
4. Search Engineer, 7.6 YOE — 94% response, active 30d
5. Sr Software Engineer (ML) @ Tech Mahindra — 79% response, active 29d

**Title distribution in top-100:** 17 ML Engineer · 17 Sr Software Engineer (ML) · 10 Jr ML Engineer · 9 AI Specialist · 6 AI Research Engineer — **zero Marketing Managers**.

---

## Design Principles

1. **Config-driven, not code-driven.** Every weight, threshold, and rule lives in 5 YAML/JSON config files. Tuning never requires code changes. The combiner equation is a config, not a hardcoded formula.

2. **Fail gracefully.** If pre-computed BGE embeddings aren't available, the pipeline builds TF-IDF embeddings in-memory. If sentence-transformers isn't installed, it falls back to scikit-learn. The pipeline runs on ANY CPU without network access.

3. **No LLM calls during ranking.** The spec forbids it, and it's the right call — template-based reasoning with anti-hallucination CI is more defensible at Stage 5 interview than LLM-generated text.

4. **Explainability by construction.** Every score decomposes into weighted sub-scores with named axes. Every reasoning string cites specific profile facts. The system cannot produce a score it cannot explain.

5. **Honeypots are not low scores.** They are zero. Soft signals let impossibilities contaminate the top-100. Hard filters don't.

---

## Project Structure

```
DeepFit/
├── rank.py                    ← Entry point: python rank.py --candidates ... --out ...
├── config/                    ← Single source of truth (5 files)
│   ├── intent_schema.json     ← JD decomposed into 16 axes + 7 disqualifiers
│   ├── skill_aliases.yaml     ← 17 canonical skills, 200+ AI aliases
│   ├── honeypot_rules.yaml    ← 8 hard rules + 5 soft trap rules
│   ├── field_weights.yaml     ← Embedding/BM25/semantic axis weights
│   └── combiner_weights.yaml  ← THE final score equation
├── ranker/                    ← 10 modules
│   ├── types.py               ← Shared dataclasses
│   ├── intent.py              ← M1: JD-Intent loader
│   ├── encoder.py             ← M2: Multi-field encoder (BGE + TF-IDF fallback)
│   ├── filters.py             ← M3: Honeypot + trap scorer
│   ├── features.py            ← M4/6/6.5: Production evidence + structural + ESCO
│   ├── availability.py        ← M5: Behavioral availability multiplier
│   ├── recall.py              ← M7: FAISS coarse recall
│   ├── rerank.py              ← M8: Multi-axis semantic scoring
│   ├── combiner.py            ← M9: Final combiner (THE equation)
│   ├── reasoning.py           ← M10: Evidence-grounded reasoning
│   └── pipeline.py            ← Orchestrator
├── precompute/                ← Optional: BGE-small embedding pre-computation
├── tests/                     ← CI tests + 50-candidate dev set
└── scripts/                   ← 4 audit suites + labeling tools
```

---

## Reproduce

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

37 seconds. CPU. No GPU. No network. No pre-computation required.

---

## Team

**Bhavishya Jain** — bhavishyajain011@gmail.com

## AI Tools Declaration

- **Claude** — architecture discussion, JD-intent decomposition, code review
- **GitHub Copilot** — autocomplete

No candidate data was fed to any LLM. No LLM calls during the ranking step. The JD-intent schema is hand-crafted and human-reviewed for defensibility at Stage 5 interview.

---

*DeepFit: Because the right answer is not "find candidates whose skills section contains the most AI keywords." It's reasoning about the gap between what the JD says and what the JD means.*
