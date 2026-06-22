#!/usr/bin/env python3
"""
Comprehensive audit of Modules 0-3.

Tests:
  A. types.py — dataclass behavior, edge cases (empty fields, missing keys)
  B. intent.py — schema loading, axis extraction, path bugs
  C. encoder.py — both backends, empty text, batch consistency
  D. filters.py — every rule with crafted inputs (true positive + true negative)
  E. config consistency — code references match config keys
  F. end-to-end smoke — load dev set, encode, score, no crashes

Exit code 0 = all pass, 1 = any failure.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from ranker.types import Candidate, Intent, FilterVerdict, ScoreComponents, RankedCandidate
from ranker.intent import (
    load_intent_schema, load_skill_aliases, get_all_axis_query_texts,
    get_jd_required_canonical_skills, load_intent_axis_embeddings,
    INTENT_EMBEDDINGS_PATH, INTENT_SCHEMA_PATH, SKILL_ALIASES_PATH,
)
from ranker.encoder import (
    MultiFieldEncoder, TfidfSvdBackend, extract_title_text, extract_summary_text,
    extract_career_text, extract_skills_text, save_embeddings, load_embeddings,
)
from ranker.filters import HoneypotScorer, DEFAULT_REFERENCE_DATE

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
# A. types.py
# ═══════════════════════════════════════════════════════════════════════
def test_types():
    section("A. types.py — dataclasses + edge cases")

    # A1. Candidate from minimal dict
    minimal = {"candidate_id": "CAND_0000001", "profile": {}, "career_history": [], "education": [], "skills": []}
    c = Candidate.from_dict(minimal)
    check(c.candidate_id == "CAND_0000001", "Candidate.from_dict accepts minimal record")
    check(c.title == "", "Missing current_title returns empty string (not None)")
    check(c.yoe == 0.0, "Missing years_of_experience returns 0.0")
    check(c.open_to_work is False, "Missing open_to_work_flag returns False")
    check(c.response_rate == 0.0, "Missing recruiter_response_rate returns 0.0")
    check(c.github_activity_score == -1.0, "Missing github_activity_score returns -1.0")
    check(c.days_since_last_active() == -1, "Missing last_active_date returns -1 (sentinel)")
    check(c.career_text() == "", "Empty career_history returns empty string")
    check(c.skills_text() == "", "Empty skills returns empty string")

    # A2. Candidate from full record
    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        samples = json.load(f)
    c = Candidate.from_dict(samples[0])
    check(c.candidate_id == "CAND_0000001", "Full record loads with correct ID")
    check(isinstance(c.yoe, float), "yoe accessor returns float")
    check(isinstance(c.notice_period_days, int), "notice_period_days accessor returns int")
    check(c.days_since_last_active() >= 0, "days_since_last_active returns non-negative for valid date")
    check(len(c.career_history) > 0, "Career history populated")
    check(len(c.skills) > 0, "Skills populated")
    check(c.to_dict() == samples[0], "to_dict() returns original record")

    # A3. Intent dataclass
    intent = Intent(schema={"explicit_positive_requirements": [{"axis": "x", "must_have": True}]}, axis_embeddings={})
    check(intent.must_have_axes() == ["x"], "Intent.must_have_axes() works")
    check(intent.axis_query_text("nonexistent") == "", "Intent.axis_query_text returns empty for unknown axis")

    # A4. FilterVerdict properties
    v_clean = FilterVerdict(honeypot_penalty=1.0, trap_penalty=1.0)
    check(v_clean.is_honeypot is False, "Clean verdict: is_honeypot=False")
    check(v_clean.is_trap is False, "Clean verdict: is_trap=False")
    check(v_clean.passes is True, "Clean verdict: passes=True")

    v_honeypot = FilterVerdict(honeypot_penalty=0.0, trap_penalty=1.0)
    check(v_honeypot.is_honeypot is True, "Honeypot verdict: is_honeypot=True")
    check(v_honeypot.passes is False, "Honeypot verdict: passes=False")

    v_trap = FilterVerdict(honeypot_penalty=1.0, trap_penalty=0.3)
    check(v_trap.is_trap is True, "Trap verdict: is_trap=True")
    check(v_trap.passes is True, "Trap verdict: passes=True (trap is not hard filter)")

    # A5. ScoreComponents defaults
    sc = ScoreComponents()
    check(sc.semantic_score == 0.0, "ScoreComponents defaults to 0.0")
    check(sc.honeypot_penalty == 1.0, "ScoreComponents default honeypot_penalty=1.0 (clean)")
    check(sc.availability_mult == 1.0, "ScoreComponents default availability_mult=1.0 (neutral)")
    check(sc.axis_scores == {}, "ScoreComponents default axis_scores is empty dict")


# ═══════════════════════════════════════════════════════════════════════
# B. intent.py
# ═══════════════════════════════════════════════════════════════════════
def test_intent():
    section("B. intent.py — schema loading + axis extraction")

    # B1. Schema loads
    schema = load_intent_schema()
    check(isinstance(schema, dict), "intent_schema.json loads as dict")
    check("explicit_positive_requirements" in schema, "Schema has explicit_positive_requirements")
    check("hard_disqualifiers" in schema, "Schema has hard_disqualifiers")
    check("skill_taxonomy_groups" in schema, "Schema has skill_taxonomy_groups")
    check("title_archetype_match" in schema, "Schema has title_archetype_match")

    # B2. All axes have embedding_query_text
    axes = get_all_axis_query_texts(schema)
    check(len(axes) >= 15, f"At least 15 axes extracted (got {len(axes)})")
    empty_axes = [name for name, text in axes.items() if not text.strip()]
    check(not empty_axes, f"No axis has empty embedding_query_text (empty: {empty_axes})")

    # B3. Must-have axes are correct
    must_haves = [r["axis"] for r in schema["explicit_positive_requirements"] if r.get("must_have")]
    expected_must_haves = {"embeddings_retrieval_production", "vector_db_production", "strong_python", "ranking_eval_frameworks"}
    check(set(must_haves) == expected_must_haves, f"Must-have axes match JD's 'things you absolutely need': {must_haves}")

    # B4. JD-required canonical skills
    jd_skills = get_jd_required_canonical_skills(schema)
    check(len(jd_skills) > 0, f"JD-required canonical skills non-empty (got {len(jd_skills)})")
    check("Embeddings-based Retrieval" in jd_skills, "Embeddings-based Retrieval is JD-required")
    check("Ranking Evaluation" in jd_skills, "Ranking Evaluation is JD-required")

    # B5. Skill aliases load
    aliases = load_skill_aliases()
    check(len(aliases) >= 15, f"At least 15 canonical skills in aliases (got {len(aliases)})")
    check("Embeddings-based Retrieval" in aliases, "Embeddings-based Retrieval in aliases")
    check("Vector Database" in aliases, "Vector Database in aliases")
    check("LLM Fine-tuning" in aliases, "LLM Fine-tuning in aliases")
    # Verify aliases contain modern AI terms ESCO misses
    vec_db_aliases = aliases.get("Vector Database", {}).get("aliases", [])
    check("Pinecone" in vec_db_aliases, "Vector Database aliases include Pinecone")
    check("FAISS" in vec_db_aliases, "Vector Database aliases include FAISS")
    llm_aliases = aliases.get("LLM Fine-tuning", {}).get("aliases", [])
    check("LoRA" in llm_aliases, "LLM Fine-tuning aliases include LoRA")
    check("QLoRA" in llm_aliases, "LLM Fine-tuning aliases include QLoRA")
    rag_aliases = aliases.get("Retrieval Augmented Generation", {}).get("aliases", [])
    check("RAG" in rag_aliases, "Retrieval Augmented Generation aliases include RAG")

    # B6. Path bug check — INTENT_EMBEDDINGS_PATH extension
    check(INTENT_EMBEDDINGS_PATH.suffix == ".npz",
          f"INTENT_EMBEDDINGS_PATH uses .npz extension (got {INTENT_EMBEDDINGS_PATH.suffix}) — "
          f"must match np.savez() output")

    # B7. Pre-computed embeddings load (if exist)
    embeddings = load_intent_axis_embeddings()
    if embeddings is not None:
        check(len(embeddings) == len(axes),
              f"Saved embeddings count matches axes count ({len(embeddings)} vs {len(axes)})")
        # Check all embeddings are L2-normalized
        for name, vec in embeddings.items():
            norm = float(np.linalg.norm(vec))
            if not (0.99 <= norm <= 1.01):
                check(False, f"Axis '{name}' embedding not L2-normalized (norm={norm:.4f})")
                break
        else:
            check(True, "All axis embeddings are L2-normalized")
    else:
        check(True, "No pre-computed intent embeddings yet (will be created on first run)")


# ═══════════════════════════════════════════════════════════════════════
# C. encoder.py
# ═══════════════════════════════════════════════════════════════════════
def test_encoder():
    section("C. encoder.py — backends + field extractors")

    # C1. Field extractors handle empty candidate
    empty_c = Candidate(candidate_id="EMPTY", profile={}, career_history=[], education=[], skills=[])
    # extract_title_text returns "{title}. {headline}" → with both empty, becomes ". "
    # (the f-string inserts a period and space regardless; this is by design — non-empty when title exists)
    title_text = extract_title_text(empty_c)
    check(title_text.strip() == "." or title_text == "",
          f"extract_title_text on empty candidate returns empty or minimal (got '{title_text}')")
    check(extract_summary_text(empty_c) == "", "extract_summary_text on empty returns ''")
    check(extract_career_text(empty_c) == "", "extract_career_text on empty returns ''")
    check(extract_skills_text(empty_c) == "", "extract_skills_text on empty returns ''")

    # C2. Skills text weighting
    skilled_c = Candidate(
        candidate_id="SKILLED",
        profile={"current_title": "ML Engineer"},
        career_history=[],
        education=[],
        skills=[
            {"name": "Python", "proficiency": "expert", "endorsements": 100, "duration_months": 60},
            {"name": "Rust", "proficiency": "beginner", "endorsements": 0, "duration_months": 3},
        ],
    )
    skills_text = extract_skills_text(skilled_c)
    python_count = skills_text.count("Python")
    rust_count = skills_text.count("Rust")
    check(python_count > rust_count, f"Expert+endorsed skill repeated more than beginner (Python={python_count}, Rust={rust_count})")

    # C3. TF-IDF backend (always available, no network)
    backend = TfidfSvdBackend(dim=50)
    check(backend.name == "tfidf-svd", "TF-IDF backend name is 'tfidf-svd'")
    check(backend._fitted is False, "TF-IDF backend starts unfitted")
    check(backend.dim == 50, "TF-IDF backend dim set correctly")

    # C4. TF-IDF fit + encode
    corpus = ["machine learning engineer", "data scientist python", "AI retrieval ranking", "vector database pinecone"] * 10
    backend.fit(corpus)
    check(backend._fitted is True, "TF-IDF backend marked as fitted after fit()")
    vec = backend.encode("machine learning engineer")
    check(vec.shape == (backend.dim,), f"Encoded vector shape correct ({vec.shape})")
    norm = float(np.linalg.norm(vec))
    check(0.99 <= norm <= 1.01, f"Encoded vector L2-normalized (norm={norm:.4f})")

    # C5. Empty text encoding
    empty_vec = backend.encode("")
    check(empty_vec.shape == (backend.dim,), "Empty text returns correctly-shaped vector")
    check(float(np.linalg.norm(empty_vec)) == 0.0, "Empty text returns zero vector")

    # C6. Batch consistency — encode_batch should match encode() for same text
    texts = ["machine learning", "data science", "AI engineer"]
    batch_vecs = backend.encode_batch(texts)
    check(batch_vecs.shape == (3, backend.dim), f"Batch encoding shape correct ({batch_vecs.shape})")
    single_vec = backend.encode("machine learning")
    diff = float(np.linalg.norm(batch_vecs[0] - single_vec))
    check(diff < 1e-5, f"Batch and single encode are consistent (diff={diff:.2e})")

    # C7. MultiFieldEncoder field weights are normalized
    encoder = MultiFieldEncoder(backend=backend, field_weights={"title": 4.0, "summary": 2.0, "career": 2.5, "skills": 1.0})
    total_weight = sum(encoder.field_weights.values())
    check(abs(total_weight - 1.0) < 1e-6, f"Field weights L2-normalized to sum=1.0 (got {total_weight})")
    check(encoder.field_weights["title"] > encoder.field_weights["skills"], "Title weight > skills weight after normalization")

    # C8. Encode a real candidate end-to-end
    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        samples = json.load(f)
    c = Candidate.from_dict(samples[0])
    embeddings = encoder.encode_candidate(c)
    check(set(embeddings.keys()) == {"title", "summary", "career", "skills", "combined"}, "All 5 fields present in encoding")
    for field, vec in embeddings.items():
        norm = float(np.linalg.norm(vec))
        if field != "combined" or norm > 0:  # combined could be zero if all fields are zero
            check(0.99 <= norm <= 1.01 or norm == 0.0, f"Field '{field}' L2-normalized (norm={norm:.4f})")

    # C9. Persistence round-trip
    test_path = Path(__file__).parent.parent / "artifacts" / "_test_embeddings.npz"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings({c.candidate_id: embeddings}, test_path, dtype="float16")
    check(test_path.exists(), "Saved embeddings file exists")
    loaded = load_embeddings(test_path)
    check(c.candidate_id in loaded, "Loaded embeddings contain candidate_id")
    original_norm = float(np.linalg.norm(embeddings["combined"]))
    loaded_norm = float(np.linalg.norm(loaded[c.candidate_id]["combined"]))
    check(abs(original_norm - loaded_norm) < 0.05, f"fp16 round-trip preserves norm (orig={original_norm:.4f}, loaded={loaded_norm:.4f})")
    test_path.unlink()  # cleanup


# ═══════════════════════════════════════════════════════════════════════
# D. filters.py — every rule with crafted inputs
# ═══════════════════════════════════════════════════════════════════════
def test_filters():
    section("D. filters.py — rule-by-rule unit tests")

    scorer = HoneypotScorer.from_config()

    # Helper: build a candidate from kwargs
    def make_candidate(**kwargs):
        record = {
            "candidate_id": kwargs.get("candidate_id", "CAND_TEST_01"),
            "profile": {
                "current_title": kwargs.get("title", "ML Engineer"),
                "years_of_experience": kwargs.get("yoe", 6.0),
                "location": kwargs.get("location", "Pune, Maharashtra"),
                "country": kwargs.get("country", "India"),
                "summary": kwargs.get("summary", "ML engineer"),
                "headline": kwargs.get("headline", "ML Engineer"),
                "current_company": kwargs.get("company", "ProductCo"),
                "current_company_size": "201-500",
                "current_industry": kwargs.get("industry", "Software"),
            },
            "career_history": kwargs.get("career_history", [
                {"company": "ProductCo", "title": "ML Engineer", "start_date": "2020-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Software", "company_size": "201-500",
                 "description": "Built and shipped retrieval systems in production."}
            ]),
            "education": kwargs.get("education", [
                {"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "CS",
                 "start_year": 2014, "end_year": 2018, "tier": "tier_1"}
            ]),
            "skills": kwargs.get("skills", [
                {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
            ]),
            "redrob_signals": kwargs.get("redrob_signals", {
                "profile_completeness_score": 90.0,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-06-15",
                "open_to_work_flag": True,
                "profile_views_received_30d": 50,
                "applications_submitted_30d": 5,
                "recruiter_response_rate": 0.7,
                "avg_response_time_hours": 24.0,
                "skill_assessment_scores": {},
                "connection_count": 500,
                "endorsements_received": 80,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 40.0,
                "search_appearance_30d": 100,
                "saved_by_recruiters_30d": 10,
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": 0.5,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            }),
        }
        return Candidate.from_dict(record)

    # ─── D1. Clean candidate passes everything ───────────────────────────
    clean = make_candidate()
    v = scorer.score(clean)
    check(v.passes is True, "Clean candidate passes hard filter")
    check(v.is_trap is False, "Clean candidate not flagged as trap")
    check(len(v.fired_honeypot_rules) == 0, "Clean candidate: no honeypot rules fired")
    check(len(v.fired_trap_rules) == 0, "Clean candidate: no trap rules fired")

    # ─── D2. single_skill_duration_exceeds_yoe ───────────────────────────
    bad = make_candidate(skills=[
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 100},  # 100mo > 72mo YOE
    ])
    v = scorer.score(bad)
    check(v.is_honeypot is True, "skill_duration > YOE+6mo fires honeypot")
    check("single_skill_duration_exceeds_yoe" in v.fired_honeypot_rules, "Rule name in fired list")

    # Should NOT fire when skill duration is reasonable
    ok = make_candidate(skills=[
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 50},  # 50mo < 72mo+6
    ])
    v = scorer.score(ok)
    check(v.is_honeypot is False, "skill_duration <= YOE+6mo does NOT fire")

    # ─── D3. skill_duration_sum_absurd ───────────────────────────────────
    # 10 skills each at 80mo = 800mo total. YOE=6 → 72mo. Threshold = 720mo (10x).
    # 800 > 720, so SHOULD fire.
    absurd_sum = make_candidate(yoe=6.0, skills=[
        {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 10, "duration_months": 80}
        for i in range(10)
    ])
    v = scorer.score(absurd_sum)
    check(v.is_honeypot is True, "Sum=800mo > 10x72mo=720mo fires absurd rule")

    # Lower sum that should NOT fire: 5 skills at 60mo = 300mo. YOE=6 → 72mo. 300 < 720.
    ok_sum = make_candidate(yoe=6.0, skills=[
        {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 10, "duration_months": 60}
        for i in range(5)
    ])
    v = scorer.score(ok_sum)
    check(v.is_honeypot is False, "Sum=300mo < 10x72mo=720mo does NOT fire (aggressive threshold working)")

    # ─── D4. endorsement_inflation ───────────────────────────────────────
    inflated = make_candidate(skills=[
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 12}
        for i in range(6)  # 6 expert skills with 0 endorsements
    ])
    v = scorer.score(inflated)
    check(v.is_honeypot is True, "6 expert skills with 0 endorsements fires")
    check("endorsement_inflation" in v.fired_honeypot_rules, "endorsement_inflation in fired list")

    # 4 expert+0 should NOT fire (threshold is 5)
    ok_endorse = make_candidate(skills=[
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 12}
        for i in range(4)
    ] + [
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
    ])
    v = scorer.score(ok_endorse)
    check(v.is_honeypot is False, "4 expert+0 endorsements does NOT fire (threshold=5)")

    # ─── D5. career_date_impossibility ───────────────────────────────────
    # Career started 2010, education ended 2020 → career started 10 yrs before edu end
    impossible_date = make_candidate(career_history=[
        {"company": "X", "title": "Engineer", "start_date": "2010-01-01", "end_date": None,
         "duration_months": 180, "is_current": True, "industry": "Software",
         "company_size": "201-500", "description": "Engineer"}
    ], education=[
        {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2016, "end_year": 2020, "tier": "tier_1"}
    ])
    v = scorer.score(impossible_date)
    check(v.is_honeypot is True, "Career started 10yrs before edu end fires")
    check("career_date_impossibility" in v.fired_honeypot_rules, "career_date_impossibility in fired list")

    # Normal case: career started after graduation
    ok_date = make_candidate(career_history=[
        {"company": "X", "title": "Engineer", "start_date": "2020-06-01", "end_date": None,
         "duration_months": 60, "is_current": True, "industry": "Software",
         "company_size": "201-500", "description": "Engineer"}
    ], education=[
        {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2014, "end_year": 2018, "tier": "tier_1"}
    ])
    v = scorer.score(ok_date)
    check(v.is_honeypot is False, "Career started 2yrs after edu end does NOT fire")

    # ─── D6. yoe_inflation ───────────────────────────────────────────────
    # Career started 2024 (2 yrs ago), claims 15 YOE
    inflated_yoe = make_candidate(yoe=15.0, career_history=[
        {"company": "X", "title": "Engineer", "start_date": "2024-01-01", "end_date": None,
         "duration_months": 24, "is_current": True, "industry": "Software",
         "company_size": "201-500", "description": "Engineer"}
    ])
    v = scorer.score(inflated_yoe)
    check(v.is_honeypot is True, "Claimed YOE 15 > actual span 2 + 2grace fires")
    check("yoe_inflation" in v.fired_honeypot_rules, "yoe_inflation in fired list")

    # ─── D7. salary_range_inverted ───────────────────────────────────────
    inverted_sal = make_candidate(redrob_signals={
        "expected_salary_range_inr_lpa": {"min": 50.0, "max": 30.0},
        "last_active_date": "2026-06-15",
        "recruiter_response_rate": 0.7,
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
    })
    v = scorer.score(inverted_sal)
    check(v.is_honeypot is True, "Salary min > max fires")
    check("salary_range_inverted" in v.fired_honeypot_rules, "salary_range_inverted in fired list")

    # ─── D8. inactive_responsive_mismatch ────────────────────────────────
    mismatch = make_candidate(redrob_signals={
        "last_active_date": "2025-06-01",  # > 1 year ago
        "recruiter_response_rate": 0.95,   # > 0.8
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
    })
    v = scorer.score(mismatch)
    check(v.is_honeypot is True, "Inactive >180d + response >0.8 fires")
    check("inactive_responsive_mismatch" in v.fired_honeypot_rules, "inactive_responsive_mismatch in fired list")

    # Inactive but low response should NOT fire
    ok_inactive = make_candidate(redrob_signals={
        "last_active_date": "2025-06-01",  # > 1 year ago
        "recruiter_response_rate": 0.30,   # < 0.8
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
    })
    v = scorer.score(ok_inactive)
    check(v.is_honeypot is False, "Inactive >180d + response <0.8 does NOT fire")

    # ─── D9. interview_rate_contradiction ────────────────────────────────
    contradiction = make_candidate(redrob_signals={
        "interview_completion_rate": 0.5,    # > 0
        "offer_acceptance_rate": -1,         # no offer history
        "applications_submitted_30d": 0,     # no recent applications
        "last_active_date": "2026-06-15",
        "recruiter_response_rate": 0.7,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
    })
    v = scorer.score(contradiction)
    check(v.is_honeypot is True, "Interview rate >0 + no offer history + no apps fires")
    check("interview_rate_contradiction" in v.fired_honeypot_rules, "interview_rate_contradiction in fired list")

    # ─── D10. trap: title_skill_mismatch ─────────────────────────────────
    trap = make_candidate(
        title="Marketing Manager",
        skills=[
            {"name": s, "proficiency": "expert", "endorsements": 10, "duration_months": 24}
            for s in ["LLM", "NLP", "ML", "PyTorch", "RAG"]  # 5 AI skills
        ],
    )
    v = scorer.score(trap)
    check(v.is_honeypot is False, "Title-skill mismatch is NOT a hard honeypot")
    check(v.is_trap is True, "Title-skill mismatch IS a soft trap")
    check("title_skill_mismatch" in v.fired_trap_rules, "title_skill_mismatch in trap rules")

    # Marketing Manager with only 3 AI skills should NOT fire
    ok_marketing = make_candidate(
        title="Marketing Manager",
        skills=[
            {"name": s, "proficiency": "expert", "endorsements": 10, "duration_months": 24}
            for s in ["LLM", "NLP", "ML"]  # only 3
        ],
    )
    v = scorer.score(ok_marketing)
    check(v.is_trap is False, "Marketing Manager with 3 AI skills does NOT fire (threshold=5)")

    # ─── D11. trap: services_only_career ─────────────────────────────────
    services_c = make_candidate(
        career_history=[
            {"company": "TCS", "title": "Engineer", "start_date": "2020-01-01", "end_date": None,
             "duration_months": 60, "is_current": True, "industry": "IT Services",
             "company_size": "10001+", "description": "Engineer at TCS"},
            {"company": "Infosys", "title": "Trainee", "start_date": "2018-01-01", "end_date": "2019-12-31",
             "duration_months": 24, "is_current": False, "industry": "IT Services",
             "company_size": "10001+", "description": "Trainee at Infosys"},
        ]
    )
    v = scorer.score(services_c)
    check(v.is_trap is True, "TCS+Infosys career fires services_only_career trap")

    # Mixed career (1 product + 1 services) should NOT fire
    mixed_c = make_candidate(
        career_history=[
            {"company": "Swiggy", "title": "Engineer", "start_date": "2022-01-01", "end_date": None,
             "duration_months": 36, "is_current": True, "industry": "Food Delivery",
             "company_size": "5001-10000", "description": "Engineer at Swiggy"},
            {"company": "TCS", "title": "Trainee", "start_date": "2018-01-01", "end_date": "2021-12-31",
             "duration_months": 48, "is_current": False, "industry": "IT Services",
             "company_size": "10001+", "description": "Trainee at TCS"},
        ]
    )
    v = scorer.score(mixed_c)
    check(v.is_trap is False, "Mixed product+services career does NOT fire (exception clause works)")

    # ─── D12. trap: pure_research_background ─────────────────────────────
    research_c = make_candidate(
        career_history=[
            {"company": "Microsoft Research", "title": "Postdoc", "start_date": "2022-01-01", "end_date": None,
             "duration_months": 36, "is_current": True, "industry": "Research Lab",
             "company_size": "10001+", "description": "Postdoc"},
            {"company": "Google Research", "title": "Research Scientist", "start_date": "2019-01-01", "end_date": "2021-12-31",
             "duration_months": 36, "is_current": False, "industry": "Research Lab",
             "company_size": "10001+", "description": "Research Scientist"},
        ]
    )
    v = scorer.score(research_c)
    check(v.is_trap is True, "All-research career fires pure_research_background trap")

    # ─── D13. trap: recent_langchain_only ────────────────────────────────
    langchain_only = make_candidate(
        skills=[
            {"name": "LangChain", "proficiency": "intermediate", "endorsements": 5, "duration_months": 6},
            {"name": "RAG", "proficiency": "intermediate", "endorsements": 3, "duration_months": 4},
            {"name": "OpenAI API", "proficiency": "intermediate", "endorsements": 2, "duration_months": 8},
        ]
    )
    v = scorer.score(langchain_only)
    check(v.is_trap is True, "All AI skills <12mo + no pre-LLM ML fires recent_langchain_only")

    # With pre-LLM ML, should NOT fire
    with_pre_llm = make_candidate(
        skills=[
            {"name": "LangChain", "proficiency": "intermediate", "endorsements": 5, "duration_months": 6},
            {"name": "scikit-learn", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},  # pre-LLM
        ]
    )
    v = scorer.score(with_pre_llm)
    check(v.is_trap is False, "LangChain + pre-LLM ML (scikit-learn) does NOT fire")

    # ─── D14. trap: title_chaser ─────────────────────────────────────────
    chaser = make_candidate(
        career_history=[
            {"company": "A", "title": "Senior Engineer", "start_date": "2024-06-01", "end_date": None,
             "duration_months": 12, "is_current": True, "industry": "Software",
             "company_size": "201-500", "description": "Senior"},
            {"company": "B", "title": "Staff Engineer", "start_date": "2023-06-01", "end_date": "2024-05-31",
             "duration_months": 12, "is_current": False, "industry": "Software",
             "company_size": "201-500", "description": "Staff"},
            {"company": "C", "title": "Principal Engineer", "start_date": "2022-06-01", "end_date": "2023-05-31",
             "duration_months": 12, "is_current": False, "industry": "Software",
             "company_size": "201-500", "description": "Principal"},
        ]
    )
    v = scorer.score(chaser)
    check(v.is_trap is True, "3 short roles + Senior→Staff→Principal fires title_chaser")

    # 3 short roles but no title progression should NOT fire
    no_progression = make_candidate(
        career_history=[
            {"company": "A", "title": "Engineer", "start_date": "2024-06-01", "end_date": None,
             "duration_months": 12, "is_current": True, "industry": "Software",
             "company_size": "201-500", "description": "Engineer"},
            {"company": "B", "title": "Engineer", "start_date": "2023-06-01", "end_date": "2024-05-31",
             "duration_months": 12, "is_current": False, "industry": "Software",
             "company_size": "201-500", "description": "Engineer"},
            {"company": "C", "title": "Engineer", "start_date": "2022-06-01", "end_date": "2023-05-31",
             "duration_months": 12, "is_current": False, "industry": "Software",
             "company_size": "201-500", "description": "Engineer"},
        ]
    )
    v = scorer.score(no_progression)
    check(v.is_trap is False, "3 short roles but no Senior/Staff/Principal does NOT fire")

    # ─── D15. Rule audit trail (rule_details) ────────────────────────────
    bad = make_candidate(redrob_signals={
        "expected_salary_range_inr_lpa": {"min": 50.0, "max": 30.0},  # inverted
        "last_active_date": "2026-06-15",
        "recruiter_response_rate": 0.7,
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
    })
    v = scorer.score(bad)
    check("salary_range_inverted" in v.rule_details, "Fired rule has entry in rule_details")
    check("reason" in v.rule_details["salary_range_inverted"], "rule_details entry has 'reason' field")
    check("evidence" in v.rule_details["salary_range_inverted"], "rule_details entry has 'evidence' field")


# ═══════════════════════════════════════════════════════════════════════
# E. Config consistency
# ═══════════════════════════════════════════════════════════════════════
def test_config_consistency():
    section("E. Config consistency — code references match config keys")

    # E1. honeypot_rules.yaml: trap_penalty_value present
    rules_path = Path(__file__).parent.parent / "config" / "honeypot_rules.yaml"
    with open(rules_path) as f:
        rules_cfg = yaml.safe_load(f)
    check("trap_penalty_value" in rules_cfg, "honeypot_rules.yaml has trap_penalty_value")
    check(rules_cfg["trap_penalty_value"] == 0.3, f"trap_penalty_value is 0.3 (got {rules_cfg['trap_penalty_value']})")

    # E2. Reference lists present
    for list_name in ["services_companies", "research_titles", "ai_skill_keywords", "pre_llm_ml_keywords"]:
        check(list_name in rules_cfg, f"honeypot_rules.yaml has {list_name}")
        check(len(rules_cfg[list_name]) > 0, f"{list_name} is non-empty")

    # E3. intent_schema.json: required top-level keys
    schema_path = Path(__file__).parent.parent / "config" / "intent_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    for key in ["explicit_positive_requirements", "nice_to_haves", "hard_disqualifiers",
                "skill_taxonomy_groups", "title_archetype_match", "seniority_expectation"]:
        check(key in schema, f"intent_schema.json has {key}")

    # E4. Each positive requirement has required fields
    for req in schema["explicit_positive_requirements"]:
        check("axis" in req, f"Positive req has 'axis': {req.get('axis', 'MISSING')}")
        check("weight" in req, f"Positive req '{req.get('axis')}' has 'weight'")
        check("embedding_query_text" in req, f"Positive req '{req.get('axis')}' has 'embedding_query_text'")
        check("must_have" in req, f"Positive req '{req.get('axis')}' has 'must_have'")

    # E5. Each disqualifier has detection_hints
    for dq in schema["hard_disqualifiers"]:
        check("axis" in dq, f"Disqualifier has 'axis': {dq.get('axis', 'MISSING')}")
        check("detection_hints" in dq, f"Disqualifier '{dq.get('axis')}' has 'detection_hints'")

    # E6. skill_aliases.yaml: all canonical skills have aliases
    aliases_path = Path(__file__).parent.parent / "config" / "skill_aliases.yaml"
    with open(aliases_path) as f:
        aliases = yaml.safe_load(f)
    canonical = aliases.get("canonical_skills", {})
    for name, entry in canonical.items():
        check("aliases" in entry, f"Canonical skill '{name}' has 'aliases' list")
        check(len(entry["aliases"]) > 0, f"Canonical skill '{name}' has non-empty aliases")

    # E7. field_weights.yaml: weights present
    fw_path = Path(__file__).parent.parent / "config" / "field_weights.yaml"
    with open(fw_path) as f:
        fw = yaml.safe_load(f)
    check("combined_embedding_weights" in fw, "field_weights.yaml has combined_embedding_weights")
    check("bm25_field_weights" in fw, "field_weights.yaml has bm25_field_weights")
    check("semantic_axis_weights" in fw, "field_weights.yaml has semantic_axis_weights")
    check("rrf_k" in fw, "field_weights.yaml has rrf_k")
    # axis weights must sum to 1.0
    axis_sum = sum(fw["semantic_axis_weights"].values())
    check(abs(axis_sum - 1.0) < 1e-6, f"semantic_axis_weights sum to 1.0 (got {axis_sum})")

    # E8. combiner_weights.yaml: base weights sum to 1.0
    cw_path = Path(__file__).parent.parent / "config" / "combiner_weights.yaml"
    with open(cw_path) as f:
        cw = yaml.safe_load(f)
    check("base_weights" in cw, "combiner_weights.yaml has base_weights")
    base_sum = sum(cw["base_weights"].values())
    check(abs(base_sum - 1.0) < 1e-6, f"base_weights sum to 1.0 (got {base_sum})")
    check("honeypot_penalty" in cw, "combiner_weights.yaml has honeypot_penalty")
    check("trap_penalty" in cw, "combiner_weights.yaml has trap_penalty")
    check("availability_mult" in cw, "combiner_weights.yaml has availability_mult")
    check(cw["availability_mult"]["min"] == 0.30, "availability_mult.min == 0.30")
    check(cw["availability_mult"]["max"] == 1.20, "availability_mult.max == 1.20")


# ═══════════════════════════════════════════════════════════════════════
# F. End-to-end smoke test on dev set
# ═══════════════════════════════════════════════════════════════════════
def test_e2e_smoke():
    section("F. End-to-end smoke test on 50 dev candidates")

    # F1. Load dev set
    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        records = json.load(f)
    candidates = [Candidate.from_dict(r) for r in records]
    check(len(candidates) == 50, f"Loaded 50 dev candidates (got {len(candidates)})")

    # F2. Score all with HoneypotScorer — no crashes
    scorer = HoneypotScorer.from_config()
    verdicts = {}
    for c in candidates:
        v = scorer.score(c)
        verdicts[c.candidate_id] = v
        check(isinstance(v, FilterVerdict), f"{c.candidate_id}: score() returns FilterVerdict")
        check(0.0 <= v.honeypot_penalty <= 1.0, f"{c.candidate_id}: honeypot_penalty in [0,1]")
        check(0.0 <= v.trap_penalty <= 1.0, f"{c.candidate_id}: trap_penalty in [0,1]")
    check(len(verdicts) == 50, f"All 50 candidates scored (got {len(verdicts)})")

    # F3. Honeypot count sanity (we know from prior run: 18 honeypots expected)
    honeypot_count = sum(1 for v in verdicts.values() if v.is_honeypot)
    check(honeypot_count > 0, f"At least 1 honeypot detected (got {honeypot_count})")
    check(honeypot_count < 30, f"Less than 30 honeypots (got {honeypot_count}) — not over-aggressive")

    # F4. Trap count sanity
    trap_count = sum(1 for v in verdicts.values() if v.is_trap)
    check(trap_count > 0, f"At least 1 trap detected (got {trap_count})")
    check(trap_count < 20, f"Less than 20 traps (got {trap_count}) — not over-aggressive")

    # F5. Load embeddings (if exist) and check shape
    emb_path = Path(__file__).parent.parent / "artifacts" / "dev_embeddings.npz"
    if emb_path.exists():
        embeddings = load_embeddings(emb_path)
        check(len(embeddings) == 50, f"Loaded 50 candidate embeddings (got {len(embeddings)})")
        sample_cid = list(embeddings.keys())[0]
        # All fields must have the SAME dim per candidate
        dims = {field: embeddings[sample_cid][field].shape[0] for field in MultiFieldEncoder.FIELDS}
        unique_dims = set(dims.values())
        check(len(unique_dims) == 1, f"All 5 fields have same dim per candidate (dims: {dims})")
        # All candidates must have same dim
        all_dims = {cid: embeddings[cid]["combined"].shape[0] for cid in embeddings}
        unique_all = set(all_dims.values())
        check(len(unique_all) == 1, f"All candidates have same embedding dim (got {unique_all})")
        # Each candidate's combined embedding should be L2-normalized (or zero)
        sample_combined_norm = float(np.linalg.norm(embeddings[sample_cid]["combined"]))
        check(0.99 <= sample_combined_norm <= 1.01 or sample_combined_norm == 0.0,
              f"Combined embedding is L2-normalized or zero (norm={sample_combined_norm:.4f})")
    else:
        check(True, "No dev embeddings yet (run precompute/02_encode_candidates.py first)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("  DeepFit Audit Suite — Modules 0-3")
    print("="*70)

    tests = [
        ("types.py", test_types),
        ("intent.py", test_intent),
        ("encoder.py", test_encoder),
        ("filters.py", test_filters),
        ("config consistency", test_config_consistency),
        ("e2e smoke", test_e2e_smoke),
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
    print(f"  AUDIT SUMMARY")
    print("="*70)
    print(f"  Passed: {PASS}")
    print(f"  Failed: {FAIL}")
    if ERRORS:
        print(f"\n  Failures:")
        for e in ERRORS:
            print(f"    - {e}")
    print()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
