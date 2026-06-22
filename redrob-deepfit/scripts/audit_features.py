#!/usr/bin/env python3
"""
Audit Modules 4, 5, 6 (features, availability) — comprehensive edge-case tests.

Tests:
  G. features.py — production evidence (regex patterns, edge cases)
  H. features.py — skill taxonomy resolver (ESCO + AI aliases)
  I. features.py — structural score (YOE, location, education, work_mode)
  J. features.py — seniority match (title level, archetype boost)
  K. availability.py — multiplicative [0.30, 1.20] scorer
  L. integration — all features on 50 dev candidates
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

from ranker.types import Candidate
from ranker.features import (
    extract_production_evidence, SkillTaxonomyResolver,
    score_location, score_yoe_band, score_education_tier, score_work_mode,
    compute_structural_score, score_seniority, extract_all_features,
    PRODUCTION_PATTERNS, RESEARCH_PATTERNS,
)
from ranker.availability import AvailabilityScorer, score_availability
from ranker.intent import get_jd_required_canonical_skills, load_intent_schema

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

def make_candidate(**kwargs):
    """Helper to build a candidate from kwargs."""
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
             "description": kwargs.get("career_desc", "Built and shipped retrieval systems in production.")}
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


# ═══════════════════════════════════════════════════════════════════════
# G. Production-Evidence Extractor
# ═══════════════════════════════════════════════════════════════════════
def test_production_evidence():
    section("G. Production-Evidence Extractor")

    # G1. Production-dominant description
    prod_c = make_candidate(career_desc=(
        "Built and shipped recommendation system to production serving 10M daily users. "
        "Set up monitoring, alerting, and SLO tracking. Ran A/B tests to validate "
        "ranking quality improvements. Migrated from keyword search to dense retrieval."
    ))
    result = extract_production_evidence(prod_c)
    check(result["has_production"] is True, "Production text detected as has_production=True")
    check(result["production_hits"] >= 3, f"Multiple production signals found (got {result['production_hits']})")
    check(result["score"] > 0.5, f"Production-dominant score > 0.5 (got {result['score']:.3f})")

    # G2. Research-only description
    research_c = make_candidate(career_desc=(
        "Published 3 papers at Neurips and ICML on novel attention mechanisms. "
        "Worked in academic research lab, no production deployment. Thesis on "
        "theoretical foundations of deep learning."
    ))
    result = extract_production_evidence(research_c)
    check(result["has_research"] is True, "Research text detected as has_research=True")
    check(result["research_hits"] >= 3, f"Multiple research signals found (got {result['research_hits']})")
    check(result["score"] < 0.3, f"Research-only score < 0.3 (got {result['score']:.3f})")

    # G3. Empty career text
    empty_c = make_candidate(career_history=[])
    result = extract_production_evidence(empty_c)
    check(result["score"] == 0.0, "Empty career text → score 0.0")
    check(result["production_hits"] == 0, "Empty career text → 0 production hits")

    # G4. Mixed production + research (should still favor production if dominant)
    # Test text has 3 production + 3 research — verify both detected, and score is moderate
    mixed_c = make_candidate(career_desc=(
        "Shipped recommendation system to production. Published 1 paper at ICML. "
        "Built A/B test framework. Served 5M users."
    ))
    result = extract_production_evidence(mixed_c)
    check(result["has_production"] is True, "Mixed has production")
    check(result["has_research"] is True, "Mixed has research")
    check(result["production_hits"] >= 2,
          f"Mixed has at least 2 production hits (got {result['production_hits']})")
    check(0.3 <= result["score"] <= 0.8,
          f"Mixed score is moderate [0.3, 0.8] (got {result['score']:.3f})")

    # G5. Real dev candidate — CAND_0000031 (Recommendation Systems Engineer @ Swiggy)
    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        samples = json.load(f)
    cand_31 = Candidate.from_dict(samples[30])
    result = extract_production_evidence(cand_31)
    check(result["production_hits"] >= 2,
          f"CAND_0000031 (Swiggy recsys) has production signals (got {result['production_hits']})")
    check("shipped" in " ".join(result["production_phrases"]).lower() or
          "production" in " ".join(result["production_phrases"]).lower() or
          result["production_hits"] >= 2,
          f"CAND_0000031 production phrases include shipped/production: {result['production_phrases']}")

    # G6. Pattern coverage — verify key JD-aligned phrases are in PRODUCTION_PATTERNS
    pattern_text = " ".join(PRODUCTION_PATTERNS).lower()
    for key_phrase in ["production", "shipped", "a/b", "monitoring", "vector search",
                       "dense retrieval", "ndcg", "retrieval"]:
        check(key_phrase in pattern_text, f"PRODUCTION_PATTERNS includes '{key_phrase}'")

    research_text = " ".join(RESEARCH_PATTERNS).lower()
    for key_phrase in ["paper", "neurips", "thesis", "arxiv", "research"]:
        check(key_phrase in research_text, f"RESEARCH_PATTERNS includes '{key_phrase}'")


# ═══════════════════════════════════════════════════════════════════════
# H. Skill Taxonomy Resolver
# ═══════════════════════════════════════════════════════════════════════
def test_skill_taxonomy():
    section("H. Skill Taxonomy Resolver (ESCO + AI aliases)")

    resolver = SkillTaxonomyResolver.from_config()
    schema = load_intent_schema()
    jd_required = get_jd_required_canonical_skills(schema)
    check(len(jd_required) > 0, f"JD-required skills non-empty (got {len(jd_required)})")

    # H1. Candidate with all modern AI terminology (ESCO misses these, our aliases catch them)
    modern_c = make_candidate(
        skills=[
            {"name": "LangChain", "proficiency": "advanced", "endorsements": 10, "duration_months": 12},
            {"name": "Pinecone", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 20, "duration_months": 36},
            {"name": "LoRA", "proficiency": "intermediate", "endorsements": 5, "duration_months": 8},
            {"name": "vLLM", "proficiency": "intermediate", "endorsements": 3, "duration_months": 6},
        ],
        career_desc="Built RAG system with LangChain and Pinecone. Fine-tuned Mistral with QLoRA. Served via vLLM.",
    )
    matched = resolver.resolve(modern_c)
    check("Vector Database" in matched, f"Vector Database matched (Pinecone/FAISS) — matched: {matched}")
    check("LLM Fine-tuning" in matched, f"LLM Fine-tuning matched (LoRA/QLoRA) — matched: {matched}")
    check("LLM Serving" in matched, f"LLM Serving matched (vLLM) — matched: {matched}")
    check("Retrieval Augmented Generation" in matched, f"RAG matched — matched: {matched}")
    check("LLM Orchestration" in matched, f"LangChain matched as LLM Orchestration — matched: {matched}")

    # H2. Coverage score
    score, breakdown = resolver.coverage_score(matched, jd_required)
    check(0.0 <= score <= 1.0, f"Coverage score in [0,1] (got {score:.3f})")
    check("covered" in breakdown, "Breakdown has 'covered' list")
    check("missing" in breakdown, "Breakdown has 'missing' list")

    # H3. Empty candidate — should return empty set, score 0
    empty_c = make_candidate(skills=[], career_history=[])
    matched_empty = resolver.resolve(empty_c)
    check(len(matched_empty) == 0, f"Empty candidate → empty matched set (got {matched_empty})")
    score_empty, _ = resolver.coverage_score(matched_empty, jd_required)
    check(score_empty == 0.0, f"Empty candidate → coverage score 0 (got {score_empty})")

    # H4. Substring matching — "Pinecone" should match "Pinecone vector db"
    pinecone_c = make_candidate(
        skills=[{"name": "Pinecone vector db", "proficiency": "advanced", "endorsements": 5, "duration_months": 12}],
        career_history=[],
    )
    matched_pinecone = resolver.resolve(pinecone_c)
    check("Vector Database" in matched_pinecone,
          f"'Pinecone vector db' substring-matched to Vector Database (got {matched_pinecone})")

    # H5. Career description only — catches implicit mentions
    implicit_c = make_candidate(
        skills=[{"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60}],
        career_desc="Built RAG system with LangChain. Deployed FAISS for vector search.",
    )
    matched_implicit = resolver.resolve(implicit_c)
    check("Retrieval Augmented Generation" in matched_implicit,
          f"RAG detected from career description (got {matched_implicit})")
    check("Vector Database" in matched_implicit,
          f"FAISS detected from career description (got {matched_implicit})")


# ═══════════════════════════════════════════════════════════════════════
# I. Structural Score (YOE, location, education, work_mode)
# ═══════════════════════════════════════════════════════════════════════
def test_structural():
    section("I. Structural Score")

    # I1. YOE band — ideal 5-9
    ideal_c = make_candidate(yoe=6.5)
    score, label = score_yoe_band(ideal_c)
    check(score == 1.0 and label == "in_band", f"YOE 6.5 in band → score 1.0 (got {score}, {label})")

    below_c = make_candidate(yoe=2.0)
    score, label = score_yoe_band(below_c)
    check(score < 1.0 and "below" in label, f"YOE 2.0 below band → score < 1.0 (got {score:.3f}, {label})")

    above_c = make_candidate(yoe=15.0)
    score, label = score_yoe_band(above_c)
    check(score < 1.0 and "above" in label, f"YOE 15.0 above band → score < 1.0 (got {score:.3f}, {label})")
    check(score >= 0.30, f"YOE decay floored at 0.30 (got {score:.3f})")

    # I2. Location tiers
    tier1_c = make_candidate(location="Pune, Maharashtra", country="India")
    score, label = score_location(tier1_c)
    check(score == 1.0 and label == "tier_1", f"Pune → tier_1, score 1.0 (got {score}, {label})")

    tier1_c2 = make_candidate(location="Bangalore, Karnataka", country="India")
    score, label = score_location(tier1_c2)
    check(score == 1.0 and label == "tier_1", f"Bangalore → tier_1 (got {score}, {label})")

    tier2_c = make_candidate(location="Chennai, Tamil Nadu", country="India")
    score, label = score_location(tier2_c)
    check(score == 0.85 and label == "tier_2", f"Chennai → tier_2, score 0.85 (got {score}, {label})")

    outside_c = make_candidate(location="San Francisco, CA", country="USA")
    score, label = score_location(outside_c)
    check(score == 0.40 and label == "outside_india", f"San Francisco → outside_india, score 0.40 (got {score}, {label})")

    # I3. Education tiers
    tier1_edu = make_candidate(education=[
        {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2014, "end_year": 2018, "tier": "tier_1"}
    ])
    score, label = score_education_tier(tier1_edu)
    check(score == 1.0 and label == "tier_1", f"tier_1 education → score 1.0 (got {score}, {label})")

    tier3_edu = make_candidate(education=[
        {"institution": "Local College", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2014, "end_year": 2018, "tier": "tier_3"}
    ])
    score, label = score_education_tier(tier3_edu)
    check(score == 0.70 and label == "tier_3", f"tier_3 education → score 0.70 (got {score}, {label})")

    no_edu = make_candidate(education=[])
    score, label = score_education_tier(no_edu)
    check(score == 0.60, f"No education → default 0.60 (got {score}, {label})")

    # I4. Work mode
    hybrid_c = make_candidate(redrob_signals={
        "preferred_work_mode": "hybrid",
        "last_active_date": "2026-06-15",
        "recruiter_response_rate": 0.7,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
    })
    score, label = score_work_mode(hybrid_c)
    check(score == 1.0 and label == "hybrid", f"Hybrid work mode → 1.0 (got {score}, {label})")

    # I5. Combined structural score
    structural_score, breakdown = compute_structural_score(ideal_c)
    check(0.0 <= structural_score <= 1.0, f"Structural score in [0,1] (got {structural_score:.3f})")
    check("yoe_band" in breakdown, "Structural breakdown has yoe_band")
    check("location" in breakdown, "Structural breakdown has location")
    check("education" in breakdown, "Structural breakdown has education")
    check("work_mode" in breakdown, "Structural breakdown has work_mode")
    # Ideal candidate should have high structural score
    check(structural_score >= 0.85, f"Ideal candidate structural score >= 0.85 (got {structural_score:.3f})")


# ═══════════════════════════════════════════════════════════════════════
# J. Seniority Match
# ═══════════════════════════════════════════════════════════════════════
def test_seniority():
    section("J. Seniority Match")

    # J1. Senior title
    senior_c = make_candidate(title="Senior ML Engineer")
    score, breakdown = score_seniority(senior_c)
    check(breakdown["level_label"] == "senior", f"Senior ML Engineer → level 'senior' (got {breakdown['level_label']})")
    check(breakdown["level_score"] == 1.0, f"Senior level_score = 1.0 (got {breakdown['level_score']})")

    # J2. Mid-level title (no prefix)
    mid_c = make_candidate(title="ML Engineer")
    score, breakdown = score_seniority(mid_c)
    check(breakdown["level_label"] == "mid", f"ML Engineer → level 'mid' (got {breakdown['level_label']})")
    check(breakdown["level_score"] == 0.85, f"Mid level_score = 0.85 (got {breakdown['level_score']})")

    # J3. Junior title
    junior_c = make_candidate(title="Junior Engineer")
    score, breakdown = score_seniority(junior_c)
    check(breakdown["level_label"] == "junior", f"Junior Engineer → level 'junior' (got {breakdown['level_label']})")
    check(breakdown["level_score"] == 0.40, f"Junior level_score = 0.40 (got {breakdown['level_score']})")

    # J4. Too-senior non-coding title (Manager/Director/Architect)
    arch_c = make_candidate(title="Solutions Architect")
    score, breakdown = score_seniority(arch_c)
    check(breakdown["level_label"] == "too_senior_noncoding",
          f"Solutions Architect → 'too_senior_noncoding' (got {breakdown['level_label']})")
    check(breakdown["level_score"] == 0.50, f"Architect level_score = 0.50 (got {breakdown['level_score']})")

    # J5. With cosine archetype boost
    senior_boosted, _ = score_seniority(senior_c, cosine_title_sim=0.85)
    senior_unboosted, _ = score_seniority(senior_c, cosine_title_sim=None)
    check(senior_boosted >= senior_unboosted,
          f"Senior with strong archetype cosine >= no cosine (boosted={senior_boosted:.3f}, unboosted={senior_unboosted:.3f})")

    # J6. Weak archetype match pulls score down
    weak_arch, breakdown = score_seniority(senior_c, cosine_title_sim=0.20)
    check(breakdown["archetype_label"] == "weak_match",
          f"Weak cosine (0.20) → 'weak_match' (got {breakdown['archetype_label']})")


# ═══════════════════════════════════════════════════════════════════════
# K. Availability Multiplier
# ═══════════════════════════════════════════════════════════════════════
def test_availability():
    section("K. Availability Multiplier")

    scorer = AvailabilityScorer.from_config()

    # K1. Ideal candidate — active today, high response, open_to_work, 30d notice, github, recruiter demand
    ideal_c = make_candidate(redrob_signals={
        "last_active_date": "2026-06-19",  # 1 day ago
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.95,
        "notice_period_days": 30,
        "github_activity_score": 80,
        "saved_by_recruiters_30d": 25,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
    })
    mult, breakdown = scorer.score(ideal_c)
    check(mult > 1.0, f"Ideal candidate mult > 1.0 (got {mult:.4f})")
    check(mult <= 1.20, f"Mult capped at max 1.20 (got {mult:.4f})")
    check(breakdown["recency"]["value"] >= 0.95, f"Recency near 1.0 for active-today candidate (got {breakdown['recency']['value']:.4f})")
    check(breakdown["response_rate"]["value"] > 0.99, f"Response rate near 1.0 for rr=0.95 (got {breakdown['response_rate']['value']:.4f})")
    check(breakdown["notice_period"]["value"] == 1.0, f"Notice 30d → multiplier 1.0 (got {breakdown['notice_period']['value']})")
    check(breakdown["github_activity"]["value"] >= 1.10, f"GitHub boost maxed at 1.10+ (got {breakdown['github_activity']['value']:.4f})")

    # K2. Dead candidate — 200d inactive, low response, not otw, 120d notice, no github
    dead_c = make_candidate(redrob_signals={
        "last_active_date": "2025-12-01",  # ~200 days ago
        "open_to_work_flag": False,
        "recruiter_response_rate": 0.05,
        "notice_period_days": 120,
        "github_activity_score": -1,
        "saved_by_recruiters_30d": 0,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
    })
    mult, breakdown = scorer.score(dead_c)
    check(mult == 0.30, f"Dead candidate mult floored at 0.30 (got {mult:.4f})")
    check(breakdown["clipped"] is True, f"Dead candidate mult was clipped (got raw_mult={breakdown['raw_mult']:.4f})")
    check(breakdown["recency"]["value"] <= 0.15, f"Recency for 200d inactive floored at min (got {breakdown['recency']['value']:.4f})")
    check(breakdown["notice_period"]["value"] == 0.70, f"Notice 120d → multiplier 0.70 (got {breakdown['notice_period']['value']})")

    # K3. Missing last_active_date — sentinel handling
    missing_date_c = make_candidate(redrob_signals={
        "last_active_date": "",  # missing
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.7,
        "notice_period_days": 30,
        "github_activity_score": 40,
        "saved_by_recruiters_30d": 10,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "applications_submitted_30d": 5,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": 0.5,
    })
    mult, breakdown = scorer.score(missing_date_c)
    check(breakdown["recency"]["days_inactive"] == -1, f"Missing date → days_inactive=-1 (got {breakdown['recency']['days_inactive']})")
    check(breakdown["recency"]["value"] <= 0.10, f"Missing date → recency floored at min (got {breakdown['recency']['value']:.4f})")

    # K4. Notice period tiers
    notice_tiers = [(30, 1.00), (45, 0.95), (75, 0.85), (120, 0.70), (200, 0.50)]
    for days, expected_mult in notice_tiers:
        c = make_candidate(redrob_signals={
            "last_active_date": "2026-06-15",
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.5,
            "notice_period_days": days,
            "github_activity_score": -1,
            "saved_by_recruiters_30d": 0,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "applications_submitted_30d": 5,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": 0.5,
        })
        _, bd = scorer.score(c)
        check(abs(bd["notice_period"]["value"] - expected_mult) < 0.001,
              f"Notice {days}d → multiplier {expected_mult} (got {bd['notice_period']['value']})")

    # K5. Range validation — all outputs in [0.30, 1.20]
    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        samples = json.load(f)
    for s in samples:
        c = Candidate.from_dict(s)
        mult, _ = scorer.score(c)
        check(0.30 <= mult <= 1.20, f"{c.candidate_id}: mult in [0.30, 1.20] (got {mult:.4f})")


# ═══════════════════════════════════════════════════════════════════════
# L. Integration test — all features on dev candidates
# ═══════════════════════════════════════════════════════════════════════
def test_integration():
    section("L. Integration — all features on 50 dev candidates")

    with open(Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json") as f:
        samples = json.load(f)

    resolver = SkillTaxonomyResolver.from_config()
    schema = load_intent_schema()
    jd_required = get_jd_required_canonical_skills(schema)
    avail_scorer = AvailabilityScorer.from_config()

    prod_scores = []
    esco_scores = []
    structural_scores = []
    seniority_scores = []
    avail_mults = []

    for s in samples:
        c = Candidate.from_dict(s)
        features = extract_all_features(c, skill_resolver=resolver, jd_required_skills=jd_required)
        avail_mult, _ = avail_scorer.score(c)

        # Validate all scores in [0, 1] (except availability which is [0.30, 1.20])
        check(0.0 <= features["production_evidence"] <= 1.0,
              f"{c.candidate_id}: production_evidence in [0,1] (got {features['production_evidence']:.3f})")
        check(0.0 <= features["esco_coverage"] <= 1.0,
              f"{c.candidate_id}: esco_coverage in [0,1] (got {features['esco_coverage']:.3f})")
        check(0.0 <= features["structural_score"] <= 1.0,
              f"{c.candidate_id}: structural_score in [0,1] (got {features['structural_score']:.3f})")
        check(0.0 <= features["seniority_match"] <= 1.0,
              f"{c.candidate_id}: seniority_match in [0,1] (got {features['seniority_match']:.3f})")
        check(0.30 <= avail_mult <= 1.20,
              f"{c.candidate_id}: availability_mult in [0.30, 1.20] (got {avail_mult:.4f})")

        prod_scores.append(features["production_evidence"])
        esco_scores.append(features["esco_coverage"])
        structural_scores.append(features["structural_score"])
        seniority_scores.append(features["seniority_match"])
        avail_mults.append(avail_mult)

    # Print summary statistics
    print(f"\n  Score distributions across 50 dev candidates:")
    print(f"    production_evidence: mean={np.mean(prod_scores):.3f}, "
          f"min={np.min(prod_scores):.3f}, max={np.max(prod_scores):.3f}")
    print(f"    esco_coverage:       mean={np.mean(esco_scores):.3f}, "
          f"min={np.min(esco_scores):.3f}, max={np.max(esco_scores):.3f}")
    print(f"    structural_score:    mean={np.mean(structural_scores):.3f}, "
          f"min={np.min(structural_scores):.3f}, max={np.max(structural_scores):.3f}")
    print(f"    seniority_match:     mean={np.mean(seniority_scores):.3f}, "
          f"min={np.min(seniority_scores):.3f}, max={np.max(seniority_scores):.3f}")
    print(f"    availability_mult:   mean={np.mean(avail_mults):.3f}, "
          f"min={np.min(avail_mults):.3f}, max={np.max(avail_mults):.3f}")

    # Sanity: scores should have variance (not all the same)
    check(np.std(prod_scores) > 0.01, f"production_evidence has variance (std={np.std(prod_scores):.3f})")
    check(np.std(esco_scores) > 0.01, f"esco_coverage has variance (std={np.std(esco_scores):.3f})")
    check(np.std(structural_scores) > 0.01, f"structural_score has variance (std={np.std(structural_scores):.3f})")
    check(np.std(seniority_scores) > 0.01, f"seniority_match has variance (std={np.std(seniority_scores):.3f})")
    check(np.std(avail_mults) > 0.01, f"availability_mult has variance (std={np.std(avail_mults):.3f})")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("  DeepFit Audit Suite — Modules 4, 5, 6 (features, availability)")
    print("="*70)

    tests = [
        ("production_evidence", test_production_evidence),
        ("skill_taxonomy", test_skill_taxonomy),
        ("structural", test_structural),
        ("seniority", test_seniority),
        ("availability", test_availability),
        ("integration", test_integration),
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
        for e in ERRORS[:20]:
            print(f"    - {e}")
    print()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
