"""
Module 4: Production-Evidence Extractor
============================================================================

Regex-mines career_history[].description to distinguish candidates who have
SHIPPED ML systems to real users from those whose work is research-only.

The JD explicitly disqualifies "pure research environments ... without any
production deployment." A candidate whose career_history mentions only papers,
posters, benchmarks should be deprioritized EVEN IF their semantic match is high.

Output: production_evidence_score ∈ [0, 1] per candidate, fed into the combiner.

Also includes:
    - Module 6.5: Skill Taxonomy Resolver (ESCO + custom AI aliases)
    - Module 6 structural: YOE band, location tier, education tier, work_mode
    - Module 6 seniority: title level match

All these "feature" computations live in this single module for simplicity.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from .types import Candidate

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Production-Evidence Extractor (Module 4)
# ═══════════════════════════════════════════════════════════════════════

# ─── Production signals (positive evidence) ───────────────────────────────
# Matched case-insensitive against concatenated career_history descriptions.
PRODUCTION_PATTERNS = [
    # Deployment / shipping
    r"deployed (?:to )?production",
    r"shipped",
    r"rollout",
    r"rolled out",
    r"launched",
    r"productionized",
    r"productionised",
    r"production deployment",
    r"deployed (?:at|on|in) (?:real users|production|live)",
    # Scale signals
    r"served [\d,]+ (?:users|requests|qps|queries|customers|transactions|impressions)",
    r"[\d,]+ (?:daily|monthly|weekly|active) (?:users|requests|queries|customers)",
    r"scale of [\d,]+",
    r"millions? of (?:users|requests|queries|documents|rows)",
    r"billions? of (?:requests|queries|rows|events)",
    r"high (?:traffic|throughput|volume)",
    # Live / real traffic
    r"production traffic",
    r"live traffic",
    r"real users",
    r"real-world",
    r"in the wild",
    # A/B testing
    r"a/b test(?:ing)?",
    r"ab test(?:ing)?",
    r"experimentation platform",
    r"controlled experiment",
    r"online experiment",
    # Monitoring / SLO
    r"monitoring",
    r"alerting",
    r"observability",
    r"\bslo\b",
    r"\bsla\b",
    r"on-call",
    r"on call",
    r"incident response",
    # Migrations / rollouts
    r"migration",
    r"migrated",
    r"canary",
    r"blue-green",
    r"progressive rollout",
    # Evaluation / ranking quality (relevant to this JD)
    r"eval framework",
    r"evaluation framework",
    r"offline (?:eval|metric|evaluation)",
    r"online (?:eval|metric|evaluation)",
    r"offline-to-online",
    r"ndcg",
    r"\bmrr\b",
    r"\bmap\b",  # mean average precision (careful: also means many things)
    r"precision@k",
    r"recall@k",
    r"relevance judgment",
    # Retrieval / ranking system signals
    r"vector search",
    r"dense retrieval",
    r"hybrid retrieval",
    r"embedding drift",
    r"retrieval quality",
    r"recall regression",
    r"ranking model",
    r"retrieval system",
    r"recommendation system",
    r"search system",
    # Code quality / engineering
    r"code review",
    r"\bci/cd\b",
    r"unit test",
    r"integration test",
    r"code quality",
]

# ─── Research signals (negative evidence) ─────────────────────────────────
RESEARCH_PATTERNS = [
    r"\bpaper\b",
    r"publication",
    r"published",
    r"arxiv",
    r"neurips",
    r"\bicml\b",
    r"\bcvpr\b",
    r"\bacl\b",
    r"\bemnlp\b",
    r"\biclr\b",
    r"research (?:only|lab|position|role)",
    r"academia",
    r"academic",
    r"thesis",
    r"dissertation",
    r"postdoc",
    r"postdoctoral",
    r"benchmarks? only",
    r"(?:no|without) production",
    r"theoretical",
    r"proof of concept",  # weak signal — only counts as mild negative
]

# Pre-compile patterns for efficiency (called once per candidate)
_PROD_REGEX = [re.compile(p, re.IGNORECASE) for p in PRODUCTION_PATTERNS]
_RESEARCH_REGEX = [re.compile(p, re.IGNORECASE) for p in RESEARCH_PATTERNS]


def extract_production_evidence(candidate: Candidate) -> dict:
    """
    Regex-mine career_history descriptions for production vs research signals.

    Returns:
        {
            'score': float in [0, 1],
            'production_hits': int (raw count of production signal matches),
            'research_hits': int (raw count of research signal matches),
            'production_phrases': list[str] (matched phrases, for reasoning),
            'research_phrases': list[str] (matched phrases, for reasoning),
            'has_production': bool,
            'has_research': bool,
        }
    """
    career_text = candidate.career_text()
    if not career_text.strip():
        return {
            "score": 0.0,
            "production_hits": 0,
            "research_hits": 0,
            "production_phrases": [],
            "research_phrases": [],
            "has_production": False,
            "has_research": False,
        }

    # Count production signals
    production_phrases = []
    for regex in _PROD_REGEX:
        matches = regex.findall(career_text)
        if matches:
            # De-duplicate (same phrase may match multiple times)
            unique_matches = list(set(matches))[:3]  # cap at 3 per pattern
            production_phrases.extend(unique_matches)
    production_hits = len(production_phrases)

    # Count research signals
    research_phrases = []
    for regex in _RESEARCH_REGEX:
        matches = regex.findall(career_text)
        if matches:
            unique_matches = list(set(matches))[:2]
            research_phrases.extend(unique_matches)
    research_hits = len(research_phrases)

    # Score: sigmoid of (production - research) signal count
    # Each production hit adds ~0.15, each research hit subtracts ~0.10
    # Capped at [0, 1]
    raw_score = 0.5 + (production_hits * 0.15) - (research_hits * 0.10)
    # Sigmoid-style saturation
    score = 1.0 / (1.0 + math.exp(-3.0 * (raw_score - 0.5)))
    # Boost if clearly production-dominant (3+ production hits and 0 research)
    if production_hits >= 3 and research_hits == 0:
        score = min(1.0, score + 0.10)
    # Heavy penalty if research-only with no production signal
    if research_hits >= 3 and production_hits == 0:
        score = max(0.0, score - 0.30)

    return {
        "score": float(max(0.0, min(1.0, score))),
        "production_hits": production_hits,
        "research_hits": research_hits,
        "production_phrases": production_phrases[:5],  # cap for reasoning
        "research_phrases": research_phrases[:3],
        "has_production": production_hits > 0,
        "has_research": research_hits > 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Skill Taxonomy Resolver (Module 6.5)
# ═══════════════════════════════════════════════════════════════════════

class SkillTaxonomyResolver:
    """
    Resolves informal AI terminology (RAG, LoRA, vLLM, Pinecone) to canonical
    skill names using config/skill_aliases.yaml. Used to compute ESCO coverage
    against JD-required skills.
    """

    def __init__(self, aliases: dict):
        # aliases: {canonical_name: {esco_uri, jd_axis, jd_must_have, aliases: [list]}}
        self.aliases = aliases
        # Pre-build lowercase alias → canonical mapping for O(1) substring check
        self._alias_lookup = {}  # {canonical_skill: [lowercase_aliases]}
        for canonical, entry in aliases.items():
            alias_list = entry.get("aliases", [])
            self._alias_lookup[canonical] = [a.lower() for a in alias_list]

    @classmethod
    def from_config(cls, path: Optional[Path] = None) -> "SkillTaxonomyResolver":
        if path is None:
            path = Path(__file__).parent.parent / "config" / "skill_aliases.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data.get("canonical_skills", {}))

    def resolve(self, candidate: Candidate) -> set[str]:
        """
        Return the set of canonical skills the candidate has evidence of.

        Sources (combined into a single text blob for substring search):
            - candidate.skills[].name (curated, high confidence)
            - career_history[].description (catches implicit mentions like
              "Built RAG with LangChain" that may not appear in skills list)

        Match logic: case-insensitive substring (e.g., "pinecone" matches
        "Pinecone", "Pinecone vector db", "pinecone-index").
        """
        # Build text blob: skills (high confidence) + career descriptions (lower confidence)
        text_parts = [s.get("name", "") for s in candidate.skills]
        text_parts.extend(j.get("description", "") for j in candidate.career_history)
        text_blob = " ".join(text_parts)
        text_lower = text_blob.lower()

        matched = set()
        for canonical, alias_list in self._alias_lookup.items():
            for alias in alias_list:
                if alias in text_lower:
                    matched.add(canonical)
                    break  # don't double-count
        return matched

    def coverage_score(
        self,
        candidate_canonical_skills: set[str],
        jd_required_skills: list[str],
    ) -> tuple[float, dict]:
        """
        Compute coverage of JD-required canonical skills.

        Returns:
            (score in [0, 1], breakdown dict)
        """
        if not jd_required_skills:
            return 0.0, {"covered": [], "missing": []}

        covered = [s for s in jd_required_skills if s in candidate_canonical_skills]
        missing = [s for s in jd_required_skills if s not in candidate_canonical_skills]
        score = len(covered) / len(jd_required_skills)
        return float(score), {"covered": covered, "missing": missing}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: Structural Score (Module 6 — structural sub-component)
# ═══════════════════════════════════════════════════════════════════════

# ─── Location tiers (per JD) ─────────────────────────────────────────────
TIER_1_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi ncr", "bangalore", "bengaluru",
    "gurgaon", "faridabad", "new delhi",
]
TIER_2_LOCATIONS = [
    "chennai", "kolkata", "ahmedabad", "jaipur", "indore", "bhubaneswar",
    "kochi", "trivandrum", "thiruvananthapuram", "coimbatore", "visakhapatnam",
    "vizag", "chandigarh",
]


def score_location(candidate: Candidate) -> tuple[float, str]:
    """
    Score location against JD preferences.
    Returns (score in [0, 1], tier_label).
    """
    loc = (candidate.location or "").lower()
    country = (candidate.country or "").lower()

    if any(t in loc for t in TIER_1_LOCATIONS):
        return 1.00, "tier_1"
    if any(t in loc for t in TIER_2_LOCATIONS):
        return 0.85, "tier_2"
    if "india" in country:
        return 0.70, "tier_3_india"
    return 0.40, "outside_india"


def score_yoe_band(candidate: Candidate, ideal_min: float = 5.0, ideal_max: float = 9.0) -> tuple[float, str]:
    """
    Score YOE against JD's ideal 5-9 band.
    - Within band: 1.0
    - Outside band: decay 0.85 per year outside, floored at 0.30
    """
    yoe = candidate.yoe
    if ideal_min <= yoe <= ideal_max:
        return 1.00, "in_band"

    if yoe < ideal_min:
        years_outside = ideal_min - yoe
        decay = 0.85 ** years_outside
        label = f"below_band_{years_outside:.1f}yr"
    else:
        years_outside = yoe - ideal_max
        decay = 0.85 ** years_outside
        label = f"above_band_{years_outside:.1f}yr"

    return max(0.30, decay), label


def score_education_tier(candidate: Candidate) -> tuple[float, str]:
    """
    Score education tier (per schema: tier_1, tier_2, tier_3, tier_4, unknown).
    """
    tier_scores = {
        "tier_1": 1.00,
        "tier_2": 0.85,
        "tier_3": 0.70,
        "tier_4": 0.55,
        "unknown": 0.60,
    }
    if not candidate.education:
        return 0.60, "no_education"
    # Use highest tier (lowest number = best)
    tiers = [e.get("tier", "unknown") for e in candidate.education]
    tier_order = ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]
    best_tier = min(tiers, key=lambda t: tier_order.index(t) if t in tier_order else 4)
    return tier_scores.get(best_tier, 0.60), best_tier


def score_work_mode(candidate: Candidate) -> tuple[float, str]:
    """
    Score work mode against JD's hybrid preference.
    """
    mode = candidate.redrob_signals.get("preferred_work_mode", "").lower()
    scores = {
        "hybrid": 1.00,
        "flexible": 1.00,
        "remote": 0.90,
        "onsite": 0.85,
    }
    return scores.get(mode, 0.85), mode or "unknown"


def compute_structural_score(candidate: Candidate, weights: Optional[dict] = None) -> tuple[float, dict]:
    """
    Combine YOE band, location, education tier, work mode into a structural score.

    Weights default from config/combiner_weights.yaml:
        yoe_band: 0.40
        location: 0.25
        education: 0.20
        work_mode: 0.15

    Returns:
        (score in [0, 1], breakdown dict)
    """
    if weights is None:
        weights = {"yoe_band": 0.40, "location": 0.25, "education": 0.20, "work_mode": 0.15}

    yoe_score, yoe_label = score_yoe_band(candidate)
    loc_score, loc_label = score_location(candidate)
    edu_score, edu_label = score_education_tier(candidate)
    mode_score, mode_label = score_work_mode(candidate)

    structural_score = (
        weights["yoe_band"] * yoe_score +
        weights["location"] * loc_score +
        weights["education"] * edu_score +
        weights["work_mode"] * mode_score
    )

    return float(structural_score), {
        "yoe_band": {"score": yoe_score, "label": yoe_label, "weight": weights["yoe_band"]},
        "location": {"score": loc_score, "label": loc_label, "weight": weights["location"]},
        "education": {"score": edu_score, "label": edu_label, "weight": weights["education"]},
        "work_mode": {"score": mode_score, "label": mode_label, "weight": weights["work_mode"]},
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Seniority Match (Module 6 — seniority sub-component)
# ═══════════════════════════════════════════════════════════════════════

# Title level keywords (extracted from current_title)
SENIOR_KEYWORDS = ["senior", "staff", "lead", "sr.", "sr "]
JUNIOR_KEYWORDS = ["junior", "intern", "trainee", "associate", "graduate"]
TOO_SENIOR_NONCODING_KEYWORDS = ["architect", "director", "vp ", "head of", "manager", "chief", "cto", "cdo"]


def score_seniority(candidate: Candidate, cosine_title_sim: Optional[float] = None) -> tuple[float, dict]:
    """
    Score seniority match based on title level.

    Args:
        candidate: Candidate object
        cosine_title_sim: optional cosine similarity between candidate title_emb
                          and intent's title_archetype embedding. If provided
                          AND above 0.65, we boost the seniority score (strong
                          archetype match).

    Returns:
        (score in [0, 1], breakdown dict)
    """
    title = (candidate.title or "").lower()

    # Detect title level
    is_senior = any(kw in title for kw in SENIOR_KEYWORDS)
    is_junior = any(kw in title for kw in JUNIOR_KEYWORDS)
    is_too_senior_noncoding = any(kw in title for kw in TOO_SENIOR_NONCODING_KEYWORDS)

    if is_junior:
        level_score = 0.40
        level_label = "junior"
    elif is_too_senior_noncoding and not is_senior:
        # Manager/Director/Architect without "Senior" prefix → likely non-coding role
        level_score = 0.50
        level_label = "too_senior_noncoding"
    elif is_senior:
        level_score = 1.00
        level_label = "senior"
    else:
        # Mid-level — just "Engineer" / "ML Engineer" with no prefix
        level_score = 0.85
        level_label = "mid"

    # Archetype match boost (if cosine provided)
    archetype_match_score = 0.0
    archetype_label = "no_archetype_match"
    if cosine_title_sim is not None:
        if cosine_title_sim >= 0.65:
            archetype_match_score = 1.0
            archetype_label = "strong_match"
        elif cosine_title_sim >= 0.45:
            archetype_match_score = 0.7
            archetype_label = "partial_match"
        else:
            archetype_match_score = 0.0
            archetype_label = "weak_match"

    # Combined seniority score: weighted average of level + archetype
    if cosine_title_sim is not None:
        # 60% level, 40% archetype match
        combined = 0.60 * level_score + 0.40 * archetype_match_score
    else:
        # No archetype data — use level only
        combined = level_score

    return float(combined), {
        "level_score": level_score,
        "level_label": level_label,
        "archetype_score": archetype_match_score,
        "archetype_label": archetype_label,
        "cosine_title_sim": cosine_title_sim,
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: Combined features orchestrator
# ═══════════════════════════════════════════════════════════════════════

def extract_all_features(
    candidate: Candidate,
    skill_resolver: Optional[SkillTaxonomyResolver] = None,
    jd_required_skills: Optional[list[str]] = None,
    cosine_title_sim: Optional[float] = None,
) -> dict:
    """
    Compute all features needed by the combiner:
        - production_evidence (Module 4)
        - esco_coverage (Module 6.5)
        - structural_score (Module 6 structural)
        - seniority_match (Module 6 seniority)

    Returns a dict with all sub-scores + breakdowns for the reasoning generator.
    """
    # ─── Production evidence ─────────────────────────────────────────────
    prod = extract_production_evidence(candidate)

    # ─── ESCO coverage ───────────────────────────────────────────────────
    if skill_resolver is None:
        skill_resolver = SkillTaxonomyResolver.from_config()
    if jd_required_skills is None:
        # Lazy import to avoid circular import
        from .intent import get_jd_required_canonical_skills, load_intent_schema
        jd_required_skills = get_jd_required_canonical_skills(load_intent_schema())

    matched_skills = skill_resolver.resolve(candidate)
    esco_score, esco_breakdown = skill_resolver.coverage_score(matched_skills, jd_required_skills)

    # ─── Structural score ────────────────────────────────────────────────
    structural_score, structural_breakdown = compute_structural_score(candidate)

    # ─── Seniority match ─────────────────────────────────────────────────
    seniority_score, seniority_breakdown = score_seniority(candidate, cosine_title_sim)

    return {
        "production_evidence": prod["score"],
        "production_evidence_breakdown": prod,
        "esco_coverage": esco_score,
        "esco_coverage_breakdown": esco_breakdown,
        "matched_canonical_skills": sorted(matched_skills),
        "structural_score": structural_score,
        "structural_breakdown": structural_breakdown,
        "seniority_match": seniority_score,
        "seniority_breakdown": seniority_breakdown,
    }
