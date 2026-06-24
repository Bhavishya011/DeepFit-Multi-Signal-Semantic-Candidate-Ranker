"""
Shared dataclasses for the DeepFit ranker.

These types flow through the pipeline:
    Candidate  →  filters  →  recall  →  rerank  →  features  →  combiner  →  reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class Candidate:
    """Lightweight wrapper around the raw candidate JSON record."""

    candidate_id: str
    profile: dict
    career_history: list[dict]
    education: list[dict]
    skills: list[dict]
    certifications: list[dict] = field(default_factory=list)
    languages: list[dict] = field(default_factory=list)
    redrob_signals: dict = field(default_factory=dict)
    # Raw record kept for reasoning generator (slot-filling)
    _raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, record: dict) -> "Candidate":
        return cls(
            candidate_id=record["candidate_id"],
            profile=record.get("profile", {}),
            career_history=record.get("career_history", []),
            education=record.get("education", []),
            skills=record.get("skills", []),
            certifications=record.get("certifications", []),
            languages=record.get("languages", []),
            redrob_signals=record.get("redrob_signals", {}),
            _raw=record,
        )

    # ─── Convenience accessors ────────────────────────────────────────────
    @property
    def title(self) -> str:
        return self.profile.get("current_title", "")

    @property
    def yoe(self) -> float:
        return float(self.profile.get("years_of_experience", 0))

    @property
    def location(self) -> str:
        return self.profile.get("location", "")

    @property
    def country(self) -> str:
        return self.profile.get("country", "")

    @property
    def industry(self) -> str:
        return self.profile.get("current_industry", "")

    @property
    def summary(self) -> str:
        return self.profile.get("summary", "")

    @property
    def headline(self) -> str:
        return self.profile.get("headline", "")

    @property
    def open_to_work(self) -> bool:
        return bool(self.redrob_signals.get("open_to_work_flag", False))

    @property
    def last_active_date(self) -> str:
        return self.redrob_signals.get("last_active_date", "")

    @property
    def response_rate(self) -> float:
        return float(self.redrob_signals.get("recruiter_response_rate", 0.0))

    @property
    def notice_period_days(self) -> int:
        return int(self.redrob_signals.get("notice_period_days", 0))

    @property
    def github_activity_score(self) -> float:
        return float(self.redrob_signals.get("github_activity_score", -1.0))

    def days_since_last_active(self, ref: date = date(2026, 6, 20)) -> int:
        try:
            d = date.fromisoformat(self.last_active_date)
            return max(0, (ref - d).days)
        except Exception:
            return -1

    def career_text(self) -> str:
        """Concatenated career_history descriptions — used for encoder + production-evidence."""
        return " ".join(j.get("description", "") for j in self.career_history)

    def skills_text(self) -> str:
        """Concatenated skill names — used for encoder."""
        return " ".join(s.get("name", "") for s in self.skills)

    def to_dict(self) -> dict:
        return self._raw


@dataclass
class Intent:
    """Parsed JD intent — output of Module 1 (ranker/intent.py)."""

    schema: dict                       # raw intent_schema.json contents
    axis_embeddings: dict[str, Any]    # axis_name -> embedding vector (np.ndarray or list)

    def axis_query_text(self, axis_name: str) -> str:
        """Return the embedding query text for a given axis."""
        for req in self.schema.get("explicit_positive_requirements", []):
            if req["axis"] == axis_name:
                return req.get("embedding_query_text", "")
        for req in self.schema.get("nice_to_haves", []):
            if req["axis"] == axis_name:
                return req.get("embedding_query_text", "")
        for group_name, group in self.schema.get("skill_taxonomy_groups", {}).items():
            if group_name == axis_name:
                return group.get("embedding_query_text", "")
        return ""

    def must_have_axes(self) -> list[str]:
        return [r["axis"] for r in self.schema.get("explicit_positive_requirements", [])
                if r.get("must_have")]


@dataclass
class FilterVerdict:
    """Output of Module 3 (ranker/filters.py) — honeypot + trap consistency scorer."""

    honeypot_penalty: float            # 0.0 (hard kill) or 1.0 (clean)
    trap_penalty: float                # 0.3 (trap detected) or 1.0 (clean)
    fired_honeypot_rules: list[str] = field(default_factory=list)
    fired_trap_rules: list[str] = field(default_factory=list)
    rule_details: dict[str, dict] = field(default_factory=dict)

    @property
    def is_honeypot(self) -> bool:
        return self.honeypot_penalty < 1.0

    @property
    def is_trap(self) -> bool:
        return self.trap_penalty < 1.0

    @property
    def passes(self) -> bool:
        """True if candidate survives hard filter (honeypot)."""
        return not self.is_honeypot


@dataclass
class ScoreComponents:
    """All inputs to the final combiner (Module 8)."""

    semantic_score: float = 0.0
    structural_score: float = 0.0
    production_evidence: float = 0.0
    esco_coverage: float = 0.0
    seniority_match: float = 0.0
    honeypot_penalty: float = 1.0
    trap_penalty: float = 1.0
    availability_mult: float = 1.0
    # Sub-component breakdowns for reasoning generator
    axis_scores: dict[str, float] = field(default_factory=dict)
    availability_breakdown: dict[str, float] = field(default_factory=dict)
    structural_breakdown: dict[str, float] = field(default_factory=dict)
    matched_canonical_skills: list[str] = field(default_factory=list)
    fired_honeypot_rules: list[str] = field(default_factory=list)
    fired_trap_rules: list[str] = field(default_factory=list)


@dataclass
class RankedCandidate:
    """Final output of the pipeline for one candidate."""

    candidate_id: str
    rank: int
    score: float
    reasoning: str
    components: ScoreComponents
