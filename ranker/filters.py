"""
Module 3: Honeypot + Trap Consistency Scorer
============================================================================

Implements the rules from config/honeypot_rules.yaml:
    - 8 hard honeypot rules  → honeypot_penalty = 0.0 (candidate excluded)
    - 5 soft trap rules       → trap_penalty = 0.3 (candidate sinks below top-100)

This is the highest-ROI module (+8-15% NDCG@10 per SOTA research).
A hard filter, not a soft signal — by design.

Usage:
    from ranker.filters import HoneypotScorer
    scorer = HoneypotScorer.from_config()
    verdict = scorer.score(candidate)
    if not verdict.passes:
        # exclude from top-100
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from .types import Candidate, FilterVerdict

log = logging.getLogger(__name__)


# ─── Paths ────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
HONEYPOT_RULES_PATH = CONFIG_DIR / "honeypot_rules.yaml"
INTENT_SCHEMA_PATH = CONFIG_DIR / "intent_schema.json"


# ─── Reference date (when the candidate pool was released) ────────────────
DEFAULT_REFERENCE_DATE = date(2026, 6, 20)


# ─── Honeypot Scorer ──────────────────────────────────────────────────────
class HoneypotScorer:
    """
    Hard-filter + soft-penalty scorer.

    Hard rules (any one fires → score=0, candidate excluded):
        1. single_skill_duration_exceeds_yoe
        1b. skill_duration_sum_absurd (> 10x YOE)
        2. endorsement_inflation (5+ expert skills with 0 endorsements)
        3. career_date_impossibility (career started > 5 yrs before education end)
        4. yoe_inflation (claimed YOE exceeds actual career span by > 2 yrs)
        5. salary_range_inverted (min > max)
        6. inactive_responsive_mismatch (inactive > 180d but response > 0.8)
        7. interview_rate_contradiction

    Soft trap rules (any one fires → score *= 0.3):
        1. title_skill_mismatch (non-tech title + 5+ AI skills)
        2. services_only_career (all jobs at TCS/Infosys/etc.)
        3. pure_research_background (all titles are Researcher/Postdoc)
        4. recent_langchain_only (all AI skills < 12mo + no pre-LLM ML)
        5. title_chaser (3+ roles < 18mo + Senior→Staff→Principal progression)
    """

    def __init__(
        self,
        rules_config: dict,
        intent_schema: Optional[dict] = None,
        reference_date: date = DEFAULT_REFERENCE_DATE,
    ):
        self.rules_config = rules_config
        self.intent_schema = intent_schema or {}
        self.reference_date = reference_date
        self.trap_penalty_value = rules_config.get("trap_penalty_value", 0.3)

        # Load reference lists from rules config
        self.services_companies = set(rules_config.get("services_companies", []))
        self.research_titles = set(rules_config.get("research_titles", []))
        self.ai_skill_keywords = set(rules_config.get("ai_skill_keywords", []))
        self.pre_llm_ml_keywords = set(rules_config.get("pre_llm_ml_keywords", []))

        # Load title archetype from intent schema (for trap detection)
        title_match = self.intent_schema.get("title_archetype_match", {})
        self.mismatched_titles = set(title_match.get("mismatched_titles", []))
        self.disqualifying_titles = set(title_match.get("disqualifying_titles", []))

        log.debug(
            f"HoneypotScorer initialized: "
            f"{len(self.services_companies)} services companies, "
            f"{len(self.research_titles)} research titles, "
            f"{len(self.ai_skill_keywords)} AI skill keywords, "
            f"{len(self.mismatched_titles)} mismatched titles"
        )

    @classmethod
    def from_config(
        cls,
        rules_path: Path = HONEYPOT_RULES_PATH,
        intent_schema_path: Path = INTENT_SCHEMA_PATH,
        reference_date: date = DEFAULT_REFERENCE_DATE,
    ) -> "HoneypotScorer":
        """Load rules from config files."""
        with open(rules_path) as f:
            rules_config = yaml.safe_load(f)
        import json
        with open(intent_schema_path) as f:
            intent_schema = json.load(f)
        return cls(
            rules_config=rules_config,
            intent_schema=intent_schema,
            reference_date=reference_date,
        )

    # ─── Main entry point ─────────────────────────────────────────────────
    def score(self, candidate: Candidate) -> FilterVerdict:
        """
        Run all honeypot + trap rules against a candidate.

        Returns FilterVerdict with:
            honeypot_penalty: 0.0 if any hard rule fires, else 1.0
            trap_penalty: 0.3 if any soft rule fires, else 1.0
            fired_honeypot_rules: list of rule names that fired
            fired_trap_rules: list of trap rule names that fired
            rule_details: {rule_name: {reason: str, evidence: str}}
        """
        fired_honeypot = []
        fired_trap = []
        rule_details = {}

        # ─── Hard honeypot rules ──────────────────────────────────────────
        for rule_fn in [
            self._rule_single_skill_duration_exceeds_yoe,
            self._rule_skill_duration_sum_absurd,
            self._rule_endorsement_inflation,
            self._rule_career_date_impossibility,
            self._rule_yoe_inflation,
            self._rule_salary_range_inverted,
            self._rule_inactive_responsive_mismatch,
            self._rule_interview_rate_contradiction,
        ]:
            fired, detail = rule_fn(candidate)
            if fired:
                fired_honeypot.append(rule_fn.__name__.replace("_rule_", ""))
                rule_details[rule_fn.__name__.replace("_rule_", "")] = detail

        # ─── Soft trap rules (only run if not already hard-killed) ────────
        if not fired_honeypot:
            for trap_fn in [
                self._trap_title_skill_mismatch,
                self._trap_services_only_career,
                self._trap_pure_research_background,
                self._trap_recent_langchain_only,
                self._trap_title_chaser,
            ]:
                fired, detail = trap_fn(candidate)
                if fired:
                    fired_trap.append(trap_fn.__name__.replace("_trap_", ""))
                    rule_details[trap_fn.__name__.replace("_trap_", "")] = detail

        honeypot_penalty = 0.0 if fired_honeypot else 1.0
        trap_penalty = self.trap_penalty_value if fired_trap else 1.0

        return FilterVerdict(
            honeypot_penalty=honeypot_penalty,
            trap_penalty=trap_penalty,
            fired_honeypot_rules=fired_honeypot,
            fired_trap_rules=fired_trap,
            rule_details=rule_details,
        )

    # ═════════════════════════════════════════════════════════════════════
    # HARD HONEYPOT RULES
    # ═════════════════════════════════════════════════════════════════════

    def _rule_single_skill_duration_exceeds_yoe(self, c: Candidate) -> tuple[bool, dict]:
        """
        Any single skill duration_months > YOE * 12 + 6 (months grace).
        Catches real impossibilities like Pinecone 88mo > YOE 72mo.
        """
        yoe_months = c.yoe * 12
        if yoe_months <= 0:
            return False, {}
        threshold = yoe_months + 6
        for skill in c.skills:
            dur = skill.get("duration_months", 0)
            if dur > threshold:
                return True, {
                    "reason": f"Skill '{skill['name']}' duration {dur}mo exceeds YOE {yoe_months:.0f}mo + 6mo grace",
                    "evidence": f"skill={skill['name']}, duration_months={dur}, yoe_months={yoe_months:.0f}",
                    "threshold": threshold,
                }
        return False, {}

    def _rule_skill_duration_sum_absurd(self, c: Candidate) -> tuple[bool, dict]:
        """Sum of all skill durations > 10x YOE in months. Catches massive stuffing."""
        yoe_months = c.yoe * 12
        if yoe_months <= 0:
            return False, {}
        total_dur = sum(s.get("duration_months", 0) for s in c.skills)
        threshold = yoe_months * 10
        if total_dur > threshold:
            return True, {
                "reason": f"Sum of skill durations {total_dur}mo > 10x YOE {yoe_months:.0f}mo",
                "evidence": f"total_skill_duration={total_dur}, yoe_months={yoe_months:.0f}",
                "threshold": threshold,
            }
        return False, {}

    def _rule_endorsement_inflation(self, c: Candidate) -> tuple[bool, dict]:
        """5+ skills with proficiency=expert AND endorsements=0."""
        expert_zero = [s for s in c.skills
                       if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0]
        if len(expert_zero) >= 5:
            return True, {
                "reason": f"{len(expert_zero)} expert skills with 0 endorsements",
                "evidence": f"expert_zero_skills={[s['name'] for s in expert_zero]}",
                "threshold": 5,
            }
        return False, {}

    def _rule_career_date_impossibility(self, c: Candidate) -> tuple[bool, dict]:
        """
        Earliest career start is more than 5 years before earliest education end_year.
        (Started working professionally before graduating by > 5 years.)
        """
        if not c.career_history or not c.education:
            return False, {}

        try:
            earliest_career_year = min(
                int(j["start_date"][:4]) for j in c.career_history
                if j.get("start_date")
            )
            earliest_edu_end = min(
                int(e.get("end_year", 9999)) for e in c.education
                if e.get("end_year")
            )
        except (ValueError, TypeError):
            return False, {}

        if earliest_career_year < earliest_edu_end - 5:
            return True, {
                "reason": f"Career started {earliest_career_year} > 5 yrs before education end {earliest_edu_end}",
                "evidence": f"earliest_career_year={earliest_career_year}, earliest_edu_end={earliest_edu_end}",
                "threshold": earliest_edu_end - 5,
            }
        return False, {}

    def _rule_yoe_inflation(self, c: Candidate) -> tuple[bool, dict]:
        """Claimed YOE exceeds (reference_date - earliest_career_start) by > 2 years."""
        if not c.career_history:
            return False, {}
        try:
            earliest = min(
                date.fromisoformat(j["start_date"][:10])
                for j in c.career_history if j.get("start_date")
            )
        except (ValueError, TypeError):
            return False, {}

        actual_yoe = (self.reference_date - earliest).days / 365.25
        if c.yoe > actual_yoe + 2:
            return True, {
                "reason": f"Claimed YOE {c.yoe:.1f} > actual career span {actual_yoe:.1f} yrs + 2 yr grace",
                "evidence": f"claimed_yoe={c.yoe}, earliest_career_start={earliest.isoformat()}, actual_span={actual_yoe:.1f}",
                "threshold": actual_yoe + 2,
            }
        return False, {}

    def _rule_salary_range_inverted(self, c: Candidate) -> tuple[bool, dict]:
        """expected_salary_range_inr_lpa.min > expected_salary_range_inr_lpa.max."""
        sal = c.redrob_signals.get("expected_salary_range_inr_lpa", {})
        min_sal = sal.get("min", 0)
        max_sal = sal.get("max", 0)
        if min_sal > max_sal:
            return True, {
                "reason": f"Salary min({min_sal}) > max({max_sal})",
                "evidence": f"min={min_sal}, max={max_sal}",
            }
        return False, {}

    def _rule_inactive_responsive_mismatch(self, c: Candidate) -> tuple[bool, dict]:
        """Inactive > 180d but recruiter_response_rate > 0.8 — suspicious combination."""
        days = c.days_since_last_active(self.reference_date)
        rr = c.response_rate
        if days > 180 and rr > 0.8:
            return True, {
                "reason": f"Inactive {days}d but response rate {rr} (suspicious)",
                "evidence": f"days_inactive={days}, response_rate={rr}",
                "thresholds": {"days": 180, "response_rate": 0.8},
            }
        return False, {}

    def _rule_interview_rate_contradiction(self, c: Candidate) -> tuple[bool, dict]:
        """
        interview_completion_rate > 0 but no offer history (offer_acceptance_rate = -1)
        AND no recent applications (applications_submitted_30d = 0).
        Internal contradiction — can't have completed interviews without applying.
        """
        sig = c.redrob_signals
        icr = sig.get("interview_completion_rate", 0)
        oar = sig.get("offer_acceptance_rate", 0)
        apps = sig.get("applications_submitted_30d", 0)
        if icr > 0 and oar == -1 and apps == 0:
            return True, {
                "reason": f"Interview completion {icr} > 0 but no offer history and no recent applications",
                "evidence": f"interview_completion_rate={icr}, offer_acceptance_rate={oar}, applications_30d={apps}",
            }
        return False, {}

    # ═════════════════════════════════════════════════════════════════════
    # SOFT TRAP RULES
    # ═════════════════════════════════════════════════════════════════════

    def _trap_title_skill_mismatch(self, c: Candidate) -> tuple[bool, dict]:
        """
        Non-technical title (Marketing Manager, HR Manager, etc.) but 5+ AI skills claimed.
        The "Marketing Manager with 9 AI skills" trap from the JD spec.

        Uses substring matching (case-insensitive) to catch title variations like
        'Senior Marketing Manager' or 'Operations Manager II'. This is more robust
        than exact-match against a fixed list.
        """
        title = c.title
        if not title:
            return False, {}

        title_lower = title.lower()
        # Check if title contains any mismatched title as a substring
        matched_mismatched = [
            mt for mt in self.mismatched_titles
            if mt.lower() in title_lower
        ]
        if not matched_mismatched:
            return False, {}

        ai_skills = [s for s in c.skills if s.get("name") in self.ai_skill_keywords]
        if len(ai_skills) >= 5:
            return True, {
                "reason": f"Title '{title}' (contains '{matched_mismatched[0]}') with {len(ai_skills)} AI skills (keyword stuffer)",
                "evidence": f"title={title}, matched_mismatched={matched_mismatched}, ai_skills={[s['name'] for s in ai_skills]}",
                "threshold": 5,
            }
        return False, {}

    def _trap_services_only_career(self, c: Candidate) -> tuple[bool, dict]:
        """
        Entire career at IT services companies (TCS, Infosys, Wipro, etc.).
        Exception: if any prior role was at a non-services company, don't fire.
        """
        if not c.career_history:
            return False, {}

        companies = [j.get("company", "") for j in c.career_history]
        # Check if ALL companies are in the services list (substring match)
        all_services = all(
            any(svc.lower() in comp.lower() for svc in self.services_companies)
            for comp in companies
        )
        if all_services:
            return True, {
                "reason": f"Entire career at IT services companies ({', '.join(companies[:2])}...)",
                "evidence": f"companies={companies}",
                "exception_clause": "Prior product-company experience would exempt this candidate",
            }
        return False, {}

    def _trap_pure_research_background(self, c: Candidate) -> tuple[bool, dict]:
        """
        All career history titles are research-only (Postdoc, Researcher, Research Scientist).
        JD: "pure research environments ... we will not move forward."
        """
        if not c.career_history:
            return False, {}

        titles = [j.get("title", "") for j in c.career_history]
        all_research = all(
            any(rt.lower() in t.lower() for rt in self.research_titles)
            for t in titles
        )
        if all_research:
            return True, {
                "reason": f"All career titles are research-only ({', '.join(titles[:2])}...)",
                "evidence": f"titles={titles}",
            }
        return False, {}

    def _trap_recent_langchain_only(self, c: Candidate) -> tuple[bool, dict]:
        """
        All AI skill durations < 12 months AND no pre-LLM-era ML experience.
        JD: "AI experience consists primarily of recent (<12mo) projects using LangChain."
        """
        ai_skills = [s for s in c.skills if s.get("name") in self.ai_skill_keywords]
        if not ai_skills:
            return False, {}

        all_recent = all(s.get("duration_months", 0) < 12 for s in ai_skills)
        if not all_recent:
            return False, {}

        has_pre_llm = any(s.get("name") in self.pre_llm_ml_keywords for s in c.skills)
        if not has_pre_llm:
            return True, {
                "reason": f"All {len(ai_skills)} AI skills < 12mo duration AND no pre-LLM ML experience",
                "evidence": f"ai_skills={[s['name'] for s in ai_skills]}",
            }
        return False, {}

    def _trap_title_chaser(self, c: Candidate) -> tuple[bool, dict]:
        """
        3+ roles with duration < 18 months AND title progression shows Senior → Staff → Principal.
        JD: "optimizing for 'Senior' → 'Staff' → 'Principal' titles by switching every 1.5 years."
        """
        if not c.career_history:
            return False, {}

        short_roles = [j for j in c.career_history if j.get("duration_months", 0) < 18]
        if len(short_roles) < 3:
            return False, {}

        # Check title progression for Senior → Staff → Principal pattern
        # Get titles in chronological order (oldest first)
        sorted_jobs = sorted(c.career_history, key=lambda j: j.get("start_date", ""))
        titles = [j.get("title", "") for j in sorted_jobs]
        title_text = " ".join(titles)

        has_senior = "Senior" in title_text or "Sr" in title_text
        has_staff = "Staff" in title_text
        has_principal = "Principal" in title_text

        # Need at least 2 of 3 to fire (avoid false positives on single occurrence)
        progression_count = sum([has_senior, has_staff, has_principal])
        if progression_count >= 2:
            return True, {
                "reason": f"{len(short_roles)} roles < 18mo + title progression (Senior/Staff/Principal count={progression_count})",
                "evidence": f"short_roles={len(short_roles)}, titles={titles}",
                "thresholds": {"min_short_roles": 3, "min_progression_count": 2},
            }
        return False, {}


# ─── Batch scoring helper ─────────────────────────────────────────────────
def score_candidates(candidates: list[Candidate], scorer: HoneypotScorer = None) -> dict[str, FilterVerdict]:
    """
    Score a list of candidates. Returns {candidate_id: FilterVerdict}.

    Useful for batch processing during pre-compute (saves honeypot_scores.parquet).
    """
    if scorer is None:
        scorer = HoneypotScorer.from_config()
    return {c.candidate_id: scorer.score(c) for c in candidates}
