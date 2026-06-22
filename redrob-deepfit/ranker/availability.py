"""
Module 5: Behavioral Availability Multiplier
============================================================================

Computes a multiplicative availability multiplier ∈ [0.30, 1.20] that captures
how "reachable" and "available" a candidate actually is — independent of their
semantic fit.

Per the JD: "A perfect-on-paper candidate who hasn't logged in for 6 months and
has a 5% response rate is, for hiring purposes, not actually available."

This is MULTIPLICATIVE (not additive) by design:
    - A perfect-on-paper dead candidate (semantic 1.0, base 0.85) caps at
      0.85 × 1.0 × 1.0 × 0.30 = 0.255 → below rank 100 cutoff
    - A slightly-less-perfect active candidate (semantic 0.75, base 0.70) gets
      0.70 × 1.0 × 1.0 × 1.15 = 0.805 → top 10 range

Components (all multiplied together, then clipped to [min, max]):
    1. recency           — exponential decay, 30-day half-life
    2. response_rate     — logistic, centered at 0.30
    3. open_to_work      — 1.15 boost if True, else 1.0
    4. verified_email    — 1.02 boost if True, else 1.0
    5. verified_phone    — 1.02 boost if True, else 1.0
    6. linkedin_connected — 1.03 boost if True, else 1.0
    7. notice_period     — tiered: 30d→1.0, 60d→0.95, 90d→0.85, 180d→0.70, 180+d→0.50
    8. github_activity   — max +10% boost (normalized by /50)
    9. recruiter_demand  — max +5% boost (saved_by_recruiters_30d / 20)
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from .types import Candidate

log = logging.getLogger(__name__)


# ─── Default config (mirrors config/combiner_weights.yaml availability_components) ───
DEFAULT_REFERENCE_DATE = date(2026, 6, 20)

DEFAULT_CONFIG = {
    "recency": {
        "half_life_days": 30,
        "min_value": 0.10,
    },
    "response_rate": {
        "sigmoid_center": 0.30,
        "sigmoid_slope": 8.0,
    },
    "open_to_work": {
        "boost": 1.15,
        "no_boost": 1.0,
    },
    "verified_email_boost": 1.02,
    "verified_phone_boost": 1.02,
    "linkedin_connected_boost": 1.03,
    "notice_period": {
        "penalty_tiers": [
            {"max_days": 30, "multiplier": 1.00},
            {"max_days": 60, "multiplier": 0.95},
            {"max_days": 90, "multiplier": 0.85},
            {"max_days": 180, "multiplier": 0.70},
            {"max_days": 999999, "multiplier": 0.50},
        ],
    },
    "github_activity": {
        "boost_max": 0.10,
        "normalize_divisor": 50,
    },
    "recruiter_demand": {
        "boost_max": 0.05,
        "normalize_divisor": 20,
    },
    "availability_mult": {
        "min": 0.30,
        "max": 1.20,
    },
}


class AvailabilityScorer:
    """
    Computes the behavioral availability multiplier.

    Usage:
        scorer = AvailabilityScorer.from_config()
        mult, breakdown = scorer.score(candidate)
        # mult ∈ [0.30, 1.20], breakdown dict for reasoning generator
    """

    def __init__(
        self,
        config: dict,
        reference_date: date = DEFAULT_REFERENCE_DATE,
    ):
        self.config = config
        self.reference_date = reference_date

        # Extract sub-configs for fast access
        self.recency_cfg = config.get("recency", DEFAULT_CONFIG["recency"])
        self.response_cfg = config.get("response_rate", DEFAULT_CONFIG["response_rate"])
        self.otw_cfg = config.get("open_to_work", DEFAULT_CONFIG["open_to_work"])
        self.verified_email_boost = config.get("verified_email_boost", 1.02)
        self.verified_phone_boost = config.get("verified_phone_boost", 1.02)
        self.linkedin_boost = config.get("linkedin_connected_boost", 1.03)
        self.notice_cfg = config.get("notice_period", DEFAULT_CONFIG["notice_period"])
        self.github_cfg = config.get("github_activity", DEFAULT_CONFIG["github_activity"])
        self.demand_cfg = config.get("recruiter_demand", DEFAULT_CONFIG["recruiter_demand"])
        self.range_cfg = config.get("availability_mult", DEFAULT_CONFIG["availability_mult"])

    @classmethod
    def from_config(
        cls,
        config_path: Optional[Path] = None,
        reference_date: date = DEFAULT_REFERENCE_DATE,
    ) -> "AvailabilityScorer":
        """
        Load config from config/combiner_weights.yaml (availability_components section).
        Falls back to DEFAULT_CONFIG if file missing.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "combiner_weights.yaml"
        if config_path.exists():
            with open(config_path) as f:
                full_cfg = yaml.safe_load(f)
            # The YAML nests availability_components at top level
            config = full_cfg.get("availability_components", DEFAULT_CONFIG)
            # Range is at top level
            config["availability_mult"] = full_cfg.get("availability_mult",
                                                       DEFAULT_CONFIG["availability_mult"])
        else:
            config = DEFAULT_CONFIG
        return cls(config=config, reference_date=reference_date)

    # ─── Main entry point ─────────────────────────────────────────────────
    def score(self, candidate: Candidate) -> tuple[float, dict]:
        """
        Compute availability multiplier and breakdown.

        Returns:
            (mult in [min, max], breakdown dict)
        """
        breakdown = {}

        # ─── 1. Recency ───────────────────────────────────────────────────
        days_inactive = candidate.days_since_last_active(self.reference_date)
        recency = self._recency_score(days_inactive)
        breakdown["recency"] = {
            "value": recency,
            "days_inactive": days_inactive,
            "last_active_date": candidate.last_active_date,
        }

        # ─── 2. Response rate ─────────────────────────────────────────────
        rr = candidate.response_rate
        response = self._response_rate_score(rr)
        breakdown["response_rate"] = {
            "value": response,
            "raw_response_rate": rr,
        }

        # ─── 3. Open-to-work ──────────────────────────────────────────────
        otw_mult = self.otw_cfg["boost"] if candidate.open_to_work else self.otw_cfg["no_boost"]
        breakdown["open_to_work"] = {
            "value": otw_mult,
            "flag": candidate.open_to_work,
        }

        # ─── 4-6. Verified signals ────────────────────────────────────────
        sig = candidate.redrob_signals
        v_email = self.verified_email_boost if sig.get("verified_email", False) else 1.0
        v_phone = self.verified_phone_boost if sig.get("verified_phone", False) else 1.0
        linkedin = self.linkedin_boost if sig.get("linkedin_connected", False) else 1.0
        breakdown["verified"] = {
            "email": v_email,
            "phone": v_phone,
            "linkedin": linkedin,
        }

        # ─── 7. Notice period ─────────────────────────────────────────────
        notice_days = candidate.notice_period_days
        notice_mult = self._notice_period_score(notice_days)
        breakdown["notice_period"] = {
            "value": notice_mult,
            "days": notice_days,
        }

        # ─── 8. GitHub activity ───────────────────────────────────────────
        gh_score = candidate.github_activity_score
        github_mult = self._github_activity_score(gh_score)
        breakdown["github_activity"] = {
            "value": github_mult,
            "raw_score": gh_score,
        }

        # ─── 9. Recruiter demand ──────────────────────────────────────────
        saved_30d = int(sig.get("saved_by_recruiters_30d", 0))
        demand_mult = self._recruiter_demand_score(saved_30d)
        breakdown["recruiter_demand"] = {
            "value": demand_mult,
            "saved_by_recruiters_30d": saved_30d,
        }

        # ─── Combine (all multiplicative) ─────────────────────────────────
        raw_mult = (
            recency
            * response
            * otw_mult
            * v_email
            * v_phone
            * linkedin
            * notice_mult
            * github_mult
            * demand_mult
        )

        # ─── Clip to [min, max] ───────────────────────────────────────────
        min_mult = self.range_cfg.get("min", 0.30)
        max_mult = self.range_cfg.get("max", 1.20)
        final_mult = max(min_mult, min(max_mult, raw_mult))

        breakdown["raw_mult"] = raw_mult
        breakdown["final_mult"] = final_mult
        breakdown["clipped"] = (raw_mult < min_mult) or (raw_mult > max_mult)

        return float(final_mult), breakdown

    # ─── Sub-scorers ──────────────────────────────────────────────────────
    def _recency_score(self, days_inactive: int) -> float:
        """
        Exponential decay with 30-day half-life.
        - active today: 1.0
        - 30d ago: 0.5
        - 60d ago: 0.25
        - 90d ago: 0.125
        - 180d+ ago: floored at min_value (0.10)
        """
        if days_inactive < 0:
            # Sentinel from missing date — assume worst case
            return self.recency_cfg.get("min_value", 0.10)

        half_life = self.recency_cfg.get("half_life_days", 30)
        min_val = self.recency_cfg.get("min_value", 0.10)
        if days_inactive == 0:
            return 1.0
        score = 0.5 ** (days_inactive / half_life)
        return max(min_val, score)

    def _response_rate_score(self, rr: float) -> float:
        """
        Logistic function centered at sigmoid_center (default 0.30).
        - rr = 0.0 → ~0.07 (very low)
        - rr = 0.30 → 0.5 (neutral)
        - rr = 0.50 → ~0.92
        - rr = 0.80 → ~0.998
        - rr = 1.00 → ~0.9997
        """
        center = self.response_cfg.get("sigmoid_center", 0.30)
        slope = self.response_cfg.get("sigmoid_slope", 8.0)
        return 1.0 / (1.0 + math.exp(-slope * (rr - center)))

    def _notice_period_score(self, notice_days: int) -> float:
        """
        Tiered multiplier:
            ≤30d  → 1.00 (ideal, can buy out)
            ≤60d  → 0.95 (slight penalty)
            ≤90d  → 0.85 (higher bar per JD)
            ≤180d → 0.70 (significant penalty)
            >180d → 0.50 (very high bar)
        """
        tiers = self.notice_cfg.get("penalty_tiers", DEFAULT_CONFIG["notice_period"]["penalty_tiers"])
        for tier in tiers:
            if notice_days <= tier["max_days"]:
                return tier["multiplier"]
        return 0.50  # fallback

    def _github_activity_score(self, gh_score: float) -> float:
        """
        Max +10% boost, normalized by /50.
        - gh_score = -1 (no GitHub linked) → 1.0 (neutral, not penalty)
        - gh_score = 0 → 1.0
        - gh_score = 25 → 1.05
        - gh_score = 50+ → 1.10 (max boost)
        """
        if gh_score < 0:
            # No GitHub linked — neutral (don't penalize)
            return 1.0
        boost_max = self.github_cfg.get("boost_max", 0.10)
        divisor = self.github_cfg.get("normalize_divisor", 50)
        boost_fraction = min(gh_score / divisor, 1.0)
        return 1.0 + boost_max * boost_fraction

    def _recruiter_demand_score(self, saved_30d: int) -> float:
        """
        Max +5% boost based on how many recruiters saved the candidate in last 30d.
        - saved = 0 → 1.0
        - saved = 10 → 1.025
        - saved = 20+ → 1.05 (max boost)
        """
        boost_max = self.demand_cfg.get("boost_max", 0.05)
        divisor = self.demand_cfg.get("normalize_divisor", 20)
        boost_fraction = min(saved_30d / divisor, 1.0)
        return 1.0 + boost_max * boost_fraction


# ─── Convenience function ─────────────────────────────────────────────────
def score_availability(candidate: Candidate, scorer: AvailabilityScorer = None) -> tuple[float, dict]:
    """
    Convenience wrapper. Returns (mult in [0.30, 1.20], breakdown dict).
    """
    if scorer is None:
        scorer = AvailabilityScorer.from_config()
    return scorer.score(candidate)
