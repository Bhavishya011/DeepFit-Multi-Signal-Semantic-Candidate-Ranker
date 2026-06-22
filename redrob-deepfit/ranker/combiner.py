"""
Module 9: Final Combiner — THE Equation
============================================================================

This is the single source of truth for the final score equation. Every
teammate must import from this module. NO HARDCODED WEIGHTS in pipeline code.

Equation:
    final_score = base_score × honeypot_penalty × trap_penalty × availability_mult

Where:
    base_score = Σ(W[i] × component[i]) for i in components
        W = {
            semantic:       0.55    # Dominant — semantic fit is king
            structural:     0.20    # Yoe/location/education/work_mode
            production:     0.10    # Career descriptions show shipped systems
            esco_coverage:  0.10    # Skill taxonomy coverage (with AI aliases)
            seniority:      0.05    # Title level match
        }
        # Weights MUST sum to 1.0 (asserted at runtime)

    honeypot_penalty ∈ {0.0, 1.0}    # Binary kill switch
    trap_penalty ∈ {0.3, 1.0}        # Soft kill (caps at 30% of base)
    availability_mult ∈ [0.30, 1.20] # Continuous modulation

Output range: [0, 1.20]
    - 0.0 = honeypot (excluded)
    - 0.30 × base = trapped candidate (capped at 30% of base)
    - 1.0 × base = clean candidate with neutral availability
    - 1.20 × base = clean candidate with strong availability signals

All weights loaded from config/combiner_weights.yaml (single source of truth).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .types import Candidate, ScoreComponents, FilterVerdict

log = logging.getLogger(__name__)


# ─── Paths ────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
COMBINER_WEIGHTS_PATH = CONFIG_DIR / "combiner_weights.yaml"


# ─── Default weights (mirrors config/combiner_weights.yaml) ───────────────
DEFAULT_BASE_WEIGHTS = {
    "semantic":       0.55,
    "structural":     0.20,
    "production":     0.10,
    "esco_coverage":  0.10,
    "seniority":      0.05,
}

DEFAULT_HONEYPOT_PENALTY = {"clean": 1.0, "flagged": 0.0}
DEFAULT_TRAP_PENALTY = {"clean": 1.0, "flagged": 0.3}
DEFAULT_AVAILABILITY_RANGE = {"min": 0.30, "max": 1.20}


class FinalCombiner:
    """
    Computes the final score from all components.

    The equation is FIXED — only the weights are tunable, and they come from
    config/combiner_weights.yaml. No hardcoded numbers in pipeline code.

    Usage:
        combiner = FinalCombiner.from_config()
        final_score = combiner.combine(score_components)
    """

    def __init__(
        self,
        base_weights: dict,
        honeypot_penalty_cfg: dict,
        trap_penalty_cfg: dict,
        availability_range: dict,
    ):
        # Verify base weights sum to 1.0
        total = sum(base_weights.values())
        assert abs(total - 1.0) < 1e-6, (
            f"base_weights must sum to 1.0 (got {total}). "
            f"Adjust config/combiner_weights.yaml."
        )
        self.base_weights = base_weights
        self.honeypot_penalty_cfg = honeypot_penalty_cfg
        self.trap_penalty_cfg = trap_penalty_cfg
        self.availability_range = availability_range

        log.info(
            f"FinalCombiner initialized. "
            f"Base weights: {base_weights} (sum={total:.4f}), "
            f"honeypot: {honeypot_penalty_cfg}, "
            f"trap: {trap_penalty_cfg}, "
            f"availability range: [{availability_range['min']}, {availability_range['max']}]"
        )

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "FinalCombiner":
        """Load weights from config/combiner_weights.yaml."""
        if config_path is None:
            config_path = COMBINER_WEIGHTS_PATH

        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            base_weights = cfg.get("base_weights", DEFAULT_BASE_WEIGHTS)
            honeypot_cfg = cfg.get("honeypot_penalty", DEFAULT_HONEYPOT_PENALTY)
            trap_cfg = cfg.get("trap_penalty", DEFAULT_TRAP_PENALTY)
            avail_range = cfg.get("availability_mult", DEFAULT_AVAILABILITY_RANGE)
        else:
            log.warning(f"Config file not found: {config_path}. Using defaults.")
            base_weights = DEFAULT_BASE_WEIGHTS
            honeypot_cfg = DEFAULT_HONEYPOT_PENALTY
            trap_cfg = DEFAULT_TRAP_PENALTY
            avail_range = DEFAULT_AVAILABILITY_RANGE

        return cls(base_weights, honeypot_cfg, trap_cfg, avail_range)

    # ─── Main entry point ─────────────────────────────────────────────────
    def combine(self, components: ScoreComponents) -> tuple[float, dict]:
        """
        Compute final score from components.

        Args:
            components: ScoreComponents dataclass with all sub-scores

        Returns:
            (final_score in [0, max], breakdown dict for reasoning generator)
        """
        # ─── Compute base score (additive, weights sum to 1.0) ────────────
        base_score = (
            self.base_weights["semantic"]      * components.semantic_score +
            self.base_weights["structural"]    * components.structural_score +
            self.base_weights["production"]    * components.production_evidence +
            self.base_weights["esco_coverage"] * components.esco_coverage +
            self.base_weights["seniority"]     * components.seniority_match
        )
        # base_score ∈ [0, 1] (since all components are in [0, 1] and weights sum to 1.0)

        # ─── Apply multipliers (multiplicative kill switches) ─────────────
        final_score = (
            base_score
            * components.honeypot_penalty      # 0.0 or 1.0
            * components.trap_penalty          # 0.3 or 1.0
            * components.availability_mult     # [0.30, 1.20]
        )

        # Clip to valid range [0, max]
        max_score = self.availability_range["max"]
        final_score = max(0.0, min(max_score, final_score))

        breakdown = {
            "base_score": base_score,
            "base_components": {
                "semantic":      {"value": components.semantic_score, "weight": self.base_weights["semantic"]},
                "structural":    {"value": components.structural_score, "weight": self.base_weights["structural"]},
                "production":    {"value": components.production_evidence, "weight": self.base_weights["production"]},
                "esco_coverage": {"value": components.esco_coverage, "weight": self.base_weights["esco_coverage"]},
                "seniority":     {"value": components.seniority_match, "weight": self.base_weights["seniority"]},
            },
            "multipliers": {
                "honeypot_penalty":   {"value": components.honeypot_penalty, "config": self.honeypot_penalty_cfg},
                "trap_penalty":       {"value": components.trap_penalty, "config": self.trap_penalty_cfg},
                "availability_mult":  {"value": components.availability_mult, "range": self.availability_range},
            },
            "final_score": final_score,
        }

        return float(final_score), breakdown

    # ─── Build ScoreComponents from pipeline outputs ──────────────────────
    def build_components(
        self,
        semantic_score: float,
        structural_score: float,
        production_evidence: float,
        esco_coverage: float,
        seniority_match: float,
        filter_verdict: FilterVerdict,
        availability_mult: float,
        axis_scores: Optional[dict] = None,
        availability_breakdown: Optional[dict] = None,
        structural_breakdown: Optional[dict] = None,
        matched_canonical_skills: Optional[list] = None,
    ) -> ScoreComponents:
        """
        Convenience method to build ScoreComponents from individual sub-scores.
        """
        return ScoreComponents(
            semantic_score=semantic_score,
            structural_score=structural_score,
            production_evidence=production_evidence,
            esco_coverage=esco_coverage,
            seniority_match=seniority_match,
            honeypot_penalty=filter_verdict.honeypot_penalty,
            trap_penalty=filter_verdict.trap_penalty,
            availability_mult=availability_mult,
            axis_scores=axis_scores or {},
            availability_breakdown=availability_breakdown or {},
            structural_breakdown=structural_breakdown or {},
            matched_canonical_skills=matched_canonical_skills or [],
            fired_honeypot_rules=filter_verdict.fired_honeypot_rules,
            fired_trap_rules=filter_verdict.fired_trap_rules,
        )

    # ─── Verify equation integrity ────────────────────────────────────────
    def verify_equation(self) -> dict:
        """
        Run integrity checks on the equation. Useful for CI tests.

        Returns:
            dict of {check_name: bool} — all should be True
        """
        checks = {}

        # Check 1: base weights sum to 1.0
        checks["base_weights_sum_to_1"] = abs(sum(self.base_weights.values()) - 1.0) < 1e-6

        # Check 2: honeypot penalty is binary
        checks["honeypot_binary"] = (
            self.honeypot_penalty_cfg["clean"] == 1.0 and
            self.honeypot_penalty_cfg["flagged"] == 0.0
        )

        # Check 3: trap penalty is in [0, 1]
        checks["trap_in_range"] = (
            0.0 <= self.trap_penalty_cfg["flagged"] <= 1.0 and
            self.trap_penalty_cfg["clean"] == 1.0
        )

        # Check 4: availability range is valid
        checks["availability_range_valid"] = (
            0.0 < self.availability_range["min"] < self.availability_range["max"]
        )

        # Check 5: simulate 4 candidate archetypes
        # Ideal: all components = 1.0, no filters, max availability
        ideal = ScoreComponents(
            semantic_score=1.0, structural_score=1.0, production_evidence=1.0,
            esco_coverage=1.0, seniority_match=1.0,
            honeypot_penalty=1.0, trap_penalty=1.0, availability_mult=1.20,
        )
        ideal_score, _ = self.combine(ideal)
        checks["ideal_score_in_range"] = 1.0 <= ideal_score <= 1.20

        # Honeypot: any component values, honeypot fires
        honeypot = ScoreComponents(
            semantic_score=0.95, structural_score=0.85, production_evidence=0.5,
            esco_coverage=0.7, seniority_match=0.8,
            honeypot_penalty=0.0, trap_penalty=1.0, availability_mult=1.10,
        )
        honeypot_score, _ = self.combine(honeypot)
        checks["honeypot_score_is_zero"] = honeypot_score == 0.0

        # Trap: any component values, trap fires
        trap = ScoreComponents(
            semantic_score=0.78, structural_score=0.50, production_evidence=0.10,
            esco_coverage=0.40, seniority_match=0.30,
            honeypot_penalty=1.0, trap_penalty=0.3, availability_mult=0.80,
        )
        trap_score, _ = self.combine(trap)
        # Trap should be: base * 0.3 * 0.8 = base * 0.24
        expected_trap_base = (
            0.55 * 0.78 + 0.20 * 0.50 + 0.10 * 0.10 + 0.10 * 0.40 + 0.05 * 0.30
        )
        expected_trap = expected_trap_base * 0.3 * 0.80
        checks["trap_score_matches_formula"] = abs(trap_score - expected_trap) < 1e-6

        # Dead candidate: perfect on paper but availability floored
        dead = ScoreComponents(
            semantic_score=1.0, structural_score=1.0, production_evidence=1.0,
            esco_coverage=1.0, seniority_match=1.0,
            honeypot_penalty=1.0, trap_penalty=1.0, availability_mult=0.30,
        )
        dead_score, _ = self.combine(dead)
        # Dead: base=1.0 * 1.0 * 1.0 * 0.30 = 0.30
        checks["dead_candidate_score_below_0.4"] = 0.29 <= dead_score <= 0.31

        return checks
