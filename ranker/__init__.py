"""
DeepFit ranker package.

Modules:
    types       — shared dataclasses (Candidate, Intent, ScoreComponents, FilterVerdict)
    intent      — Module 1: JD-Intent loader
    encoder     — Module 2: Multi-field candidate encoder
    filters     — Module 3: Honeypot + trap consistency scorer
    features    — Module 4: Production-evidence + ESCO coverage + structural
    availability — Module 5: Behavioral availability multiplier
    recall      — Module 6: Field-weighted BM25 + FAISS + RRF
    rerank      — Module 7: Cross-encoder + multi-axis semantic
    combiner    — Module 8: Final score equation
    reasoning   — Module 9: Evidence-grounded reasoning generator
    pipeline    — orchestrator
    loader      — fast artifact loading
"""

from .types import Candidate, Intent, ScoreComponents, FilterVerdict

__version__ = "0.1.0"
__all__ = ["Candidate", "Intent", "ScoreComponents", "FilterVerdict"]
