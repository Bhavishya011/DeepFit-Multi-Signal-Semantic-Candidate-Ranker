"""
Pipeline Orchestrator — ties all modules together.

Flow:
    1. Load candidates from .jsonl / .json
    2. Load pre-computed embeddings (or compute if missing)
    3. Load intent (schema + axis embeddings)
    4. Phase 1: Hard filters (Module 3 — honeypot + trap)
    5. Phase 2: Coarse recall (Module 7 — BM25 + FAISS + RRF) → top-1000
    6. Phase 3: Rerank (Module 8 — cross-encoder + multi-axis semantic) → top-500
    7. Phase 4: Features (Module 4/6 — production evidence, ESCO, structural, seniority)
    8. Phase 5: Availability multiplier (Module 5)
    9. Phase 6: Final combiner (Module 9 — THE equation)
    10. Phase 7: Reasoning generator (Module 10)
    11. Sort by final_score, take top-100, assign ranks
    12. Write CSV

Time budget (5min wall, CPU, no network):
    Load: ~5s
    Phase 1 (filters): ~3s
    Phase 2 (recall): ~10s
    Phase 3 (rerank): ~10s
    Phase 4 (features): ~15s
    Phase 5 (availability): ~1s
    Phase 6 (combiner): ~2s
    Phase 7 (reasoning): ~3s
    Write CSV: ~1s
    Total: ~50s (250s safety margin)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Candidate, ScoreComponents, RankedCandidate
from .intent import load_intent, load_intent_schema, get_jd_required_canonical_skills, Intent
from .encoder import MultiFieldEncoder, load_embeddings
from .filters import HoneypotScorer
from .features import SkillTaxonomyResolver, extract_all_features
from .availability import AvailabilityScorer
from .recall import CoarseRecall, build_jd_query
from .rerank import Reranker
from .combiner import FinalCombiner
from .reasoning import ReasoningGenerator

log = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end ranking pipeline.

    Usage:
        pipeline = Pipeline.build()
        results = pipeline.run(candidates)
        pipeline.write_csv(results, "submission.csv")
    """

    def __init__(
        self,
        intent: Intent,
        honeypot_scorer: HoneypotScorer,
        skill_resolver: SkillTaxonomyResolver,
        availability_scorer: AvailabilityScorer,
        combiner: FinalCombiner,
        reasoning_generator: ReasoningGenerator,
        jd_required_skills: list[str],
    ):
        self.intent = intent
        self.honeypot_scorer = honeypot_scorer
        self.skill_resolver = skill_resolver
        self.availability_scorer = availability_scorer
        self.combiner = combiner
        self.reasoning_generator = reasoning_generator
        self.jd_required_skills = jd_required_skills
        # These are built lazily in run()
        self._reranker: Optional[Reranker] = None

    @classmethod
    def build(cls) -> "Pipeline":
        """Build the pipeline with all components loaded from config."""
        log.info("Building pipeline...")

        # Load intent (schema + axis embeddings)
        intent = load_intent()
        log.info(f"Loaded intent with {len(intent.axis_embeddings)} axis embeddings")

        # Load honeypot scorer
        honeypot_scorer = HoneypotScorer.from_config()
        log.info("Loaded honeypot scorer")

        # Load skill resolver
        skill_resolver = SkillTaxonomyResolver.from_config()
        log.info("Loaded skill taxonomy resolver")

        # Load availability scorer
        availability_scorer = AvailabilityScorer.from_config()
        log.info("Loaded availability scorer")

        # Load final combiner
        combiner = FinalCombiner.from_config()
        log.info("Loaded final combiner")

        # Load reasoning generator
        reasoning_generator = ReasoningGenerator.from_config()
        log.info("Loaded reasoning generator")

        # Get JD-required skills
        jd_required_skills = get_jd_required_canonical_skills(intent.schema)
        log.info(f"JD requires {len(jd_required_skills)} canonical skills: {jd_required_skills}")

        return cls(
            intent=intent,
            honeypot_scorer=honeypot_scorer,
            skill_resolver=skill_resolver,
            availability_scorer=availability_scorer,
            combiner=combiner,
            reasoning_generator=reasoning_generator,
            jd_required_skills=jd_required_skills,
        )

    def run(
        self,
        candidates: list[Candidate],
        candidate_embeddings: Optional[dict[str, dict[str, np.ndarray]]] = None,
        top_n: int = 100,
        coarse_recall_k: int = 1000,
        rerank_k: int = 500,
        verbose: bool = True,
    ) -> list[RankedCandidate]:
        """
        Run the full ranking pipeline.

        Args:
            candidates: list of Candidate objects
            candidate_embeddings: pre-computed embeddings {cid: {field: np.ndarray}}
                                  If None, will try to load from artifacts/ or compute
            top_n: number of candidates to return (default 100, per spec)
            coarse_recall_k: top-K from coarse recall (default 1000)
            rerank_k: top-K to rerank (default 500)
            verbose: print per-stage timing

        Returns:
            list of RankedCandidate sorted by score descending
        """
        t_start = time.time()
        total_n = len(candidates)
        if verbose:
            log.info(f"\n{'='*70}")
            log.info(f"  DeepFit Pipeline — {total_n} candidates")
            log.info(f"{'='*70}")

        # ─── Load embeddings if not provided ──────────────────────────────
        if candidate_embeddings is None:
            t0 = time.time()
            emb_path = Path(__file__).parent.parent / "artifacts" / "candidate_embeddings.npz"
            if emb_path.exists():
                candidate_embeddings = load_embeddings(emb_path)
                if verbose:
                    log.info(f"[{time.time() - t0:.1f}s] Loaded {len(candidate_embeddings)} pre-computed embeddings")
            else:
                # Compute on the fly (will be slow for 100K — should use precompute script)
                log.warning("No pre-computed embeddings found. Computing on-the-fly (slow for large pools).")
                encoder = MultiFieldEncoder.from_auto()
                candidate_embeddings = encoder.encode_candidates(candidates)
                if verbose:
                    log.info(f"[{time.time() - t0:.1f}s] Computed {len(candidate_embeddings)} embeddings on-the-fly")

        # ─── Phase 1: Hard filters (honeypot + trap) ──────────────────────
        t1 = time.time()
        filter_verdicts = {}
        survived_candidates = []
        honeypot_count = 0
        trap_count = 0
        for c in candidates:
            verdict = self.honeypot_scorer.score(c)
            filter_verdicts[c.candidate_id] = verdict
            if verdict.passes:  # not a hard honeypot
                survived_candidates.append(c)
                if verdict.is_trap:
                    trap_count += 1
            else:
                honeypot_count += 1
        if verbose:
            log.info(f"[{time.time() - t1:.1f}s] Phase 1 — Filters: "
                     f"{honeypot_count} honeypots removed, {trap_count} traps flagged, "
                     f"{len(survived_candidates)} survived")

        # ─── Phase 2: Coarse recall (BM25 + FAISS + RRF) ──────────────────
        t2 = time.time()
        # Build query from JD intent
        query_text = build_jd_query(self.intent.schema)
        # Build query embedding: average of intent axis embeddings
        if self.intent.axis_embeddings:
            query_emb = np.mean(np.stack(list(self.intent.axis_embeddings.values())), axis=0)
            query_emb = query_emb / max(np.linalg.norm(query_emb), 1e-9)
        else:
            log.error("No intent axis embeddings available — cannot proceed with coarse recall")
            return []

        # Build coarse recall system on survived candidates
        recall = CoarseRecall.build(survived_candidates, candidate_embeddings)
        recall_k = min(coarse_recall_k, len(survived_candidates))
        top_ids, top_scores = recall.recall(query_text, query_emb, k=recall_k)
        top_recall_candidates = [c for c in survived_candidates if c.candidate_id in top_ids]
        # Sort by recall order
        top_recall_candidates.sort(key=lambda c: top_ids.index(c.candidate_id))
        if verbose:
            log.info(f"[{time.time() - t2:.1f}s] Phase 2 — Coarse recall: "
                     f"top-{len(top_recall_candidates)} from {len(survived_candidates)}")

        # ─── Phase 3: Rerank (cross-encoder + multi-axis semantic) ────────
        t3 = time.time()
        if self._reranker is None:
            self._reranker = Reranker.from_intent(self.intent)
        rerank_results = self._reranker.rerank(
            candidates=top_recall_candidates,
            candidate_embeddings=candidate_embeddings,
            intent=self.intent,
            top_k=min(rerank_k, len(top_recall_candidates)),
        )
        if verbose:
            log.info(f"[{time.time() - t3:.1f}s] Phase 3 — Rerank: "
                     f"scored {len(rerank_results)} candidates")

        # ─── Phase 4-6: Features + Availability + Combiner ────────────────
        t4 = time.time()
        ranked_results = []
        cand_by_id = {c.candidate_id: c for c in candidates}

        for result in rerank_results:
            cid = result["candidate_id"]
            candidate = cand_by_id.get(cid)
            if candidate is None:
                continue

            # Extract all features (production, ESCO, structural, seniority)
            features = extract_all_features(
                candidate,
                skill_resolver=self.skill_resolver,
                jd_required_skills=self.jd_required_skills,
                cosine_title_sim=result["axis_scores"].get("title_fit", 0.0),
            )

            # Availability multiplier
            avail_mult, avail_breakdown = self.availability_scorer.score(candidate)

            # Build score components
            components = self.combiner.build_components(
                semantic_score=result["semantic_score"],
                structural_score=features["structural_score"],
                production_evidence=features["production_evidence"],
                esco_coverage=features["esco_coverage"],
                seniority_match=features["seniority_match"],
                filter_verdict=filter_verdicts[cid],
                availability_mult=avail_mult,
                axis_scores=result["axis_scores"],
                availability_breakdown=avail_breakdown,
                structural_breakdown=features["structural_breakdown"],
                matched_canonical_skills=features["matched_canonical_skills"],
            )

            # Compute final score
            final_score, breakdown = self.combiner.combine(components)

            ranked_results.append({
                "candidate_id": cid,
                "score": final_score,
                "components": components,
                "breakdown": breakdown,
                "features": features,
            })

        if verbose:
            log.info(f"[{time.time() - t4:.1f}s] Phase 4-6 — Features + Availability + Combiner: "
                     f"scored {len(ranked_results)} candidates")

        # ─── Sort by final score descending ───────────────────────────────
        ranked_results.sort(key=lambda r: -r["score"])

        # ─── Phase 7: Reasoning generator + assign ranks ──────────────────
        t5 = time.time()
        final_results = []
        for rank, result in enumerate(ranked_results[:top_n], start=1):
            cid = result["candidate_id"]
            candidate = cand_by_id[cid]

            reasoning = self.reasoning_generator.generate(
                candidate=candidate,
                rank=rank,
                components=result["components"],
                breakdown=result["breakdown"],
            )

            final_results.append(RankedCandidate(
                candidate_id=cid,
                rank=rank,
                score=result["score"],
                reasoning=reasoning,
                components=result["components"],
            ))
        if verbose:
            log.info(f"[{time.time() - t5:.1f}s] Phase 7 — Reasoning: generated for top-{len(final_results)}")

        total_time = time.time() - t_start
        if verbose:
            log.info(f"\n{'='*70}")
            log.info(f"  Pipeline complete in {total_time:.1f}s")
            log.info(f"  Top score: {final_results[0].score:.4f} ({final_results[0].candidate_id})")
            log.info(f"  100th score: {final_results[-1].score:.4f} ({final_results[-1].candidate_id})")
            log.info(f"{'='*70}")

        return final_results

    # ─── CSV writer ───────────────────────────────────────────────────────
    def write_csv(self, results: list[RankedCandidate], output_path: Path):
        """
        Write ranked results to CSV per submission spec:
            candidate_id,rank,score,reasoning

        - Exactly 100 data rows (+ 1 header)
        - Ranks 1-100, each appearing exactly once
        - Scores non-increasing by rank
        - Tie-break: candidate_id ascending
        """
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for r in results:
                writer.writerow([
                    r.candidate_id,
                    r.rank,
                    f"{r.score:.4f}",
                    r.reasoning,
                ])

        log.info(f"Wrote {len(results)} ranked candidates to {output_path}")


# ─── Convenience: load candidates from file ───────────────────────────────
def load_candidates_from_file(path: Path) -> list[Candidate]:
    """Load candidates from .json (list) or .jsonl / .jsonl.gz (one per line)."""
    import gzip

    if path.suffix == ".json":
        with open(path) as f:
            records = json.load(f)
    elif path.suffix == ".jsonl":
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return [Candidate.from_dict(r) for r in records]
