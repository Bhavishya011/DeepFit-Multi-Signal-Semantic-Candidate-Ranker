"""
Pipeline Orchestrator — fast FAISS-only version.

Flow:
    1. Load candidates from .jsonl / .json
    2. Load pre-computed embeddings (or build in-memory TF-IDF fallback)
    3. Load intent (schema + axis embeddings)
    4. Phase 1: Hard filters (Module 3 — honeypot + trap)
    5. Phase 2: FAISS-only coarse recall → top-1000
       (BM25 skipped for speed — it takes 1-2 hrs on CPU for 100K)
    6. Phase 3: Multi-axis semantic scoring (cosine only, no cross-encoder)
    7. Phase 4: Features (Module 4/6 — production evidence, ESCO, structural, seniority)
    8. Phase 5: Availability multiplier (Module 5)
    9. Phase 6: Final combiner (Module 9 — THE equation)
    10. Phase 7: Reasoning generator (Module 10)
    11. Sort by final_score desc, candidate_id asc (tie-break), take top-100
    12. Write CSV

Time budget (5min wall, CPU, no network):
    Load: ~10s
    Phase 1 (filters): ~3s
    Phase 2 (FAISS recall): ~5s
    Phase 3-6 (score top-1000): ~5s
    Phase 7 (reasoning): ~3s
    Write CSV: ~1s
    Total: ~30s (270s safety margin)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Candidate, ScoreComponents, RankedCandidate, Intent
from .intent import load_intent, load_intent_schema, get_jd_required_canonical_skills
from .encoder import MultiFieldEncoder, load_embeddings
from .filters import HoneypotScorer
from .features import SkillTaxonomyResolver, extract_all_features
from .availability import AvailabilityScorer
from .combiner import FinalCombiner
from .reasoning import ReasoningGenerator
from .recall import DenseIndex
from .rerank import MultiAxisSemanticScorer

log = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end ranking pipeline (fast FAISS-only version).

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

    @classmethod
    def build(cls) -> "Pipeline":
        """Build the pipeline with all components loaded from config."""
        log.info("Building pipeline...")

        # Load intent (schema + pre-computed axis embeddings if available)
        # If no pre-computed embeddings, axis_embeddings will be empty {}
        # and will be computed by _build_tfidf_embeddings() during run()
        intent = load_intent()
        honeypot_scorer = HoneypotScorer.from_config()
        skill_resolver = SkillTaxonomyResolver.from_config()
        availability_scorer = AvailabilityScorer.from_config()
        combiner = FinalCombiner.from_config()
        reasoning_generator = ReasoningGenerator.from_config()
        jd_required_skills = get_jd_required_canonical_skills(intent.schema)

        log.info(f"Loaded intent with {len(intent.axis_embeddings)} axis embeddings")
        log.info(f"JD requires {len(jd_required_skills)} canonical skills")
        if not intent.axis_embeddings:
            log.info("No pre-computed intent embeddings — will compute via TF-IDF fallback during run()")

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
        verbose: bool = True,
    ) -> list[RankedCandidate]:
        """
        Run the full ranking pipeline (fast FAISS-only version).

        Args:
            candidates: list of Candidate objects
            candidate_embeddings: pre-computed embeddings {cid: {field: np.ndarray}}
                                  If None, builds in-memory TF-IDF embeddings
            top_n: number of candidates to return (default 100, per spec)
            coarse_recall_k: top-K from FAISS recall (default 1000)
            verbose: print per-stage timing

        Returns:
            list of RankedCandidate sorted by score descending
        """
        t_start = time.time()
        if verbose:
            log.info(f"\n{'='*70}")
            log.info(f"  DeepFit Pipeline — {len(candidates)} candidates")
            log.info(f"{'='*70}")

        # ─── Build embeddings if not provided ────────────────────────────
        if candidate_embeddings is None:
            t0 = time.time()
            emb_path = Path(__file__).parent.parent / "artifacts" / "candidate_embeddings.npz"
            if emb_path.exists():
                try:
                    candidate_embeddings = load_embeddings(emb_path)
                    if verbose:
                        log.info(f"[{time.time() - t0:.1f}s] Loaded {len(candidate_embeddings)} pre-computed embeddings")
                except Exception as e:
                    log.warning(f"Failed to load pre-computed embeddings: {e}")
                    log.warning("Falling back to in-memory TF-IDF embeddings")
                    candidate_embeddings = self._build_tfidf_embeddings(candidates)
                    if verbose:
                        log.info(f"[{time.time() - t0:.1f}s] Built {len(candidate_embeddings)} in-memory TF-IDF embeddings")
            else:
                log.info("No pre-computed embeddings found. Building in-memory TF-IDF embeddings...")
                candidate_embeddings = self._build_tfidf_embeddings(candidates)
                if verbose:
                    log.info(f"[{time.time() - t0:.1f}s] Built {len(candidate_embeddings)} in-memory TF-IDF embeddings")

        # ─── Phase 1: Hard filters (honeypot + trap) ──────────────────────
        t1 = time.time()
        filter_verdicts = {}
        survived_candidates = []
        honeypot_count = 0
        trap_count = 0
        for c in candidates:
            verdict = self.honeypot_scorer.score(c)
            filter_verdicts[c.candidate_id] = verdict
            if verdict.passes:
                survived_candidates.append(c)
                if verdict.is_trap:
                    trap_count += 1
            else:
                honeypot_count += 1
        if verbose:
            log.info(f"[{time.time() - t1:.1f}s] Phase 1 — Filters: "
                     f"{honeypot_count} honeypots removed, {trap_count} traps flagged, "
                     f"{len(survived_candidates)} survived")

        # ─── Phase 2: FAISS-only coarse recall ────────────────────────────
        t2 = time.time()
        # If intent axis embeddings are missing (e.g., pre-computed candidate
        # embeddings exist but intent_embeddings.npz doesn't), compute them
        # using the same TF-IDF approach
        if not self.intent.axis_embeddings:
            log.info("Intent axis embeddings missing — computing via TF-IDF fallback")
            # Build a temporary TF-IDF encoder and compute intent axes
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from .intent import get_all_axis_query_texts
            from .encoder import extract_title_text, extract_summary_text, extract_career_text, extract_skills_text

            axes = get_all_axis_query_texts(self.intent.schema)
            jd_path = Path(__file__).parent.parent / "job_description.md"
            jd_text = jd_path.read_text() if jd_path.exists() else ""

            # Build corpus from candidate texts + JD + axes
            corpus = []
            for c in candidates[:5000]:  # sample for speed
                corpus.append(extract_title_text(c))
                corpus.append(extract_summary_text(c)[:500])
            corpus.extend(list(axes.values()))
            corpus.extend([jd_text] * 5)

            vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words="english",
                                  max_features=10000, sublinear_tf=True).fit(corpus)
            # Match DIM to candidate embeddings
            sample_emb = next(iter(candidate_embeddings.values()))
            DIM = sample_emb["combined"].shape[0]
            svd = TruncatedSVD(n_components=min(DIM, len(corpus) - 1), random_state=42).fit(vec.transform(corpus))

            def enc(t):
                if not t: return np.zeros(svd.n_components, dtype=np.float32)
                x = svd.transform(vec.transform([t]))[0]
                n = np.linalg.norm(x)
                return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)

            self.intent.axis_embeddings = {n: enc(t) for n, t in axes.items()}
            log.info(f"  Computed {len(self.intent.axis_embeddings)} intent axes (dim={svd.n_components})")

        if self.intent.axis_embeddings:
            query_emb = np.mean(np.stack(list(self.intent.axis_embeddings.values())), axis=0)
            query_emb = query_emb / max(np.linalg.norm(query_emb), 1e-9)
        else:
            log.error("No intent axis embeddings available — cannot proceed with coarse recall")
            return []

        ids = [c.candidate_id for c in survived_candidates]
        try:
            combined = np.stack([candidate_embeddings[cid]["combined"] for cid in ids])
        except KeyError as e:
            log.error(f"Candidate {e} not found in embeddings. Falling back to TF-IDF.")
            candidate_embeddings = self._build_tfidf_embeddings(candidates)
            combined = np.stack([candidate_embeddings[cid]["combined"] for cid in ids])

        dense = DenseIndex.from_embeddings(ids, combined)
        recall_k = min(coarse_recall_k, len(survived_candidates))
        _, dense_indices = dense.search(query_emb, k=recall_k)
        top_ids = [ids[i] for i in dense_indices]
        if verbose:
            log.info(f"[{time.time() - t2:.1f}s] Phase 2 — FAISS recall: "
                     f"top-{len(top_ids)} from {len(survived_candidates)}")

        # ─── Phase 3-6: Features + Availability + Combiner ────────────────
        t3 = time.time()
        semantic_scorer = MultiAxisSemanticScorer(intent=self.intent)
        cand_by_id = {c.candidate_id: c for c in candidates}
        ranked_results = []

        for cid in top_ids:
            candidate = cand_by_id.get(cid)
            if candidate is None:
                continue

            c_emb = candidate_embeddings.get(cid, {})
            if not c_emb:
                continue

            matched_skills = self.skill_resolver.resolve(candidate)
            esco_score, _ = self.skill_resolver.coverage_score(matched_skills, self.jd_required_skills)

            semantic_score, axis_scores = semantic_scorer.score(
                candidate=candidate,
                candidate_embeddings=c_emb,
                cross_enc_score=0.5,
                esco_coverage_score=esco_score,
            )

            features = extract_all_features(
                candidate,
                skill_resolver=self.skill_resolver,
                jd_required_skills=self.jd_required_skills,
                cosine_title_sim=axis_scores.get("title_fit", 0.0),
            )

            avail_mult, avail_breakdown = self.availability_scorer.score(candidate)

            components = self.combiner.build_components(
                semantic_score=semantic_score,
                structural_score=features["structural_score"],
                production_evidence=features["production_evidence"],
                esco_coverage=features["esco_coverage"],
                seniority_match=features["seniority_match"],
                filter_verdict=filter_verdicts[cid],
                availability_mult=avail_mult,
                axis_scores=axis_scores,
                availability_breakdown=avail_breakdown,
                structural_breakdown=features["structural_breakdown"],
                matched_canonical_skills=features["matched_canonical_skills"],
            )

            final_score, breakdown = self.combiner.combine(components)

            ranked_results.append({
                "candidate_id": cid,
                "score": final_score,
                "components": components,
                "breakdown": breakdown,
            })

        if verbose:
            log.info(f"[{time.time() - t3:.1f}s] Phase 3-6 — Features + Availability + Combiner: "
                     f"scored {len(ranked_results)} candidates")

        # ─── Sort by final score desc, candidate_id asc (tie-break) ────────
        # IMPORTANT: sort by ROUNDED score (4 decimal places, matching CSV output)
        # so that ties in the CSV output are broken by candidate_id ascending
        # (per spec Section 3: "If two candidates have the same score, you must
        # still assign unique ranks. Break score ties deterministically using a
        # secondary signal from your model, or by candidate_id ascending.")
        ranked_results.sort(key=lambda r: (-round(r["score"], 4), r["candidate_id"]))

        # ─── Phase 7: Reasoning generator + assign ranks ──────────────────
        t4 = time.time()
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
            log.info(f"[{time.time() - t4:.1f}s] Phase 7 — Reasoning: generated for top-{len(final_results)}")

        total_time = time.time() - t_start
        if verbose:
            log.info(f"\n{'='*70}")
            log.info(f"  Pipeline complete in {total_time:.1f}s")
            if final_results:
                log.info(f"  Top score: {final_results[0].score:.4f} ({final_results[0].candidate_id})")
                log.info(f"  {top_n}th score: {final_results[-1].score:.4f} ({final_results[-1].candidate_id})")
            log.info(f"{'='*70}")

        return final_results

    def _build_tfidf_embeddings(self, candidates: list[Candidate]) -> dict[str, dict[str, np.ndarray]]:
        """Build in-memory TF-IDF embeddings (fallback when .npz unavailable)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from .encoder import (
            extract_title_text, extract_summary_text,
            extract_career_text, extract_skills_text
        )

        texts = {"title": [], "summary": [], "career": [], "skills": []}
        for c in candidates:
            texts["title"].append(extract_title_text(c))
            texts["summary"].append(extract_summary_text(c)[:800])
            texts["career"].append(extract_career_text(c)[:800])
            texts["skills"].append(extract_skills_text(c))

        from .intent import get_all_axis_query_texts
        axes = get_all_axis_query_texts(self.intent.schema)
        jd_path = Path(__file__).parent.parent / "job_description.md"
        jd_text = jd_path.read_text() if jd_path.exists() else ""

        corpus = (texts["title"][:15000] + texts["summary"][:15000] + texts["career"][:15000] +
                  texts["skills"][:15000] + list(axes.values()) + [jd_text] * 5)
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=2,
                              max_features=20000, stop_words="english", sublinear_tf=True).fit(corpus)
        DIM = min(256, vec.transform(corpus[:2000]).shape[1] - 1)
        svd = TruncatedSVD(n_components=DIM, random_state=42).fit(vec.transform(corpus[:20000]))

        field_weights = {"title": 3.0, "summary": 1.5, "career": 2.0, "skills": 1.0}
        total_w = sum(field_weights.values())
        field_weights = {k: v / total_w for k, v in field_weights.items()}

        field_vecs = {}
        for field in ["title", "summary", "career", "skills"]:
            v = svd.transform(vec.transform(texts[field]))
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n == 0] = 1
            field_vecs[field] = (v / n).astype(np.float32)

        embeddings = {}
        for i, c in enumerate(candidates):
            emb = {f: field_vecs[f][i] for f in ["title", "summary", "career", "skills"]}
            combined = np.zeros(DIM, dtype=np.float32)
            for f in ["title", "summary", "career", "skills"]:
                combined += field_weights[f] * emb[f]
            n = np.linalg.norm(combined)
            if n > 0:
                combined = combined / n
            emb["combined"] = combined
            embeddings[c.candidate_id] = emb

        def enc(t):
            if not t:
                return np.zeros(DIM, dtype=np.float32)
            x = svd.transform(vec.transform([t]))[0]
            n = np.linalg.norm(x)
            return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)

        self.intent.axis_embeddings = {n: enc(t) for n, t in axes.items()}

        return embeddings

    def write_csv(self, results: list[RankedCandidate], output_path: Path):
        """Write ranked results to CSV per submission spec."""
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
