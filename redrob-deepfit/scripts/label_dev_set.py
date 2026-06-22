#!/usr/bin/env python3
"""
Interactive dev-set labeler for the 50-candidate hand-labeling task.

Usage:
    python scripts/label_dev_set.py
    python scripts/label_dev_set.py --resume        # skip already-labeled
    python scripts/label_dev_set.py --candidate CAND_0000007  # label specific
    python scripts/label_dev_set.py --summary       # print summary table of all 50

Workflow:
    1. Loads tests/dev_set/dev_candidates.json (50 candidates)
    2. Loads tests/dev_set/dev_labels.json (existing labels, can be empty)
    3. For each unlabeled candidate, prints a COMPACT SUMMARY + heuristic tier suggestion
    4. Prompts: tier (0-3) + optional note
    5. Saves incrementally to dev_labels.json (so you can stop and resume)

Tier definitions:
    3 = TOP-10 QUALITY. Ideal AI Engineer: 5-9 YOE at product company, shipped
        retrieval/ranking to real users, active on Redrob (last_active < 30d,
        response_rate > 0.5), Pune/Noida/Hyderabad/Mumbai/Delhi NCR location,
        notice < 30d. Strong semantic + strong engagement.
    2 = RELEVANT. Solid match with 1 minor concern (e.g. slightly outside YOE
        band, or notice 60d, or location tier-2). Should appear in top-50.
    1 = MARGINAL. Adjacent skills, missing key requirement (e.g. Backend
        Engineer with ML side projects). May appear in ranks 50-100.
    0 = IRRELEVANT. Honeypot, trap, or hard disqualifier (services-only,
        research-only, marketing manager with AI skills, inactive > 180d,
        inverted salary range, etc.).
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────
DEV_CANDIDATES = Path(__file__).parent.parent / "tests" / "dev_set" / "dev_candidates.json"
DEV_LABELS = Path(__file__).parent.parent / "tests" / "dev_set" / "dev_labels.json"

# ─── Reference data for heuristics ────────────────────────────────────────
TIER_1_LOCATIONS = {"Pune", "Noida", "Hyderabad", "Mumbai", "Delhi NCR",
                    "Bangalore", "Bengaluru", "Gurgaon", "Faridabad", "New Delhi"}

AI_ENGINEER_TITLES = {
    "AI Engineer", "ML Engineer", "Senior AI Engineer", "Senior ML Engineer",
    "Senior Machine Learning Engineer", "Machine Learning Engineer",
    "Search Engineer", "Recommendation Systems Engineer", "Ranking Engineer",
    "Retrieval Engineer", "Senior Search Engineer", "Applied Scientist",
    "ML Platform Engineer", "Data Scientist", "Senior Data Scientist",
}

MISMATCHED_TITLES = {
    "Marketing Manager", "HR Manager", "Operations Manager", "Accountant",
    "Mechanical Engineer", "Civil Engineer", "Sales Executive",
    "Customer Support", "Content Writer", "Graphic Designer",
    "Project Manager", "Business Analyst",
}

SERVICES_COMPANIES = {
    "TCS", "Tata Consultancy Services", "Infosys", "Wipro", "Accenture",
    "Cognizant", "Capgemini", "Tech Mahindra", "HCL Technologies", "HCLTech",
    "LTI", "Larsen & Toubro Infotech", "Mindtree", "Mphasis", "NTT Data",
    "Genpact", "IBM Global Services", "DXC Technology", "Atos",
}

AI_SKILL_KEYWORDS = {
    "LLM", "NLP", "ML", "Deep Learning", "PyTorch", "TensorFlow", "RAG",
    "Embedding", "Embeddings", "Fine-tuning", "Fine-tuning LLMs", "LoRA",
    "QLoRA", "PEFT", "Vector Database", "Pinecone", "Weaviate", "Qdrant",
    "Milvus", "FAISS", "LangChain", "LlamaIndex", "Recommendation Systems",
    "Search", "Ranking", "Retrieval", "Transformer", "BERT", "GPT",
}

PRE_LLM_ML_KEYWORDS = {
    "XGBoost", "LightGBM", "scikit-learn", "sklearn", "TensorFlow",
    "PyTorch", "Keras", "SVM", "Random Forest", "Gradient Boosting",
    "Statistical Modeling", "Regression",
}


def load_candidates():
    with open(DEV_CANDIDATES) as f:
        return json.load(f)


def load_labels():
    if not DEV_LABELS.exists():
        return {"labels": {}}
    with open(DEV_LABELS) as f:
        data = json.load(f)
    if "labels" not in data:
        data["labels"] = {}
    return data


def save_labels(data):
    data["_last_updated"] = datetime.now().isoformat()
    with open(DEV_LABELS, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def days_since(date_str, ref=date(2026, 6, 20)):
    try:
        d = date.fromisoformat(date_str)
        return (ref - d).days
    except Exception:
        return -1


def compact_skills(skills, max_n=8):
    """Show top-N skills by endorsements, with proficiency + duration."""
    if not skills:
        return "(none)"
    sorted_skills = sorted(skills, key=lambda s: -s.get("endorsements", 0))[:max_n]
    parts = []
    for s in sorted_skills:
        prof = s.get("proficiency", "?")[:4]
        end = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        parts.append(f"{s['name']}({prof[:3]}/e{end}/{dur}mo)")
    return ", ".join(parts)


def heuristic_tier(c):
    """
    Suggest a tier 0-3 based on simple rules.
    The user overrides if needed. Returns (tier, reasons).
    """
    profile = c["profile"]
    signals = c["redrob_signals"]
    skills = c.get("skills", [])
    career = c.get("career_history", [])

    reasons = []
    tier = 2  # default — assume relevant

    # ─── Hard disqualifiers → tier 0 ──────────────────────────────────────
    title = profile.get("current_title", "")

    # Honeypot: any single skill duration > YOE * 12 (real impossibility)
    # Allow 6-month grace for parallel learning/self-study before first job
    yoe_months = profile.get("years_of_experience", 0) * 12
    if yoe_months > 0:
        impossible_skills = [s for s in skills
                             if s.get("duration_months", 0) > yoe_months + 6]
        if impossible_skills:
            tier = 0
            worst = max(impossible_skills, key=lambda s: s.get("duration_months", 0))
            reasons.append(f"HONEYPOT: skill '{worst['name']}' duration {worst['duration_months']}mo > YOE {yoe_months:.0f}mo")

    # Honeypot: expert in 5+ skills with 0 endorsements
    expert_zero = [s for s in skills if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0]
    if len(expert_zero) >= 5:
        tier = 0
        reasons.append(f"HONEYPOT: {len(expert_zero)} expert skills with 0 endorsements")

    # Honeypot: salary range inverted
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("min", 0) > sal.get("max", 0):
        tier = 0
        reasons.append(f"HONEYPOT: salary min({sal.get('min')}) > max({sal.get('max')})")

    # Honeypot: inactive > 180d but high response rate
    days = days_since(signals.get("last_active_date", ""))
    if days > 180 and signals.get("recruiter_response_rate", 0) > 0.8:
        tier = 0
        reasons.append(f"HONEYPOT: inactive {days}d but response rate {signals['recruiter_response_rate']}")

    # Trap: mismatched title with 5+ AI skills
    if title in MISMATCHED_TITLES:
        ai_skills = [s for s in skills if s["name"] in AI_SKILL_KEYWORDS]
        if len(ai_skills) >= 5:
            tier = 0
            reasons.append(f"TRAP: {title} with {len(ai_skills)} AI skills (keyword stuffer)")

    # Trap: services-only career
    companies = [j.get("company", "") for j in career]
    if companies and all(any(svc in comp for svc in SERVICES_COMPANIES) for comp in companies):
        tier = 0
        reasons.append(f"TRAP: services-only career ({', '.join(companies[:2])})")

    if tier == 0:
        return tier, reasons

    # ─── Positive signals → bump up ───────────────────────────────────────
    if title in AI_ENGINEER_TITLES:
        tier = max(tier, 3)
        reasons.append(f"GOOD title: {title}")

    # Active on Redrob
    if 0 <= days <= 30:
        tier = max(tier, 2)
        reasons.append(f"active {days}d ago")
    elif days > 90:
        tier = min(tier, 1)
        reasons.append(f"stale {days}d ago")

    # Response rate
    rr = signals.get("recruiter_response_rate", 0)
    if rr >= 0.5:
        tier = max(tier, 2)
        reasons.append(f"response_rate {rr}")
    elif rr < 0.1 and rr >= 0:
        tier = min(tier, 1)
        reasons.append(f"low response_rate {rr}")

    # YOE band 5-9
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        reasons.append(f"YOE {yoe} in ideal band")
    elif yoe < 3:
        tier = min(tier, 1)
        reasons.append(f"YOE {yoe} too low")
    elif yoe > 12:
        tier = min(tier, 2)
        reasons.append(f"YOE {yoe} above band")

    # Location
    loc = profile.get("location", "")
    if any(t in loc for t in TIER_1_LOCATIONS):
        reasons.append(f"location {loc} (tier-1)")
    elif "India" in profile.get("country", "") or any(s in loc for s in ["Chennai", "Kolkata", "Ahmedabad"]):
        reasons.append(f"location {loc} (tier-2)")
    else:
        tier = min(tier, 1)
        reasons.append(f"location {loc} (outside India)")

    # Notice period
    notice = signals.get("notice_period_days", 0)
    if notice <= 30:
        reasons.append(f"notice {notice}d (ideal)")
    elif notice <= 60:
        reasons.append(f"notice {notice}d (buyout ok)")
    elif notice > 90:
        tier = min(tier, 1)
        reasons.append(f"notice {notice}d (high bar)")

    # Open to work
    if signals.get("open_to_work_flag"):
        reasons.append("open_to_work ✓")

    # GitHub activity
    gh = signals.get("github_activity_score", -1)
    if gh >= 30:
        reasons.append(f"github {gh}")
    elif gh == -1:
        reasons.append("no github")

    # Pre-LLM ML experience (JD wants pre-LLM depth)
    has_pre_llm = any(s["name"] in PRE_LLM_ML_KEYWORDS for s in skills)
    if has_pre_llm:
        reasons.append("has pre-LLM ML ✓")

    # Production evidence in career descriptions
    career_text = " ".join(j.get("description", "") for j in career).lower()
    prod_signals = sum(1 for p in ["deployed", "production", "served", "shipped", "users", "traffic", "a/b", "rollout"] if p in career_text)
    if prod_signals >= 3:
        reasons.append(f"production evidence ({prod_signals} hits)")
    elif prod_signals == 0:
        tier = min(tier, 2)
        reasons.append("no production evidence in career text")

    return tier, reasons


def print_candidate_summary(c, suggested_tier, reasons):
    """Print a compact, scannable summary of one candidate."""
    cid = c["candidate_id"]
    p = c["profile"]
    s = c["redrob_signals"]
    career = c.get("career_history", [])

    print("\n" + "─" * 78)
    print(f"  {cid}  —  {p['current_title']} @ {p['current_company']} ({p['current_industry']})")
    print("─" * 78)
    print(f"  Name:       {p.get('anonymized_name', '?')}")
    print(f"  Headline:   {p.get('headline', '?')}")
    print(f"  Location:   {p.get('location', '?')}, {p.get('country', '?')}")
    print(f"  YOE:        {p.get('years_of_experience', '?')} years")
    print(f"  Company:    {p.get('current_company', '?')} (size: {p.get('current_company_size', '?')})")
    print(f"  Education:  ", end="")
    for e in c.get("education", []):
        print(f"{e.get('degree', '?')} in {e.get('field_of_study', '?')} from {e.get('institution', '?')} ({e.get('tier', '?')})", end="; ")
    print()

    # Career history — compact
    print(f"  Career ({len(career)} roles):")
    for j in career[:3]:
        end = j.get("end_date") or "present"
        print(f"    • {j['title']} @ {j['company']} ({j['start_date']} → {end}, {j['duration_months']}mo)")
        desc = j.get("description", "")
        if desc:
            # Show first 200 chars of description
            print(f"      \"{desc[:200]}{'...' if len(desc) > 200 else ''}\"")

    # Skills — top by endorsements
    print(f"  Skills ({len(c.get('skills', []))} total): {compact_skills(c.get('skills', []))}")

    # Signals — the important ones
    days = days_since(s.get("last_active_date", ""))
    print(f"  Signals:")
    print(f"    last_active: {s.get('last_active_date', '?')} ({days}d ago)  |  open_to_work: {s.get('open_to_work_flag', '?')}")
    print(f"    response_rate: {s.get('recruiter_response_rate', '?')}  |  avg_response_time: {s.get('avg_response_time_hours', '?')}h")
    print(f"    notice: {s.get('notice_period_days', '?')}d  |  salary: {s.get('expected_salary_range_inr_lpa', '?')}")
    print(f"    github: {s.get('github_activity_score', '?')}  |  saved_by_recruiters_30d: {s.get('saved_by_recruiters_30d', '?')}")
    print(f"    interview_completion: {s.get('interview_completion_rate', '?')}  |  offer_acceptance: {s.get('offer_acceptance_rate', '?')}")
    print(f"    verified_email: {s.get('verified_email', '?')}  |  verified_phone: {s.get('verified_phone', '?')}  |  linkedin: {s.get('linkedin_connected', '?')}")

    # Skill assessment scores (if any)
    sas = s.get("skill_assessment_scores", {})
    if sas:
        print(f"    skill_assessments: {sas}")

    # Summary
    summary = p.get("summary", "")
    if summary:
        print(f"  Summary: \"{summary[:400]}{'...' if len(summary) > 400 else ''}\"")

    # Heuristic suggestion
    print("─" * 78)
    print(f"  🤖 SUGGESTED TIER: {suggested_tier}")
    for r in reasons:
        print(f"      • {r}")
    print("─" * 78)


def prompt_for_label(cid, suggested_tier):
    """Prompt the user for a tier and optional note."""
    print(f"\n  Enter tier for {cid} (0-3, or 's' for suggested={suggested_tier}, 'k' to skip, 'q' to quit): ", end="")
    sys.stdout.flush()
    choice = input().strip().lower()

    if choice == "q":
        return None, None, "quit"
    if choice == "k":
        return None, None, "skip"
    if choice == "s" or choice == "":
        tier = suggested_tier
    else:
        try:
            tier = int(choice)
            if tier not in (0, 1, 2, 3):
                print(f"  Invalid tier {tier}. Must be 0, 1, 2, or 3.")
                return None, None, "retry"
        except ValueError:
            print(f"  Invalid input '{choice}'. Use 0/1/2/3/s/k/q.")
            return None, None, "retry"

    print(f"  Optional note (press Enter to skip): ", end="")
    sys.stdout.flush()
    note = input().strip()
    if not note:
        note = None

    return tier, note, "ok"


def print_summary_table(candidates, labels):
    """Print a one-line-per-candidate summary table."""
    print("\n" + "=" * 100)
    print(f"{'ID':<15} {'Title':<28} {'YOE':>5} {'Loc':<14} {'LastAct':>8} {'RR':>5} {'Notice':>7} {'Tier':>5} {'Note':<30}")
    print("=" * 100)

    for c in candidates:
        cid = c["candidate_id"]
        p = c["profile"]
        s = c["redrob_signals"]
        days = days_since(s.get("last_active_date", ""))
        labeled = labels.get("labels", {}).get(cid, {})
        tier = labeled.get("tier", "-")
        note = (labeled.get("note") or "")[:28]

        title = (p.get("current_title", "?") or "?")[:27]
        loc = (p.get("location", "?") or "?")[:13]
        print(f"{cid:<15} {title:<28} {p.get('years_of_experience', 0):>5.1f} {loc:<14} "
              f"{days:>6}d {s.get('recruiter_response_rate', 0):>5.2f} "
              f"{s.get('notice_period_days', 0):>5}d {str(tier):>5} {note:<30}")

    print("=" * 100)
    labeled_count = sum(1 for v in labels.get("labels", {}).values() if v.get("tier") is not None)
    print(f"\n  Labeled: {labeled_count}/{len(candidates)}")


def main():
    parser = argparse.ArgumentParser(description="Interactive dev-set labeler")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-labeled candidates")
    parser.add_argument("--candidate", type=str, default=None,
                        help="Label a specific candidate by ID (e.g. CAND_0000007)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table of all 50 candidates and exit")
    parser.add_argument("--auto-accept", action="store_true",
                        help="Batch-accept all heuristic suggestions and exit (review with --summary after)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all existing labels and start fresh")
    args = parser.parse_args()

    candidates = load_candidates()
    labels_data = load_labels()
    labels = labels_data.setdefault("labels", {})

    if args.reset:
        labels_data["labels"] = {}
        labels = labels_data["labels"]
        save_labels(labels_data)
        print(f"  Cleared all labels. File: {DEV_LABELS}")
        return

    if args.summary:
        print_summary_table(candidates, labels_data)
        return

    if args.auto_accept:
        print(f"\n  Auto-accepting heuristic suggestions for all 50 candidates...")
        from collections import Counter
        tier_dist = Counter()
        for c in candidates:
            tier, reasons = heuristic_tier(c)
            labels[c["candidate_id"]] = {
                "tier": tier,
                "note": "auto-accepted heuristic: " + " | ".join(reasons[:2])
            }
            tier_dist[tier] += 1
        save_labels(labels_data)
        print(f"  ✓ Saved {sum(1 for v in labels.values() if v.get('tier') is not None)} labels.")
        print(f"\n  Tier distribution after auto-accept:")
        for t in [3, 2, 1, 0]:
            print(f"    Tier {t}: {tier_dist.get(t, 0)} candidates")
        print(f"\n  Next: review with --summary, then override individual candidates with:")
        print(f"    python scripts/label_dev_set.py --candidate CAND_0000031")
        return

    if args.candidate:
        target = next((c for c in candidates if c["candidate_id"] == args.candidate), None)
        if not target:
            print(f"Candidate {args.candidate} not found in dev set.")
            return
        candidates_to_label = [target]
    else:
        if args.resume:
            candidates_to_label = [c for c in candidates
                                   if c["candidate_id"] not in labels
                                   or labels[c["candidate_id"]].get("tier") is None]
        else:
            candidates_to_label = candidates

    print(f"\n  DeepFit Dev-Set Labeler")
    print(f"  Candidates to label: {len(candidates_to_label)}")
    print(f"  Already labeled:     {sum(1 for v in labels.values() if v.get('tier') is not None)}")
    print(f"  Output file:         {DEV_LABELS}")
    print(f"\n  Tier guide:")
    print(f"    3 = TOP-10 quality (ideal AI Engineer, active, Pune/Noida etc.)")
    print(f"    2 = RELEVANT (solid match, 1 minor concern)")
    print(f"    1 = MARGINAL (adjacent skills, missing key req)")
    print(f"    0 = IRRELEVANT (honeypot, trap, hard disqualifier)")
    print(f"\n  Commands at prompt: 0/1/2/3 = tier | s = suggested | k = skip | q = quit")

    for c in candidates_to_label:
        cid = c["candidate_id"]
        suggested_tier, reasons = heuristic_tier(c)

        # Show existing label if any
        existing = labels.get(cid, {})
        if existing.get("tier") is not None:
            print(f"\n  ⚠ Already labeled: tier={existing['tier']}, note={existing.get('note', '')}")
            print(f"  Press Enter to keep, or enter new tier to override.")

        print_candidate_summary(c, suggested_tier, reasons)

        while True:
            tier, note, status = prompt_for_label(cid, suggested_tier)
            if status == "quit":
                print(f"\n  Saving labels to {DEV_LABELS}...")
                save_labels(labels_data)
                print(f"  Saved {sum(1 for v in labels.values() if v.get('tier') is not None)} labels. Goodbye.")
                return
            if status == "skip":
                print(f"  Skipping {cid}")
                break
            if status == "retry":
                continue

            labels[cid] = {"tier": tier}
            if note:
                labels[cid]["note"] = note
            save_labels(labels_data)
            print(f"  ✓ Saved {cid}: tier={tier}" + (f", note='{note}'" if note else ""))
            break

    print(f"\n  ✅ Done! All candidates labeled.")
    print(f"  Total labeled: {sum(1 for v in labels.values() if v.get('tier') is not None)}")
    print(f"  File: {DEV_LABELS}")

    # Print tier distribution
    from collections import Counter
    tier_dist = Counter(v.get("tier") for v in labels.values() if v.get("tier") is not None)
    print(f"\n  Tier distribution:")
    for t in [3, 2, 1, 0]:
        print(f"    Tier {t}: {tier_dist.get(t, 0)} candidates")


if __name__ == "__main__":
    main()
