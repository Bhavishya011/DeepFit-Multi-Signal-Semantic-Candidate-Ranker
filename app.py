"""
DeepFit — HuggingFace Spaces Sandbox
============================================================================

A lightweight demo of the DeepFit ranker. Accepts a small candidate sample
(≤100 candidates) as JSON input and produces a ranked CSV output.

Deploy to HuggingFace Spaces:
    1. Create a new Space at https://huggingface.co/spaces/YOUR_USERNAME/deepfit
    2. Choose "Streamlit" SDK
    3. Upload this file as app.py + all ranker/ code + config/ + artifacts/
    4. Add requirements.txt with the dependencies

The sandbox runs the SAME pipeline as rank.py but on a small sample,
so organizers can verify the code runs end-to-end.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Make ranker importable
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from ranker.pipeline import Pipeline, load_candidates_from_file
from ranker.types import Candidate


# ─── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFit — Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    """Load the pipeline once and cache it."""
    try:
        return Pipeline.build()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None


def main():
    st.title("🎯 DeepFit — Intelligent Candidate Discovery")
    st.markdown("""
    **Multi-signal semantic ranker for the Redrob Hackathon.**

    Upload a JSON file of candidates (≤100 for the sandbox) and get a ranked
    CSV with evidence-grounded reasoning for each candidate.
    """)

    # ─── Sidebar: pipeline info ───────────────────────────────────────────
    with st.sidebar:
        st.header("Pipeline Info")
        st.markdown("""
        **10 modules:**
        1. JD-Intent Decoder
        2. Multi-field Encoder
        3. Honeypot + Trap Filter
        4. Production-Evidence Extractor
        5. Behavioral Availability
        6. Structural + Seniority
        7. BM25 + FAISS + RRF Recall
        8. Cross-encoder Rerank
        9. Final Combiner
        10. Reasoning Generator

        **Constraints:**
        - ≤5 min runtime
        - 16GB RAM, CPU only
        - No network during ranking
        """)
        st.divider()
        st.caption("Built by Bhavishya Jain for H2S Redrob Hackathon")

    # ─── Load pipeline ────────────────────────────────────────────────────
    with st.spinner("Loading pipeline (this may take 30-60s on first run)..."):
        pipeline = load_pipeline()

    if pipeline is None:
        st.error("Pipeline failed to load. Check logs.")
        st.stop()

    st.success("✅ Pipeline loaded successfully!")

    # ─── Input: file upload or sample data ────────────────────────────────
    st.header("1. Upload Candidates")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload candidates JSON file",
            type=["json"],
            help="JSON array of candidate objects (≤100 recommended for sandbox)"
        )

    with col2:
        st.markdown("**Or try sample data:**")
        if st.button("Load 10 sample candidates"):
            sample_path = Path(__file__).parent / "tests" / "dev_set" / "dev_candidates.json"
            if sample_path.exists():
                with open(sample_path) as f:
                    samples = json.load(f)[:10]
                st.session_state["candidates"] = samples
                st.success(f"Loaded {len(samples)} sample candidates")
            else:
                st.error("Sample file not found")

    # Parse uploaded file
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            candidates_data = json.loads(content)
            if not isinstance(candidates_data, list):
                st.error("File must be a JSON array of candidate objects")
                st.stop()
            st.session_state["candidates"] = candidates_data
            st.success(f"Loaded {len(candidates_data)} candidates from {uploaded_file.name}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

    # ─── Display loaded candidates ────────────────────────────────────────
    if "candidates" in st.session_state:
        candidates_data = st.session_state["candidates"]
        st.info(f"📊 {len(candidates_data)} candidates loaded")

        # Show preview
        with st.expander("Preview first candidate"):
            st.json(candidates_data[0])

        # ─── Run pipeline ─────────────────────────────────────────────────
        st.header("2. Run Ranker")

        if st.button("🚀 Rank Candidates", type="primary"):
            # Convert to Candidate objects
            candidates = [Candidate.from_dict(r) for r in candidates_data]

            # Run pipeline
            with st.spinner(f"Ranking {len(candidates)} candidates..."):
                t0 = time.time()
                try:
                    results = pipeline.run(
                        candidates=candidates,
                        top_n=min(100, len(candidates)),
                        verbose=False,
                    )
                    runtime = time.time() - t0
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

            st.success(f"✅ Ranked {len(results)} candidates in {runtime:.1f}s")

            # ─── Display results ──────────────────────────────────────────
            st.header("3. Results")

            # Build DataFrame
            rows = []
            for r in results:
                rows.append({
                    "rank": r.rank,
                    "candidate_id": r.candidate_id,
                    "score": round(r.score, 4),
                    "reasoning": r.reasoning,
                })
            df = pd.DataFrame(rows)

            # Show table
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                    "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                    "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                },
            )

            # ─── Download CSV ─────────────────────────────────────────────
            st.header("4. Download Results")

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📅 Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
            )

            # ─── Score distribution ───────────────────────────────────────
            st.subheader("Score Distribution")
            chart_data = pd.DataFrame({
                "rank": df["rank"],
                "score": df["score"],
            })
            st.line_chart(chart_data.set_index("rank"))

            # ─── Pipeline stats ───────────────────────────────────────────
            st.subheader("Pipeline Stats")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Candidates Input", len(candidates))
            with col_b:
                st.metric("Ranked Output", len(results))
            with col_c:
                st.metric("Runtime", f"{runtime:.1f}s")
            with col_d:
                st.metric("Top Score", f"{results[0].score:.4f}")


if __name__ == "__main__":
    main()
