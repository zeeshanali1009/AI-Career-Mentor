import streamlit as st
import tempfile
import os

from modules.resume_parser import load_resume_text
from modules.vector_store import build_index_from_texts, query_index
from modules.mentor_engine import mentor_recommendation

st.set_page_config(page_title="AI Career Mentor", layout="centered")
st.title("🧭 AI Career Mentor")
st.markdown(
    "Upload a resume (PDF / DOCX / TXT) and get career suggestions: best-fit roles, missing skills, "
    "learning roadmap, and resume improvement tips (domain-agnostic)."
)

uploaded = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
kb_file = st.file_uploader("Optional: Upload a job-roles knowledge-base (TXT). If not provided, built-in KB will be used.", type=["txt"])

if uploaded:
    with st.spinner("Reading resume..."):
        # load_resume_text accepts file-like object or path
        resume_text = load_resume_text(uploaded)
    st.success("✅ Resume loaded")

    # optionally save KB to temp file or use default inside mentor_engine
    kb_path = None
    if kb_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.write(kb_file.getbuffer())
        tmp.close()
        kb_path = tmp.name

    st.markdown("---")
    if st.button("Get Career Advice"):
        with st.spinner("Generating career recommendations..."):
            try:
                result = mentor_recommendation(resume_text, kb_path=kb_path, top_k=6)
            except Exception as e:
                st.error(f"Error while generating recommendation: {e}")
                raise

        # Present the result
        st.subheader("🔎 Suggested Roles")
        for r in result.get("suggested_roles", []):
            st.markdown(f"- **{r.get('title','')}** — {r.get('reason','')}")

        st.subheader("🧩 Missing / Recommended Skills")
        for s in result.get("missing_skills", []):
            st.markdown(f"- {s}")

        st.subheader("📈 Learning Roadmap")
        for step in result.get("learning_path", []):
            st.markdown(f"- {step}")

        st.subheader("📝 Resume Improvement Tips")
        for tip in result.get("resume_improvements", []):
            st.markdown(f"- {tip}")

        st.markdown("---")
        st.info("You can re-run with another resume or upload a custom knowledge-base file for company-specific roles.")
