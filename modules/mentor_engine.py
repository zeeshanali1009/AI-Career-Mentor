import os
from dotenv import load_dotenv
from modules.vector_store import build_index_from_texts, query_index
from sentence_transformers import SentenceTransformer

# load .env for GROQ key
load_dotenv()

# Try Groq first, otherwise fallback to Ollama (phi3)
API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = bool(API_KEY)
if USE_GROQ:
    try:
        from groq import Groq
        groq_client = Groq(api_key=API_KEY)
    except Exception:
        USE_GROQ = False

if not USE_GROQ:
    try:
        from langchain_community.llms import Ollama
        def _ollama_call(prompt, model_name="phi3"):
            llm = Ollama(model=model_name)
            return llm.invoke(prompt)
    except Exception:
        _ollama_call = None

# light embedder for similarity (used by KB retrieval)
_EMB = SentenceTransformer("all-MiniLM-L6-v2")

def _call_llm(prompt: str) -> str:
    """
    Use Groq if available, otherwise Ollama fallback.
    """
    if USE_GROQ:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    else:
        if _ollama_call is None:
            raise RuntimeError("No LLM backend available. Set GROQ_API_KEY in .env or install Ollama.")
        return _ollama_call(prompt)

def _load_default_kb():
    """
    A small built-in knowledge base. For production replace with a large curated KB file
    or pass kb_path to mentor_recommendation.
    Each entry format: title\ndescription\n\n
    """
    kb_text = """
Data Analyst
Required skills: Python (Pandas), SQL, data visualization, statistics. Typical responsibilities: data cleaning, reporting, dashboarding.

Business Analyst
Required skills: business analysis, stakeholder communication, SQL, visualization, requirements elicitation. Typical responsibilities: translate business needs, design solutions.

Software Engineer (Backend)
Required skills: Python/Java/Go, REST APIs, databases, cloud basics. Responsibilities: build backend services, write tests, deploy.

Machine Learning Engineer
Required skills: Python, ML frameworks (scikit-learn, PyTorch), model deployment, feature engineering. Responsibilities: model development, evaluation, productionization.

Product Manager
Required skills: stakeholder management, product discovery, prioritization, data-informed decisions. Responsibilities: roadmap, user requirements, go-to-market.
"""
    # Split entries by blank line; return list of entries (title + desc)
    entries = [e.strip() for e in kb_text.strip().split("\n\n") if e.strip()]
    return entries

def mentor_recommendation(resume_text: str, kb_path: str = None, top_k: int = 5):
    """
    Main function: returns a dict with:
      - suggested_roles: list of {title, reason}
      - missing_skills: list[str]
      - learning_path: list[str]
      - resume_improvements: list[str]
    kb_path: optional path to a TXT KB file where entries are separated by blank lines.
    """
    # Load KB
    if kb_path and os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        entries = [e.strip() for e in raw.split("\n\n") if e.strip()]
    else:
        entries = _load_default_kb()

    # Build index
    index, embeddings, chunks = build_index_from_texts(entries)

    # Query KB with resume text to find top relevant roles
    top_chunks = query_index(resume_text, index, embeddings, chunks, top_k=top_k)

    # Prepare context for LLM: include resume and the top matched KB entries
    kb_context = "\n\n".join(top_chunks)
    prompt = f"""
You are an expert career advisor and hiring manager.

Given the candidate resume and the following job-role knowledge snippets, produce a JSON-like output with the following keys:
- suggested_roles: list of objects with 'title' and short 'reason' (1-2 sentences).
- missing_skills: short list (5 max) of concrete skills/certifications the candidate lacks for top roles.
- learning_path: 4-6 concrete steps or courses to close the gaps (short bullets).
- resume_improvements: 4 short actionable tips to improve resume clarity and impact.

Respond in valid JSON.

Resume:
{resume_text[:4000]}

Top knowledge snippets:
{kb_context}
"""
    llm_out = _call_llm(prompt)

    # Try to parse JSON-ish output. We will attempt simple eval safely.
    import json, re
    # Extract JSON object from text (attempt)
    json_text = None
    try:
        # Find first '{' and last '}' and parse
        m = re.search(r"\{.*\}", llm_out, flags=re.S)
        if m:
            json_text = m.group(0)
            result = json.loads(json_text)
        else:
            # If not JSON, try to ask LLM again to output strict JSON
            follow = ("The previous output wasn't returned as strict JSON. "
                      "Please return ONLY a JSON object with the keys: suggested_roles, missing_skills, learning_path, resume_improvements.")
            llm_out2 = _call_llm(prompt + "\n\n" + follow)
            m2 = re.search(r"\{.*\}", llm_out2, flags=re.S)
            if m2:
                result = json.loads(m2.group(0))
            else:
                # As last fallback, return parsed heuristics
                result = {"suggested_roles": [], "missing_skills": [], "learning_path": [], "resume_improvements": []}
    except Exception:
        # fallback empty structured result
        result = {"suggested_roles": [], "missing_skills": [], "learning_path": [], "resume_improvements": []}

    # Ensure keys exist
    for k in ["suggested_roles", "missing_skills", "learning_path", "resume_improvements"]:
        result.setdefault(k, [])

    return result
