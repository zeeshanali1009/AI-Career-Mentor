import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_EMBEDDER = None

def get_embedder(model_name="all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    """
    Split long text into chunks with small overlap.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def build_index_from_texts(texts: list, model_name="all-MiniLM-L6-v2"):
    """
    texts: list[str] - pieces of text (e.g., each KB entry)
    returns: index, embeddings_array, texts_list
    """
    embedder = get_embedder(model_name)
    # For each provided text, we can chunk further but for KB entries keep as-is
    chunks = []
    for t in texts:
        # if text long, chunk it; else keep
        if len(t) > 1000:
            chunks.extend(chunk_text(t, chunk_size=800, overlap=100))
        else:
            chunks.append(t)

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("No embeddings created; check input texts.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings, chunks

def query_index(query: str, index, embeddings, chunks, top_k: int = 5, model_name="all-MiniLM-L6-v2"):
    """
    Returns top_k most relevant chunks as list[str].
    """
    embedder = get_embedder(model_name)
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results
