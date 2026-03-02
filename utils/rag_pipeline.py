"""RAG Pipeline — ingest the paper PDF and answer questions with local HF models.

Uses:
  • sentence-transformers/all-MiniLM-L6-v2  for embeddings   (free, local)
  • google/flan-t5-base                     for generation   (free, local)
  • FAISS                                   for vector search (saved to disk)
"""

from __future__ import annotations

import os
import functools

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

RAG_AVAILABLE = True
_MISSING_REASON = ""

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError as exc:
    RAG_AVAILABLE = False
    _MISSING_REASON = f"Missing dependency: {exc}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_PDF_PATH = os.path.join(_ASSETS_DIR, "attention_paper.pdf")
_FAISS_INDEX_DIR = os.path.join(_ASSETS_DIR, "faiss_index")


# ---------------------------------------------------------------------------
# Embedding model (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ---------------------------------------------------------------------------
# Build / load the FAISS vector store  (persisted to disk)
# ---------------------------------------------------------------------------

_vectorstore_cache = None

def _build_or_load_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    embeddings = _get_embeddings()

    if os.path.isdir(_FAISS_INDEX_DIR):
        try:
            vs = FAISS.load_local(
                _FAISS_INDEX_DIR, embeddings,
                allow_dangerous_deserialization=True,
            )
            _vectorstore_cache = vs
            return vs
        except Exception:
            pass

    if not os.path.isfile(_PDF_PATH):
        return None

    loader = PyPDFLoader(_PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(_FAISS_INDEX_DIR)
    _vectorstore_cache = vs
    return vs


# ---------------------------------------------------------------------------
# Generation model (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_generator():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Retrieve-then-generate
# ---------------------------------------------------------------------------

def _retrieve_and_generate(question: str, vectorstore) -> dict:
    docs = vectorstore.similarity_search(question, k=4)
    tokenizer, model = _get_generator()

    # flan-t5-base has a 512-token encoder limit.
    # Reserve ~80 tokens for the prompt frame + question, rest for context.
    question_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
    max_context_tokens = 512 - len(question_tokens) - 40  # 40 tokens for framing

    # Build context, trimming chunks to fit
    context_parts = []
    used_tokens = 0
    for doc in docs:
        chunk_tokens = tokenizer(doc.page_content, add_special_tokens=False)["input_ids"]
        if used_tokens + len(chunk_tokens) > max_context_tokens:
            # Take partial chunk if room left
            remaining = max_context_tokens - used_tokens
            if remaining > 50:
                partial = tokenizer.decode(chunk_tokens[:remaining], skip_special_tokens=True)
                context_parts.append(partial)
            break
        context_parts.append(doc.page_content)
        used_tokens += len(chunk_tokens)

    context = "\n".join(context_parts)

    prompt = (
        f"Based on these excerpts from 'Attention Is All You Need', "
        f"give a detailed answer.\n\n"
        f"{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    sources = [doc.page_content for doc in docs]
    return {"answer": answer, "sources": sources, "error": None}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def ask(question: str) -> dict:
    if not RAG_AVAILABLE:
        return {"answer": "", "sources": [], "error": _MISSING_REASON}

    if not os.path.isfile(_PDF_PATH):
        return {
            "answer": "", "sources": [],
            "error": f"Paper PDF not found at `{_PDF_PATH}`.",
        }

    vs = _build_or_load_vectorstore()
    if vs is None:
        return {"answer": "", "sources": [], "error": "Vector store build failed."}

    try:
        return _retrieve_and_generate(question, vs)
    except Exception as exc:
        return {"answer": "", "sources": [], "error": f"Generation error: {exc}"}
