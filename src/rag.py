"""
src/rag.py — Retrieval-Augmented Generation pipeline (Phase 3)

Corpus  : alibayram/turkish-hospital-medical-articles (gated, requires HF_TOKEN)
Embedder: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
Index   : FAISS IndexFlatIP on L2-normalised embeddings

Pipeline: build_corpus → chunk_article → build_index → (load_index at runtime)
          → retrieve → build_rag_prompt → generate_rag

Requires HF_TOKEN env var for corpus download (dataset is gated).
"""

import os
import re
import numpy as np
import pandas as pd
import faiss
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


HOSPITAL_SPLITS = [
    "acibadem", "anadolusaglik", "atlas", "baskentistanbul", "bayindir",
    "florence", "guven", "liv", "medicalpark", "medicalpoint",
    "medicana", "medipol", "memorial", "yeditepe",
]

MPNET_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MIN_TEXT_LENGTH = 100


# ──────────────────────────────────────────────
# 1. Build corpus
# ──────────────────────────────────────────────

def build_corpus(cache_dir: str | None = None) -> list[dict]:
    """
    Loads all 14 hospital splits from alibayram/turkish-hospital-medical-articles
    and returns them as a single list of article dicts.

    Each record contains: hospital, title, text, url.
    Records with missing or short text (< 100 chars) are dropped.
    Requires the HF_TOKEN environment variable (dataset is gated).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Export it with: export HF_TOKEN=hf_..."
        )

    all_splits = []
    for split_name in tqdm(HOSPITAL_SPLITS, desc="Loading hospital splits"):
        ds = load_dataset(
            "alibayram/turkish-hospital-medical-articles",
            split=split_name,
            token=token,
            cache_dir=cache_dir,
        )
        for row in ds:
            text = row.get("text") or ""
            if len(text) < MIN_TEXT_LENGTH:
                continue
            all_splits.append({
                "hospital":  split_name,
                "title":     row.get("title") or "",
                "text":      text,
                "url":       row.get("url") or "",
            })

    print(f"Corpus built: {len(all_splits):,} articles across {len(HOSPITAL_SPLITS)} hospitals.")
    return all_splits


# ──────────────────────────────────────────────
# 2. Chunk a single article
# ──────────────────────────────────────────────

def chunk_article(
    article: dict,
    target_tokens: int = 256,
    tokenizer=None,
    article_idx: int = 0,
) -> list[dict]:
    """
    Splits an article into token-bounded chunks, each prepended with the title.

    Strategy: split on double newlines (fallback: single newlines), then
    greedily merge paragraphs up to target_tokens. Paragraphs that alone
    exceed target_tokens are further split on sentence boundaries.
    Token counting uses the mpnet tokenizer passed as `tokenizer`.
    """
    title    = article["title"]
    text     = article["text"]
    hospital = article["hospital"]
    url      = article["url"]

    # ── Paragraph splitting ───────────────────
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(paragraphs) < 2:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # ── Sentence-split long paragraphs ────────
    def _count_tokens(s: str) -> int:
        if tokenizer is None:
            return len(s.split())
        return len(tokenizer.tokenize(s))

    def _split_sentences(para: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", para)
        return [p.strip() for p in parts if p.strip()]

    expanded: list[str] = []
    for para in paragraphs:
        if _count_tokens(para) > target_tokens:
            expanded.extend(_split_sentences(para))
        else:
            expanded.append(para)

    # ── Greedy merge up to target_tokens ──────
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for seg in expanded:
        seg_tokens = _count_tokens(seg)
        if current_tokens + seg_tokens > target_tokens and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [seg]
            current_tokens = seg_tokens
        else:
            current_parts.append(seg)
            current_tokens += seg_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    # ── Build output dicts ─────────────────────
    result = []
    for chunk_idx, chunk_text in enumerate(chunks):
        full_text = f"{title}\n\n{chunk_text}"
        result.append({
            "chunk_text":    full_text,
            "hospital":      hospital,
            "article_title": title,
            "url":           url,
            "chunk_id":      f"{hospital}_{article_idx}_{chunk_idx}",
        })
    return result


# ──────────────────────────────────────────────
# 3. Build FAISS index
# ──────────────────────────────────────────────

def build_index(
    corpus: list[dict],
    output_dir: str,
    batch_size: int = 64,
    force: bool = False,
) -> None:
    """
    Chunks every article, embeds all chunks with mpnet, and writes a FAISS
    IndexFlatIP (inner-product on L2-normalised vectors = cosine similarity)
    plus a parquet of chunk metadata to output_dir.

    Skips rebuild if index.faiss already exists and force=False.
    """
    index_path  = os.path.join(output_dir, "index.faiss")
    chunks_path = os.path.join(output_dir, "chunks.parquet")

    if os.path.exists(index_path) and not force:
        print(f"Index already exists at {index_path}. Pass force=True to rebuild.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Load mpnet ────────────────────────────
    print(f"Loading embedding model: {MPNET_MODEL_NAME}")
    embed_model = SentenceTransformer(MPNET_MODEL_NAME)
    mpnet_tokenizer = embed_model.tokenizer

    # ── Chunk all articles ────────────────────
    print("Chunking articles...")
    all_chunks: list[dict] = []
    for article_idx, article in enumerate(tqdm(corpus, desc="Chunking")):
        all_chunks.extend(
            chunk_article(article, tokenizer=mpnet_tokenizer, article_idx=article_idx)
        )

    print(f"Total chunks: {len(all_chunks):,}")

    # ── Embed in batches ──────────────────────
    texts = [c["chunk_text"] for c in all_chunks]
    print("Embedding chunks...")
    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise
    )

    # ── Build FAISS index ─────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # ── Save ──────────────────────────────────
    faiss.write_index(index, index_path)

    metadata = [{k: v for k, v in c.items() if k != "chunk_text"} | {"chunk_text": c["chunk_text"]}
                for c in all_chunks]
    chunks_df = pd.DataFrame(metadata)
    chunks_df.to_parquet(chunks_path, index=False)

    index_size_mb = os.path.getsize(index_path) / (1024 ** 2)
    print(
        f"\nIndex built successfully."
        f"\n  Articles : {len(corpus):,}"
        f"\n  Chunks   : {len(all_chunks):,}"
        f"\n  Emb dim  : {dim}"
        f"\n  Index    : {index_size_mb:.1f} MB  →  {index_path}"
        f"\n  Metadata : {chunks_path}"
    )


# ──────────────────────────────────────────────
# 4. Load index at runtime
# ──────────────────────────────────────────────

def load_index(index_dir: str) -> tuple:
    """
    Loads and returns (faiss_index, chunks_df, embedding_model) from index_dir.

    chunks_df row i corresponds to FAISS index row i.
    """
    index_path  = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.parquet")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run build_index() first."
        )

    faiss_index  = faiss.read_index(index_path)
    chunks_df    = pd.read_parquet(chunks_path)
    embed_model  = SentenceTransformer(MPNET_MODEL_NAME)

    print(
        f"Index loaded: {faiss_index.ntotal:,} vectors, "
        f"{len(chunks_df):,} chunk records."
    )
    return faiss_index, chunks_df, embed_model


# ──────────────────────────────────────────────
# 5. Retrieve
# ──────────────────────────────────────────────

def retrieve(
    query: str,
    faiss_index,
    chunks_df: pd.DataFrame,
    embedding_model,
    k: int = 3,
) -> list[dict]:
    """
    Embeds `query` with mpnet, searches the FAISS index for the top-k
    nearest chunks, and returns them as dicts ordered by descending score.

    Each returned dict contains: chunk_text, hospital, article_title, url, score.
    """
    query_vec = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = faiss_index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = chunks_df.iloc[idx]
        results.append({
            "chunk_text":    row["chunk_text"],
            "hospital":      row["hospital"],
            "article_title": row["article_title"],
            "url":           row["url"],
            "score":         float(score),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ──────────────────────────────────────────────
# 6. Build RAG prompt
# ──────────────────────────────────────────────

def build_rag_prompt(
    question: str,
    doctor_title: str,
    doctor_specialty: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Assembles the Turkish RAG prompt: system persona + numbered retrieved
    excerpts + patient question, with the answer left open for generation.

    Works for any k (not hardcoded to 3).
    """
    context_lines = "\n".join(
        f"[{i+1}] {chunk['article_title']}: {chunk['chunk_text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = (
        f"Sen {doctor_specialty} alanında uzman bir Türk {doctor_title}sın. "
        f"Aşağıda Türk hastane kaynaklarından alınmış ilgili tıbbi bilgiler bulunmaktadır. "
        f"Bu bilgileri dikkate alarak hastanın sorusuna cevap ver.\n\n"
        f"İlgili tıbbi bilgiler:\n{context_lines}\n\n"
        f"Hasta sorusu: {question}\n"
        f"Cevap:"
    )
    return prompt


# ──────────────────────────────────────────────
# 7. Full RAG generation
# ──────────────────────────────────────────────

def generate_rag(
    model,
    tokenizer,
    question: str,
    doctor_title: str,
    doctor_specialty: str,
    faiss_index,
    chunks_df: pd.DataFrame,
    embedding_model,
    k: int = 3,
    max_new_tokens: int = 512,
) -> dict:
    """
    Runs the full RAG pipeline: retrieve → build prompt → generate → decode.

    Returns a dict with keys:
      generated_answer : the model's response text
      retrieved_chunks : list of dicts from retrieve()
      prompt           : the full prompt string fed to the model
    """
    retrieved_chunks = retrieve(question, faiss_index, chunks_df, embedding_model, k=k)
    prompt = build_rag_prompt(question, doctor_title, doctor_specialty, retrieved_chunks)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len       = inputs["input_ids"].shape[1]
    generated_ids    = output_ids[0][prompt_len:]
    generated_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_answer": generated_answer,
        "retrieved_chunks": retrieved_chunks,
        "prompt":           prompt,
    }
