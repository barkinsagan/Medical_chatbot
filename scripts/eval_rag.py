"""
scripts/eval_rag.py — RAG evaluation pipeline (Phase 3)

Loads the QLoRA adapter via HuggingFace transformers + PEFT (no Unsloth),
retrieves context from the FAISS index, generates RAG answers on the test
set, and computes BLEU, token-F1, and SBERT cosine similarity against
reference answers.

Batched design: retrieval and generation are both batched so the 4090
stays saturated throughout the run.

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py \
        --adapter_dir   outputs/checkpoints/qlora \
        --index_dir     outputs/rag_index \
        --n_samples     100 \
        --k             3 \
        --batch_size    8 \
        --save_path     outputs/eval_results/rag_eval.csv

Requires:
    export HF_TOKEN=hf_...
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import sacrebleu
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import load_index, build_rag_prompt
from src.data import load_doktorsitesi


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MPNET_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--adapter_dir",    default="outputs/checkpoints/qlora",
                        help="Path to QLoRA adapter directory")
    parser.add_argument("--base_model",     default=BASE_MODEL,
                        help="HuggingFace base model ID")
    parser.add_argument("--index_dir",      default="outputs/rag_index",
                        help="Path to directory with index.faiss and chunks.parquet")
    parser.add_argument("--n_samples",      type=int, default=None,
                        help="Number of test samples to evaluate (default: all)")
    parser.add_argument("--k",              type=int, default=3,
                        help="Number of chunks to retrieve per query")
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Generation batch size (tune to VRAM — try 8 or 16 on 4090)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_path",      default="outputs/eval_results/rag_eval.csv",
                        help="Where to save the results CSV")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_qlora_model(adapter_dir: str, base_model: str = BASE_MODEL):
    """Loads base model in 4-bit and applies the QLoRA adapter via PEFT."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batch generation

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────
# Batched retrieval
# ──────────────────────────────────────────────

def batch_retrieve(
    queries: list[str],
    faiss_index,
    chunks_df: pd.DataFrame,
    embed_model,
    k: int = 3,
) -> list[list[dict]]:
    """
    Embeds a batch of queries in one forward pass, searches FAISS for all,
    and returns a list (one per query) of top-k chunk dicts.
    """
    vecs = embed_model.encode(
        queries,
        batch_size=len(queries),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    all_scores, all_indices = faiss_index.search(vecs, k)

    results = []
    for score_row, idx_row in zip(all_scores, all_indices):
        hits = []
        for score, idx in zip(score_row, idx_row):
            if idx == -1:
                continue
            row = chunks_df.iloc[idx]
            hits.append({
                "chunk_text":    row["chunk_text"],
                "hospital":      row["hospital"],
                "article_title": row["article_title"],
                "url":           row["url"],
                "score":         float(score),
            })
        results.append(sorted(hits, key=lambda x: x["score"], reverse=True))
    return results


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_token_f1(generated: str, reference: str) -> float:
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    if not gen_tokens or not ref_tokens:
        return 0.0
    gen_counts = Counter(gen_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((gen_counts & ref_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(gen_tokens)
    recall    = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(df: pd.DataFrame) -> float:
    result = sacrebleu.corpus_bleu(
        df["generated"].tolist(),
        [df["reference"].tolist()],
        tokenize="intl",
    )
    return result.score


def compute_f1_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["f1"] = [
        compute_token_f1(gen, ref)
        for gen, ref in zip(df["generated"], df["reference"])
    ]
    return df


def compute_similarity(
    references: list[str],
    generated: list[str],
    batch_size: int = 64,
) -> list[float]:
    sbert = SentenceTransformer(MPNET_NAME)
    ref_emb = sbert.encode(references, batch_size=batch_size, show_progress_bar=True,
                           convert_to_tensor=True, normalize_embeddings=True)
    gen_emb = sbert.encode(generated,  batch_size=batch_size, show_progress_bar=True,
                           convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(ref_emb, gen_emb).diagonal().cpu().tolist()


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

def print_summary(label: str, bleu: float, df: pd.DataFrame) -> dict:
    f1_mean  = df["f1"].mean()
    sim_mean = df["similarity"].mean()
    sim_med  = df["similarity"].median()
    sim_std  = df["similarity"].std()

    print(f"\n{'='*65}")
    print(f"EVAL RESULTS — {label}")
    print(f"{'='*65}")
    print(f"  Samples          : {len(df):,}")
    print(f"  BLEU             : {bleu:.2f}")
    print(f"  Token-F1 (mean)  : {f1_mean:.4f}")
    print(f"  SBERT sim (mean) : {sim_mean:.4f}")
    print(f"  SBERT sim (med)  : {sim_med:.4f}")
    print(f"  SBERT sim (std)  : {sim_std:.4f}")
    print(f"  Avg tokens gen   : {df['n_tokens'].mean():.1f}")
    print(f"{'='*65}\n")

    return {
        "label":      label,
        "n":          len(df),
        "bleu":       round(bleu, 4),
        "f1_mean":    round(float(f1_mean), 4),
        "sbert_mean": round(float(sim_mean), 4),
        "sbert_med":  round(float(sim_med),  4),
        "sbert_std":  round(float(sim_std),  4),
        "avg_tokens": round(float(df["n_tokens"].mean()), 2),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("EVAL RAG — Phase 3")
    print("=" * 60)
    print(f"  Adapter     : {args.adapter_dir}")
    print(f"  Index       : {args.index_dir}")
    print(f"  Samples     : {args.n_samples or 'all'}")
    print(f"  k           : {args.k}")
    print(f"  Batch size  : {args.batch_size}")
    print()

    # ── 1. Load model ─────────────────────────
    print("Loading QLoRA model...")
    model, tokenizer = load_qlora_model(args.adapter_dir, base_model=args.base_model)

    # ── 2. Load RAG index ─────────────────────
    faiss_index, chunks_df, embed_model = load_index(args.index_dir)

    # ── 3. Load test set ──────────────────────
    print("Loading test dataset...")
    ds = load_doktorsitesi()
    test = ds["test"]
    if args.n_samples:
        test = test.select(range(min(args.n_samples, len(test))))

    total = len(test)
    rows  = []

    # ── 4. Batched retrieval + generation ─────
    for start in tqdm(range(0, total, args.batch_size), desc="Generating RAG answers"):
        batch = test.select(range(start, min(start + args.batch_size, total)))

        questions    = list(batch["question_content"])
        references   = list(batch["question_answer"])
        titles       = list(batch["doctor_title"])
        specialties  = list(batch["doctor_speciality"])

        # Retrieve for all questions in one encode call
        retrieved_batch = batch_retrieve(
            questions, faiss_index, chunks_df, embed_model, k=args.k
        )

        # Build prompts
        prompts = [
            build_rag_prompt(q, t, s, chunks)
            for q, t, s, chunks in zip(questions, titles, specialties, retrieved_batch)
        ]

        # Tokenize with left-padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]

        for i in range(len(questions)):
            gen_ids   = outputs[i][prompt_len:]
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
            rows.append({
                "question":         questions[i],
                "reference":        references[i],
                "generated":        generated,
                "prompt":           prompts[i],
                "n_tokens":         len(gen_ids),
                "retrieved_titles": " | ".join(
                    c["article_title"] for c in retrieved_batch[i]
                ),
            })

    df = pd.DataFrame(rows)

    # ── 5. BLEU ───────────────────────────────
    print("\nComputing BLEU...")
    bleu = compute_bleu(df)

    # ── 6. Token-F1 ───────────────────────────
    print("Computing token-F1...")
    df = compute_f1_column(df)

    # ── 7. SBERT similarity ───────────────────
    print("Computing SBERT similarity...")
    df["similarity"] = compute_similarity(df["reference"].tolist(), df["generated"].tolist())

    # ── 8. Summary + save ─────────────────────
    label = f"rag_k{args.k}_{os.path.basename(args.adapter_dir.rstrip('/'))}"
    summary = print_summary(label, bleu, df)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_csv(args.save_path, index=False)
    print(f"Per-sample results saved to {args.save_path}")

    summary_path = args.save_path.replace(".csv", "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
