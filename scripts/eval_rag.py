"""
scripts/eval_rag.py — RAG evaluation pipeline (Phase 3)

Loads the QLoRA adapter via HuggingFace transformers + PEFT (no Unsloth),
retrieves context from the FAISS index, generates RAG answers on the test
set, and computes cosine similarity against reference answers (same metric
as Phase 1/2).

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py \
        --adapter_dir outputs/checkpoints/qlora \
        --index_dir   outputs/rag_index \
        --n_samples   100 \
        --k           3 \
        --save_path   outputs/eval_results/rag_eval.csv

Requires:
    export HF_TOKEN=hf_...
"""

import argparse
import os
import sys

import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import load_index, retrieve, build_rag_prompt, generate_rag
from src.data import load_doktorsitesi


BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
MPNET_NAME  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--adapter_dir",  default="outputs/checkpoints/qlora",
                        help="Path to QLoRA adapter directory")
    parser.add_argument("--base_model",   default=BASE_MODEL,
                        help="HuggingFace base model ID")
    parser.add_argument("--index_dir",    default="outputs/rag_index",
                        help="Path to directory with index.faiss and chunks.parquet")
    parser.add_argument("--n_samples",    type=int, default=None,
                        help="Number of test samples to evaluate (default: all)")
    parser.add_argument("--k",            type=int, default=3,
                        help="Number of chunks to retrieve per query")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_path",    default="outputs/eval_results/rag_eval.csv",
                        help="Where to save the results CSV")
    return parser.parse_args()


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

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def compute_similarity(references: list[str], generated: list[str], batch_size: int = 64) -> list[float]:
    """Cosine similarity between reference and generated answers via mpnet."""
    sbert = SentenceTransformer(MPNET_NAME)
    ref_emb = sbert.encode(references, batch_size=batch_size, show_progress_bar=True,
                           convert_to_tensor=True, normalize_embeddings=True)
    gen_emb = sbert.encode(generated,  batch_size=batch_size, show_progress_bar=True,
                           convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(ref_emb, gen_emb).diagonal().cpu().tolist()
    return scores


def print_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY — RAG (Phase 3)")
    print(f"{'='*60}")
    print(f"  Samples evaluated : {len(df):,}")
    print(f"  Avg similarity    : {df['similarity'].mean():.4f}")
    print(f"  Median similarity : {df['similarity'].median():.4f}")
    print(f"  Std similarity    : {df['similarity'].std():.4f}")
    print(f"  Min / Max         : {df['similarity'].min():.4f} / {df['similarity'].max():.4f}")
    print(f"  Avg tokens gen    : {df['n_tokens'].mean():.1f}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    print("=" * 60)
    print("EVAL RAG — Phase 3")
    print("=" * 60)
    print(f"  Adapter    : {args.adapter_dir}")
    print(f"  Index      : {args.index_dir}")
    print(f"  Samples    : {args.n_samples}")
    print(f"  k          : {args.k}")
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

    # ── 4. Generate RAG answers ───────────────
    rows = []
    for example in tqdm(test, desc="Generating RAG answers"):
        result = generate_rag(
            model=model,
            tokenizer=tokenizer,
            question=example["question_content"],
            doctor_title=example["doctor_title"],
            doctor_specialty=example["doctor_speciality"],
            faiss_index=faiss_index,
            chunks_df=chunks_df,
            embedding_model=embed_model,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append({
            "question":         example["question_content"],
            "reference":        example["question_answer"],
            "generated":        result["generated_answer"],
            "prompt":           result["prompt"],
            "n_tokens":         len(tokenizer.encode(result["generated_answer"])),
            "retrieved_titles": " | ".join(c["article_title"] for c in result["retrieved_chunks"]),
        })

    df = pd.DataFrame(rows)

    # ── 5. Similarity scoring ─────────────────
    print("\nComputing cosine similarity...")
    df["similarity"] = compute_similarity(df["reference"].tolist(), df["generated"].tolist())

    # ── 6. Summary + save ─────────────────────
    print_summary(df)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_csv(args.save_path, index=False)
    print(f"Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
