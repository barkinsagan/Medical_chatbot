"""
scripts/test_retrieval.py — Inspect RAG retrieval quality + side-by-side answer comparison

For each sample question prints:
  1. The retrieved chunks + scores
  2. The QLoRA answer (no RAG)
  3. The RAG answer (QLoRA + retrieved context)
  4. The reference (real doctor answer)
  5. Cosine similarity scores for both answers vs reference

Usage:
    python scripts/test_retrieval.py                        # retrieval only, no model
    python scripts/test_retrieval.py --compare              # load model + show side-by-side
    python scripts/test_retrieval.py --compare \
        --adapter_dir outputs/checkpoints/qlora \
        --n_questions 5 --k 3
"""

import argparse
import os
import sys

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import load_index, retrieve, build_rag_prompt
from src.data import load_doktorsitesi, INFERENCE_TEMPLATE


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MPNET_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect retrieval quality + compare answers")
    parser.add_argument("--index_dir",     default="outputs/rag_index")
    parser.add_argument("--adapter_dir",   default="outputs/checkpoints/qlora")
    parser.add_argument("--base_model",    default=BASE_MODEL)
    parser.add_argument("--n_questions",   type=int, default=5)
    parser.add_argument("--k",             type=int, default=3)
    parser.add_argument("--max_new_tokens",type=int, default=256)
    parser.add_argument("--chunk_preview", type=int, default=300,
                        help="Characters of each chunk to preview")
    parser.add_argument("--answer_preview",type=int, default=400,
                        help="Characters of each answer to preview")
    parser.add_argument("--compare",       action="store_true",
                        help="Load the model and show side-by-side answer comparison")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_qlora_model(adapter_dir: str, base_model: str):
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def cosine_sim(a: str, b: str, sbert) -> float:
    embs = sbert.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(embs[0], embs[1]))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading RAG index...")
    faiss_index, chunks_df, embed_model = load_index(args.index_dir)

    model, tokenizer, sbert = None, None, None
    if args.compare:
        print("Loading QLoRA model...")
        model, tokenizer = load_qlora_model(args.adapter_dir, args.base_model)
        print("Loading SBERT for scoring...")
        sbert = SentenceTransformer(MPNET_NAME)

    print("Loading test questions...")
    ds = load_doktorsitesi()
    test = ds["test"].select(range(args.n_questions))

    for i, example in enumerate(test):
        question   = example["question_content"]
        reference  = example["question_answer"]
        title      = example["doctor_title"]
        specialty  = example["doctor_speciality"]

        print(f"\n{'='*70}")
        print(f"QUESTION {i+1}")
        print(f"{'='*70}")
        print(f"  {question}")

        # ── Retrieved chunks ───────────────────
        hits = retrieve(question, faiss_index, chunks_df, embed_model, k=args.k)
        avg_score = sum(h["score"] for h in hits) / len(hits) if hits else 0

        print(f"\n  --- Retrieved chunks (avg score: {avg_score:.4f}) ---")
        for j, hit in enumerate(hits):
            print(f"\n  [{j+1}] score={hit['score']:.4f} | {hit['hospital']} | {hit['article_title']}")
            print(f"       {hit['chunk_text'][:args.chunk_preview]}")

        if args.compare and model is not None:
            # ── QLoRA answer (no RAG) ──────────
            plain_prompt = INFERENCE_TEMPLATE.format(
                doctor_speciality=specialty,
                doctor_title=title,
                question_content=question,
            )
            qlora_answer = generate(model, tokenizer, plain_prompt, args.max_new_tokens)

            # ── RAG answer ─────────────────────
            rag_prompt  = build_rag_prompt(question, title, specialty, hits)
            rag_answer  = generate(model, tokenizer, rag_prompt, args.max_new_tokens)

            # ── Similarity scores ──────────────
            sim_qlora = cosine_sim(reference, qlora_answer, sbert)
            sim_rag   = cosine_sim(reference, rag_answer,   sbert)

            print(f"\n  --- Reference (real doctor) ---")
            print(f"  {reference[:args.answer_preview]}")

            print(f"\n  --- QLoRA answer (no RAG) | sim={sim_qlora:.4f} ---")
            print(f"  {qlora_answer[:args.answer_preview]}")

            print(f"\n  --- RAG answer | sim={sim_rag:.4f} ---")
            print(f"  {rag_answer[:args.answer_preview]}")

            delta = sim_rag - sim_qlora
            direction = "RAG better" if delta > 0 else "QLoRA better"
            print(f"\n  Similarity delta: {delta:+.4f}  →  {direction}")

    print(f"\n{'='*70}")
    if args.compare:
        print("INTERPRETATION")
        print(f"{'='*70}")
        print("  Read the answers qualitatively — does the RAG answer use the")
        print("  retrieved context? Is it more detailed/accurate than QLoRA alone?")
        print("  If yes but similarity is lower, the metric is penalising style,")
        print("  not factual quality. That is a thesis finding worth reporting.")
    else:
        print("Run with --compare to also load the model and see answer comparisons.")


if __name__ == "__main__":
    main()
