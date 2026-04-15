"""
scripts/generate_answers.py — Generate QLoRA and RAG answers and save to CSV

Runs on the lab computer (4090). No scoring — just generation.
The output CSV is passed to scripts/score_answers.py (on Colab or elsewhere)
for independent scoring with a stronger model.

Output CSV columns:
    question, reference, doctor_title, doctor_speciality,
    qlora_answer, rag_answer

Usage:
    python scripts/generate_answers.py

    python scripts/generate_answers.py \
        --adapter_dir  outputs/checkpoints/qlora \
        --index_dir    outputs/rag_index \
        --n_samples    200 \
        --k            5 \
        --batch_size   8 \
        --save_path    outputs/eval_results/answers.csv

    # SLERP merged model
    python scripts/generate_answers.py \
        --adapter_dir  outputs/merged/qlora_t0.50 \
        --merged \
        --save_path    outputs/eval_results/answers_slerp.csv

Requires:
    conda env config vars set HF_TOKEN=your_token
"""

import argparse
import os
import sys

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import load_index, build_rag_prompt
from src.data import load_doktorsitesi, INFERENCE_TEMPLATE
from scripts.eval_rag import batch_retrieve


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QLoRA and RAG answers")
    parser.add_argument("--adapter_dir",    default="outputs/checkpoints/qlora",
                        help="Path to QLoRA adapter OR merged SLERP model directory")
    parser.add_argument("--merged",         action="store_true",
                        help="Set if adapter_dir is a merged model (e.g. SLERP), not a PEFT adapter")
    parser.add_argument("--base_model",     default=BASE_MODEL)
    parser.add_argument("--index_dir",      default="outputs/rag_index")
    parser.add_argument("--n_samples",      type=int, default=200)
    parser.add_argument("--k",              type=int, default=5,
                        help="Chunks to retrieve per question")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_path",      default="outputs/eval_results/answers.csv")
    return parser.parse_args()


def load_model(adapter_dir: str, base_model: str, merged: bool = False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_path = adapter_dir if merged else base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if not merged:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        for i in range(len(prompts))
    ]


def main():
    args = parse_args()

    print("=" * 60)
    print("GENERATE ANSWERS")
    print("=" * 60)
    model_label = f"SLERP ({args.adapter_dir})" if args.merged else f"QLoRA ({args.adapter_dir})"
    print(f"  Model      : {model_label}")
    print(f"  Index      : {args.index_dir}")
    print(f"  Samples    : {args.n_samples}")
    print(f"  k          : {args.k}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Save path  : {args.save_path}")
    print()

    print("Loading model...")
    model, tokenizer = load_model(args.adapter_dir, args.base_model, merged=args.merged)

    print("Loading RAG index...")
    faiss_index, chunks_df, embed_model = load_index(args.index_dir)

    print("Loading test dataset...")
    ds = load_doktorsitesi()
    test = ds["test"].select(range(args.n_samples))
    total = len(test)

    rows = []
    for start in tqdm(range(0, total, args.batch_size), desc="Generating"):
        batch = test.select(range(start, min(start + args.batch_size, total)))

        questions   = list(batch["question_content"])
        references  = list(batch["question_answer"])
        titles      = list(batch["doctor_title"])
        specialties = list(batch["doctor_speciality"])

        qlora_prompts = [
            INFERENCE_TEMPLATE.format(
                doctor_speciality=s,
                doctor_title=t,
                question_content=q,
            )
            for q, t, s in zip(questions, titles, specialties)
        ]

        retrieved_batch = batch_retrieve(
            questions, faiss_index, chunks_df, embed_model, k=args.k
        )
        rag_prompts = [
            build_rag_prompt(q, t, s, chunks)
            for q, t, s, chunks in zip(questions, titles, specialties, retrieved_batch)
        ]

        qlora_answers = generate_batch(model, tokenizer, qlora_prompts, args.max_new_tokens)
        rag_answers   = generate_batch(model, tokenizer, rag_prompts,   args.max_new_tokens)

        for q, ref, t, s, qa, ra in zip(
            questions, references, titles, specialties, qlora_answers, rag_answers
        ):
            rows.append({
                "question":          q,
                "reference":         ref,
                "doctor_title":      t,
                "doctor_speciality": s,
                "qlora_answer":      qa,
                "rag_answer":        ra,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_csv(args.save_path, index=False)
    print(f"\nSaved {len(df)} rows to {args.save_path}")
    print("Transfer this file to Colab and run scripts/score_answers.py")


if __name__ == "__main__":
    main()
