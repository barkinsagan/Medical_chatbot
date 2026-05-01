"""
scripts/eval_metrics.py — BLEU, token-F1, and SBERT similarity for a trained adapter

Given a config YAML (same ones used for training), loads the corresponding adapter
or Slerp-merged model and evaluates it on the full doktorsitesi test split.

Metrics:
    BLEU        — sacrebleu corpus BLEU (tokenize="intl", handles Turkish unicode)
    Token-F1    — unigram overlap F1 averaged per sample (same as SQuAD F1)
    SBERT sim   — cosine similarity via paraphrase-multilingual-mpnet-base-v2

Usage:
    # Adapter checkpoint (path taken from config's output_dir)
    python scripts/eval_metrics.py --config configs/qlora.yaml

    # Slerp-merged standalone model
    python scripts/eval_metrics.py --config configs/qlora.yaml \
        --slerp_path outputs/merged/qlora_t0.50

    # Override adapter path explicitly
    python scripts/eval_metrics.py --config configs/qlora.yaml \
        --adapter_path outputs/checkpoints/qlora

    # Quick smoke test
    python scripts/eval_metrics.py --config configs/qlora.yaml --n_samples 200

Output:
    <save_path>            — per-sample CSV  (bleu_1, f1, similarity, ...)
    <save_path>_summary    — one-row summary CSV with all aggregate metrics
"""

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import pandas as pd
import sacrebleu
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_doktorsitesi
from src.train import load_config


SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Compute BLEU, F1, SBERT for a trained adapter")
    parser.add_argument("--config",       required=True,
                        help="Path to training config YAML (e.g. configs/qlora.yaml)")
    parser.add_argument("--adapter_path", default=None,
                        help="Override adapter path (default: config's output_dir)")
    parser.add_argument("--slerp_path",   default=None,
                        help="Path to a Slerp-merged standalone model (skips adapter loading)")
    parser.add_argument("--n_samples",    type=int, default=None,
                        help="Number of test samples (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--save_path",    default=None,
                        help="Path for per-sample CSV output (default: derived from config)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_adapter(cfg: dict, adapter_path: str):
    bnb_config = None
    if cfg.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def load_standalone(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

def generate_responses(model, tokenizer, test_dataset, n_samples, max_new_tokens, batch_size) -> pd.DataFrame:
    if n_samples is not None:
        test_dataset = test_dataset.select(range(min(n_samples, len(test_dataset))))

    rows = []
    total = len(test_dataset)

    for start in tqdm(range(0, total, batch_size), desc="Generating"):
        batch     = test_dataset.select(range(start, min(start + batch_size, total)))
        prompts   = list(batch["prompt"])
        refs      = list(batch["reference"])

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = (time.time() - t0) / len(prompts)

        prompt_len = inputs["input_ids"].shape[1]
        for i, (prompt, ref) in enumerate(zip(prompts, refs)):
            gen_ids   = outputs[i][prompt_len:]
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
            rows.append({
                "prompt":    prompt,
                "reference": ref,
                "generated": generated,
                "n_tokens":  len(gen_ids),
                "time_sec":  round(elapsed, 3),
            })

    return pd.DataFrame(rows)


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
    hypotheses = df["generated"].tolist()
    references = [df["reference"].tolist()]
    result = sacrebleu.corpus_bleu(hypotheses, references, tokenize="intl")
    return result.score  # 0–100


def compute_f1_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["f1"] = [
        compute_token_f1(gen, ref)
        for gen, ref in zip(df["generated"], df["reference"])
    ]
    return df


def compute_sbert(df: pd.DataFrame) -> pd.DataFrame:
    sbert = SentenceTransformer(SBERT_MODEL)
    print("  Encoding references...")
    ref_emb = sbert.encode(df["reference"].tolist(), batch_size=64,
                           show_progress_bar=True, convert_to_tensor=True)
    print("  Encoding generated...")
    gen_emb = sbert.encode(df["generated"].tolist(), batch_size=64,
                           show_progress_bar=True, convert_to_tensor=True)
    sims = util.cos_sim(ref_emb, gen_emb).diagonal().cpu().numpy()
    df = df.copy()
    df["similarity"] = sims
    return df


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

def print_summary(label: str, bleu: float, df: pd.DataFrame):
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
        "label":    label,
        "n":        len(df),
        "bleu":     round(bleu, 4),
        "f1_mean":  round(float(f1_mean), 4),
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

    cfg = load_config(args.config)

    # Determine label and model source
    if args.slerp_path:
        label       = os.path.basename(args.slerp_path.rstrip("/"))
        model_source = f"slerp: {args.slerp_path}"
    else:
        adapter_path = args.adapter_path or cfg["output_dir"]
        label        = os.path.basename(adapter_path.rstrip("/"))
        model_source = f"adapter: {adapter_path}"

    save_path = args.save_path or os.path.join(
        "outputs/eval_results", f"{label}_metrics.csv"
    )

    print("=" * 65)
    print("EVAL METRICS")
    print("=" * 65)
    print(f"  Config      : {args.config}")
    print(f"  Model       : {model_source}")
    print(f"  n_samples   : {args.n_samples or 'all'}")
    print(f"  max_new_tok : {args.max_new_tokens}")
    print(f"  Save path   : {save_path}")
    print()

    # ── Load model ────────────────────────────
    print("Loading model...")
    if args.slerp_path:
        model, tokenizer = load_standalone(args.slerp_path)
    else:
        model, tokenizer = load_adapter(cfg, adapter_path)

    # ── Load data ─────────────────────────────
    print("Loading dataset...")
    ds = load_doktorsitesi()
    test_dataset = ds["test"]

    # ── Generate ──────────────────────────────
    print(f"\nGenerating responses...")
    df = generate_responses(
        model, tokenizer, test_dataset,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    del model
    torch.cuda.empty_cache()

    # ── BLEU ──────────────────────────────────
    print("\nComputing BLEU...")
    bleu = compute_bleu(df)

    # ── Token-F1 ──────────────────────────────
    print("Computing token-F1...")
    df = compute_f1_column(df)

    # ── SBERT ─────────────────────────────────
    print("Computing SBERT similarity...")
    df = compute_sbert(df)

    # ── Summary + save ────────────────────────
    summary = print_summary(label, bleu, df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Per-sample results saved to {save_path}")

    summary_path = save_path.replace(".csv", "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
