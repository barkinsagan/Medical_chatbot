"""
eval_all.py — Auto-discover and compare all trained adapters under outputs/checkpoints/

Scans outputs/checkpoints/ for any directory containing adapter_config.json,
reads the base model from it, and evaluates all of them.

Usage:
    python eval_all.py                        # eval all discovered adapters
    python eval_all.py --n_samples 500        # quick run on a subset
    python eval_all.py --checkpoints_dir outputs/checkpoints
"""

import argparse
import json
import os
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from src.data import load_doktorsitesi
from src.evaluate import evaluate

RESULTS_DIR = "outputs/eval_results"


# ──────────────────────────────────────────────
# Discover adapters
# ──────────────────────────────────────────────

def discover_adapters(checkpoints_dir: str) -> list[dict]:
    """
    Walks checkpoints_dir and returns a list of adapter dicts for every
    subdirectory that contains an adapter_config.json.
    """
    adapters = []

    if not os.path.isdir(checkpoints_dir):
        print(f"[ERROR] Checkpoints directory not found: {checkpoints_dir}")
        return adapters

    for entry in sorted(os.scandir(checkpoints_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        config_path = os.path.join(entry.path, "adapter_config.json")
        if not os.path.isfile(config_path):
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        base_model = cfg.get("base_model_name_or_path", "meta-llama/Meta-Llama-3-8B-Instruct")
        # Normalize unsloth model names → HF equivalents
        base_model = base_model.replace("unsloth/llama-3-8b-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")

        adapters.append({
            "label":        entry.name,
            "adapter_path": entry.path,
            "base_model":   base_model,
        })
        print(f"  Found: {entry.name}  (base: {base_model})")

    return adapters


# ──────────────────────────────────────────────
# Load base model
# ──────────────────────────────────────────────

def load_base(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


# ──────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────

def print_comparison(rows: list[dict]):
    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)
    print(f"{'Model':<25} {'Samples':>8} {'Avg Sim':>9} {'Med Sim':>9} {'Std':>7} {'Avg Tok':>8} {'Time(s)':>8}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['label']:<25} "
            f"{r['n_samples']:>8,} "
            f"{r['avg_sim']:>9.4f} "
            f"{r['med_sim']:>9.4f} "
            f"{r['std_sim']:>7.4f} "
            f"{r['avg_tokens']:>8.1f} "
            f"{r['total_time']:>8.1f}"
        )
    print("=" * 75)
    best = max(rows, key=lambda x: x["avg_sim"])
    print(f"\nBest avg similarity : {best['label']} ({best['avg_sim']:.4f})")
    print(f"Paper base LLaMA 3  : 0.4627")
    print(f"Paper fine-tuned    : 0.5151\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="outputs/checkpoints",
                        help="Directory to scan for adapter folders")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Test samples per model (default: all 37,527)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Samples per batch (increase if VRAM allows, e.g. 16)")
    args = parser.parse_args()

    print(f"Scanning {args.checkpoints_dir} for adapters...")
    adapters = discover_adapters(args.checkpoints_dir)

    if not adapters:
        print("No adapters found. Train a model first.")
        return

    print(f"\nFound {len(adapters)} adapter(s). Loading dataset...")
    ds = load_doktorsitesi()
    test_dataset = ds["test"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []

    for adapter in adapters:
        label        = adapter["label"]
        adapter_path = adapter["adapter_path"]
        base_model   = adapter["base_model"]

        print(f"\n{'='*60}")
        print(f"Evaluating : {label}")
        print(f"Base model : {base_model}")
        print(f"Adapter    : {adapter_path}")
        print(f"{'='*60}")

        base, tokenizer = load_base(base_model)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        t0 = time.time()
        save_path = os.path.join(RESULTS_DIR, f"{label}.csv")

        df = evaluate(
            model, tokenizer, test_dataset,
            n_samples=args.n_samples,
            label=label,
            save_path=save_path,
            batch_size=args.batch_size,
        )
        total_time = time.time() - t0

        summary_rows.append({
            "label":      label,
            "n_samples":  len(df),
            "avg_sim":    df["similarity"].mean(),
            "med_sim":    df["similarity"].median(),
            "std_sim":    df["similarity"].std(),
            "avg_tokens": df["n_tokens"].mean(),
            "total_time": total_time,
        })

        del model, base
        torch.cuda.empty_cache()

    if summary_rows:
        print_comparison(summary_rows)
        compare_path = os.path.join(RESULTS_DIR, "comparison.csv")
        pd.DataFrame(summary_rows).to_csv(compare_path, index=False)
        print(f"Comparison saved to {compare_path}")


if __name__ == "__main__":
    main()
