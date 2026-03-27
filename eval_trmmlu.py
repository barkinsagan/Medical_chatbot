"""
eval_trmmlu.py — TR-MMLU benchmark evaluation for catastrophic forgetting measurement

Evaluates models on alibayram/turkish_mmlu (6,200 MCQ, 62 sections, 4-choice A–D).
Uses log-probability scoring (no generation): picks the answer token with highest logprob.

Paper baselines (Bayram et al., 2025, DOI: 10.1145/3772000):
  - Fine-tuned only (before Slerp): 19/100
  - Slerp-merged:                   53/100

Usage:
    python eval_trmmlu.py                             # eval base + all adapters in outputs/checkpoints
    python eval_trmmlu.py --n_samples 200             # quick smoke test
    python eval_trmmlu.py --checkpoints_dir outputs/checkpoints
    python eval_trmmlu.py --slerp_path outputs/checkpoints/slerp_merged
"""

import argparse
import json
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Answer choices — we score log-prob of these single tokens
CHOICES = ["A", "B", "C", "D"]

# Paper baselines for the comparison table
PAPER_BASELINES = {
    "paper_finetuned_only": 19.0,
    "paper_slerp_merged":   53.0,
}

# MCQ prompt template — present question + options, ask for the answer letter
MCQ_TEMPLATE = """{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Cevap:"""


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_trmmlu(n_samples: int = None):
    """
    Loads alibayram/turkish_mmlu.
    The dataset has a single 'test' split with columns:
      question, option_a, option_b, option_c, option_d, answer (one of A/B/C/D), category
    """
    print("Loading alibayram/turkish_mmlu...")
    ds = load_dataset("alibayram/turkish_mmlu", split="test")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    print(f"  {len(ds):,} questions, {ds.unique('category').__len__()} sections")
    return ds


def format_prompt(example: dict) -> str:
    return MCQ_TEMPLATE.format(
        question=example["question"],
        option_a=example["option_a"],
        option_b=example["option_b"],
        option_c=example["option_c"],
        option_d=example["option_d"],
    )


# ──────────────────────────────────────────────
# Log-probability scoring
# ──────────────────────────────────────────────

def get_answer_token_ids(tokenizer) -> list[int]:
    """
    Returns the single-token ID for each of A, B, C, D.
    Verifies each encodes to exactly one token (important for correct scoring).
    """
    token_ids = []
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            # Try with a leading space (some tokenizers need it)
            ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        assert len(ids) == 1, (
            f"Choice '{choice}' tokenizes to {len(ids)} tokens — "
            "cannot do single-token log-prob scoring. Check tokenizer."
        )
        token_ids.append(ids[0])
    return token_ids  # [id_A, id_B, id_C, id_D]


@torch.no_grad()
def score_example(model, tokenizer, prompt: str, answer_token_ids: list[int]) -> int:
    """
    Returns the index (0–3) of the answer choice with the highest log-probability.
    Does a single forward pass and reads logits at the last prompt position.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model(**inputs)

    # Logits at the last token position (where the model would predict next token)
    last_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    # Score only the 4 answer tokens
    scores = torch.tensor([last_logits[tid].item() for tid in answer_token_ids])
    return int(scores.argmax().item())


# ──────────────────────────────────────────────
# Evaluate one model
# ──────────────────────────────────────────────

def evaluate_model(model, tokenizer, dataset, label: str) -> pd.DataFrame:
    """
    Runs TR-MMLU evaluation on the given model.
    Returns a DataFrame with one row per question.
    """
    model.eval()
    answer_token_ids = get_answer_token_ids(tokenizer)

    letter_to_idx = {c: i for i, c in enumerate(CHOICES)}

    rows = []
    correct = 0

    for i, example in enumerate(dataset):
        if (i + 1) % 500 == 0:
            acc_so_far = correct / (i + 1) * 100
            print(f"  [{label}] {i+1}/{len(dataset)} — running accuracy: {acc_so_far:.1f}/100")

        prompt    = format_prompt(example)
        pred_idx  = score_example(model, tokenizer, prompt, answer_token_ids)
        pred_char = CHOICES[pred_idx]
        true_char = example["answer"].strip().upper()

        is_correct = pred_char == true_char
        if is_correct:
            correct += 1

        rows.append({
            "category":  example["category"],
            "question":  example["question"][:120],
            "true":      true_char,
            "predicted": pred_char,
            "correct":   is_correct,
        })

    df = pd.DataFrame(rows)
    overall = correct / len(df) * 100
    print(f"\n  [{label}] Overall accuracy: {overall:.1f}/100  ({correct}/{len(df)})")
    return df


# ──────────────────────────────────────────────
# Per-section breakdown
# ──────────────────────────────────────────────

def section_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Returns per-section accuracy sorted by accuracy descending."""
    grp = df.groupby("category")["correct"].agg(["sum", "count"])
    grp["accuracy"] = grp["sum"] / grp["count"] * 100
    grp.columns = ["correct", "total", "accuracy"]
    return grp.sort_values("accuracy", ascending=False).reset_index()


def print_section_breakdown(sec: pd.DataFrame):
    print(f"\n{'Section':<45} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("-" * 73)
    for _, row in sec.iterrows():
        print(f"{row['category']:<45} {int(row['correct']):>8} {int(row['total']):>7} {row['accuracy']:>9.1f}%")
    print("-" * 73)


# ──────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────

def load_base_model(model_name: str):
    print(f"  Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def load_adapter_model(base_model_name: str, adapter_path: str):
    print(f"  Loading adapter: {adapter_path}  (base: {base_model_name})")
    base, tokenizer = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tokenizer


def discover_adapters(checkpoints_dir: str) -> list[dict]:
    """Same logic as eval_all.py — finds all adapter_config.json subdirs."""
    adapters = []
    if not os.path.isdir(checkpoints_dir):
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
        base_model = base_model.replace(
            "unsloth/llama-3-8b-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        adapters.append({
            "label":        entry.name,
            "adapter_path": entry.path,
            "base_model":   base_model,
        })
        print(f"  Found adapter: {entry.name}  (base: {base_model})")

    return adapters


def discover_slerp_models(slerp_dir: str) -> list[dict]:
    """
    Finds standalone merged models in slerp_dir.
    Looks for subdirs with config.json but no adapter_config.json
    (i.e. full merged models, not adapters).
    """
    models = []
    if not os.path.isdir(slerp_dir):
        return models

    for entry in sorted(os.scandir(slerp_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        has_config   = os.path.isfile(os.path.join(entry.path, "config.json"))
        has_adapter  = os.path.isfile(os.path.join(entry.path, "adapter_config.json"))
        if has_config and not has_adapter:
            models.append({"label": entry.name, "path": entry.path})
            print(f"  Found slerp model: {entry.name}")

    return models


# ──────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────

def print_comparison(rows: list[dict]):
    print("\n" + "=" * 65)
    print("TR-MMLU COMPARISON  (paper format: X/100)")
    print("=" * 65)
    print(f"{'Model':<35} {'Correct':>8} {'Total':>7} {'Score':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{r['label']:<35} {r['correct']:>8} {r['total']:>7} {r['score']:>7.1f}/100")
    print("-" * 65)
    print(f"{'[Paper] Fine-tuned only (before Slerp)':<35} {'':>8} {'':>7} {'19.0':>8}/100")
    print(f"{'[Paper] Slerp-merged':<35} {'':>8} {'':>7} {'53.0':>8}/100")
    print("=" * 65)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TR-MMLU evaluation for catastrophic forgetting")
    parser.add_argument("--checkpoints_dir", default="outputs/checkpoints",
                        help="Directory to scan for LoRA adapter folders")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Base model HF path (used for base eval and as fallback for adapters)")
    parser.add_argument("--slerp_paths", nargs="+", default=None, metavar="PATH",
                        help="One or more paths to Slerp-merged standalone models")
    parser.add_argument("--slerp_dir", default=None, metavar="DIR",
                        help="Directory to auto-discover all Slerp-merged models (looks for config.json subdirs)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of TR-MMLU questions to evaluate (default: all 6,200)")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip base model evaluation (useful when re-running adapter evals)")
    args = parser.parse_args()

    # Load dataset once
    dataset = load_trmmlu(n_samples=args.n_samples)

    summary_rows = []

    # ── 1. Base model ────────────────────────────
    if not args.skip_base:
        print(f"\n{'='*60}")
        print(f"Evaluating: base model  ({args.base_model})")
        print(f"{'='*60}")
        model, tokenizer = load_base_model(args.base_model)
        df = evaluate_model(model, tokenizer, dataset, label="base_model")
        print_section_breakdown(section_breakdown(df))

        summary_rows.append({
            "label":   "base_model",
            "correct": int(df["correct"].sum()),
            "total":   len(df),
            "score":   df["correct"].mean() * 100,
        })

        del model
        torch.cuda.empty_cache()

    # ── 2. Adapter models ────────────────────────
    adapters = discover_adapters(args.checkpoints_dir)

    if not adapters:
        print(f"\nNo adapters found in {args.checkpoints_dir}.")
    else:
        print(f"\nFound {len(adapters)} adapter(s).")

    for adapter in adapters:
        label        = adapter["label"]
        adapter_path = adapter["adapter_path"]
        base_model   = adapter["base_model"]

        print(f"\n{'='*60}")
        print(f"Evaluating adapter: {label}")
        print(f"{'='*60}")

        model, tokenizer = load_adapter_model(base_model, adapter_path)
        df = evaluate_model(model, tokenizer, dataset, label=label)
        print_section_breakdown(section_breakdown(df))

        summary_rows.append({
            "label":   label,
            "correct": int(df["correct"].sum()),
            "total":   len(df),
            "score":   df["correct"].mean() * 100,
        })

        del model
        torch.cuda.empty_cache()

    # ── 3. Slerp-merged models (optional) ────────
    slerp_models = []

    # Explicit paths: --slerp_paths path1 path2 ...
    if args.slerp_paths:
        for path in args.slerp_paths:
            slerp_models.append({"label": os.path.basename(path.rstrip("/")) or "slerp_merged", "path": path})

    # Auto-discovery: --slerp_dir outputs/checkpoints/slerp/
    if args.slerp_dir:
        print(f"\nScanning {args.slerp_dir} for Slerp models...")
        slerp_models.extend(discover_slerp_models(args.slerp_dir))

    for slerp in slerp_models:
        print(f"\n{'='*60}")
        print(f"Evaluating Slerp model: {slerp['label']}  ({slerp['path']})")
        print(f"{'='*60}")

        model, tokenizer = load_base_model(slerp["path"])
        df = evaluate_model(model, tokenizer, dataset, label=slerp["label"])
        print_section_breakdown(section_breakdown(df))

        summary_rows.append({
            "label":   slerp["label"],
            "correct": int(df["correct"].sum()),
            "total":   len(df),
            "score":   df["correct"].mean() * 100,
        })

        del model
        torch.cuda.empty_cache()

    # ── 4. Summary ───────────────────────────────
    if summary_rows:
        print_comparison(summary_rows)


if __name__ == "__main__":
    main()
