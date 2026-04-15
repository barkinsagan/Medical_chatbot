"""
scripts/eval_gemini.py — Gemini-scored comparison of QLoRA vs RAG + paired t-test

For each sample:
  1. Generates a QLoRA answer (no retrieval)
  2. Generates a RAG answer (with retrieved context)
  3. Asks Gemini Flash to score both answers 1-10 on medical quality
  4. Runs a paired t-test to determine if the difference is statistically significant

Free tier: 15 req/min, 1500 req/day on Gemini Flash.
With 2 scoring calls per sample, 200 samples = 400 Gemini calls — well within daily limit.
Rate limiter built in to stay under 15 req/min.

Usage:
    python scripts/eval_gemini.py --n_samples 200
    python scripts/eval_gemini.py \
        --n_samples    200 \
        --k            5 \
        --adapter_dir  outputs/checkpoints/qlora \
        --index_dir    outputs/rag_index \
        --save_path    outputs/eval_results/gemini_comparison.csv

Requires:
    conda env config vars set GEMINI_API_KEY=your_key
    conda env config vars set HF_TOKEN=your_token
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import pandas as pd
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import google.generativeai as genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import load_index, build_rag_prompt
from src.data import load_doktorsitesi, INFERENCE_TEMPLATE
from scripts.eval_rag import batch_retrieve


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
GEMINI_MODEL = "gemini-2.0-flash"
RATE_LIMIT_DELAY = 4.5   # seconds between Gemini calls to stay under 15 req/min


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Gemini-scored QLoRA vs RAG comparison")
    parser.add_argument("--adapter_dir",    default="outputs/checkpoints/qlora")
    parser.add_argument("--base_model",     default=BASE_MODEL)
    parser.add_argument("--index_dir",      default="outputs/rag_index")
    parser.add_argument("--n_samples",      type=int, default=200,
                        help="Number of samples to compare (200 recommended)")
    parser.add_argument("--k",              type=int, default=5,
                        help="Chunks to retrieve per question")
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Generation batch size")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_path",      default="outputs/eval_results/gemini_comparison.csv")
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
    tokenizer.padding_side = "left"

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
# Batched generation
# ──────────────────────────────────────────────

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
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


# ──────────────────────────────────────────────
# Gemini scoring
# ──────────────────────────────────────────────

SCORING_PROMPT = """Bir Türk tıbbi soru-cevap sisteminin cevabını değerlendiriyorsun.

Hasta sorusu: {question}

Doktor cevabı: {answer}

Bu cevabı aşağıdaki kriterlere göre 1-10 arasında puanla:
- Tıbbi doğruluk ve kesinlik
- Hastanın sorusunu tam olarak yanıtlama
- Pratik ve uygulanabilir bilgi içerme

Sadece bir tam sayı yaz (1-10). Başka hiçbir şey yazma."""


def score_with_gemini(
    gemini_model,
    question: str,
    answer: str,
) -> float | None:
    """Asks Gemini to score a medical answer 1-10. Returns None on failure."""
    prompt = SCORING_PROMPT.format(question=question, answer=answer)
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        score = float(text.split()[0])
        if 1.0 <= score <= 10.0:
            return score
        return None
    except Exception as e:
        print(f"    [Gemini error] {e}")
        return None


# ──────────────────────────────────────────────
# Statistical test
# ──────────────────────────────────────────────

def paired_ttest(qlora_scores: list[float], rag_scores: list[float]) -> dict:
    """Paired t-test: tests whether RAG and QLoRA scores differ significantly."""
    t_stat, p_value = stats.ttest_rel(rag_scores, qlora_scores)
    n = len(qlora_scores)
    mean_diff = np.mean(rag_scores) - np.mean(qlora_scores)
    # 95% confidence interval on the mean difference
    se = stats.sem(np.array(rag_scores) - np.array(qlora_scores))
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se)
    return {
        "n":         n,
        "mean_qlora": np.mean(qlora_scores),
        "mean_rag":   np.mean(rag_scores),
        "mean_diff":  mean_diff,
        "t_stat":     t_stat,
        "p_value":    p_value,
        "ci_95_low":  ci_low,
        "ci_95_high": ci_high,
        "significant": p_value < 0.05,
    }


def print_stats(r: dict):
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISON — QLoRA vs RAG (Gemini scoring)")
    print(f"{'='*60}")
    print(f"  Samples            : {r['n']}")
    print(f"  QLoRA mean score   : {r['mean_qlora']:.3f} / 10")
    print(f"  RAG   mean score   : {r['mean_rag']:.3f} / 10")
    print(f"  Mean difference    : {r['mean_diff']:+.3f}  (RAG − QLoRA)")
    print(f"  95% CI             : [{r['ci_95_low']:+.3f}, {r['ci_95_high']:+.3f}]")
    print(f"  t-statistic        : {r['t_stat']:.4f}")
    print(f"  p-value            : {r['p_value']:.4f}")
    if r["significant"]:
        direction = "RAG is significantly better" if r["mean_diff"] > 0 else "QLoRA is significantly better"
        print(f"  Result             : p < 0.05 → {direction}")
    else:
        print(f"  Result             : p ≥ 0.05 → no statistically significant difference")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Check API key ─────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Run: conda env config vars set GEMINI_API_KEY=your_key"
        )
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    print("=" * 60)
    print("GEMINI COMPARISON — QLoRA vs RAG")
    print("=" * 60)
    print(f"  Samples    : {args.n_samples}")
    print(f"  k          : {args.k}")
    print(f"  Batch size : {args.batch_size}")
    print()

    # ── Load everything ───────────────────────
    print("Loading QLoRA model...")
    model, tokenizer = load_qlora_model(args.adapter_dir, args.base_model)

    print("Loading RAG index...")
    faiss_index, chunks_df, embed_model = load_index(args.index_dir)

    print("Loading test dataset...")
    ds = load_doktorsitesi()
    test = ds["test"].select(range(args.n_samples))
    total = len(test)

    # ── Generate answers in batches ───────────
    print("\nGenerating QLoRA and RAG answers...")
    all_qlora, all_rag, all_questions, all_references = [], [], [], []

    for start in tqdm(range(0, total, args.batch_size), desc="Generating"):
        batch = test.select(range(start, min(start + args.batch_size, total)))

        questions   = list(batch["question_content"])
        references  = list(batch["question_answer"])
        titles      = list(batch["doctor_title"])
        specialties = list(batch["doctor_speciality"])

        # QLoRA prompts (no RAG)
        qlora_prompts = [
            INFERENCE_TEMPLATE.format(
                doctor_speciality=s,
                doctor_title=t,
                question_content=q,
            )
            for q, t, s in zip(questions, titles, specialties)
        ]

        # RAG prompts
        retrieved_batch = batch_retrieve(
            questions, faiss_index, chunks_df, embed_model, k=args.k
        )
        rag_prompts = [
            build_rag_prompt(q, t, s, chunks)
            for q, t, s, chunks in zip(questions, titles, specialties, retrieved_batch)
        ]

        qlora_answers = generate_batch(model, tokenizer, qlora_prompts, args.max_new_tokens)
        rag_answers   = generate_batch(model, tokenizer, rag_prompts,   args.max_new_tokens)

        all_qlora.extend(qlora_answers)
        all_rag.extend(rag_answers)
        all_questions.extend(questions)
        all_references.extend(references)

    # ── Gemini scoring ────────────────────────
    print(f"\nScoring with Gemini ({GEMINI_MODEL}) — this will take ~{total * 2 * RATE_LIMIT_DELAY / 60:.0f} min...")
    qlora_scores, rag_scores = [], []
    rows = []

    for i in tqdm(range(total), desc="Gemini scoring"):
        q        = all_questions[i]
        qlora_a  = all_qlora[i]
        rag_a    = all_rag[i]
        ref      = all_references[i]

        score_qlora = score_with_gemini(gemini_model, q, qlora_a)
        time.sleep(RATE_LIMIT_DELAY)
        score_rag   = score_with_gemini(gemini_model, q, rag_a)
        time.sleep(RATE_LIMIT_DELAY)

        if score_qlora is None or score_rag is None:
            print(f"  Skipping sample {i+1} — Gemini returned invalid score")
            continue

        qlora_scores.append(score_qlora)
        rag_scores.append(score_rag)

        rows.append({
            "question":      q,
            "reference":     ref,
            "qlora_answer":  qlora_a,
            "rag_answer":    rag_a,
            "qlora_score":   score_qlora,
            "rag_score":     score_rag,
            "score_delta":   score_rag - score_qlora,
        })

    # ── Statistical test ──────────────────────
    stats_result = paired_ttest(qlora_scores, rag_scores)
    print_stats(stats_result)

    # ── Save results ──────────────────────────
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_csv(args.save_path, index=False)
    print(f"Detailed results saved to {args.save_path}")

    # Save stats summary separately
    stats_path = args.save_path.replace(".csv", "_stats.csv")
    pd.DataFrame([stats_result]).to_csv(stats_path, index=False)
    print(f"Stats summary saved to {stats_path}")


if __name__ == "__main__":
    main()
