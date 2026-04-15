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
    # QLoRA adapter (base + adapter)
    python scripts/eval_gemini.py --n_samples 200

    # SLERP merged model (pass --merged, point adapter_dir at the merged model)
    python scripts/eval_gemini.py \
        --adapter_dir  outputs/merged/qlora_t0.50 \
        --merged \
        --n_samples    200 \
        --k            5 \
        --save_path    outputs/eval_results/gemini_slerp_rag.csv

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
    parser.add_argument("--adapter_dir",    default="outputs/checkpoints/qlora",
                        help="Path to QLoRA adapter OR merged SLERP model directory")
    parser.add_argument("--merged",         action="store_true",
                        help="Set this if adapter_dir is a merged model (e.g. SLERP), not a PEFT adapter")
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

def load_model(adapter_dir: str, base_model: str, merged: bool = False):
    """
    Loads the model for inference.
    - merged=False: loads base_model in 4-bit + applies PEFT adapter from adapter_dir
    - merged=True:  loads adapter_dir directly as a full merged model in 4-bit (e.g. SLERP)
    """
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

# Three separate dimension prompts — each asks for exactly one integer.
# This matches the MT-Bench / RAGAS convention of scoring dimensions
# independently so scores are not conflated.

PROMPT_ACCURACY = """Bir Türk tıbbi soru-cevap sisteminin cevabını değerlendiriyorsun.

Hasta sorusu: {question}

Doktor cevabı: {answer}

SADECE ŞU KRİTERE GÖRE puanla (1-10):
Tıbbi doğruluk — Cevap tıbbi açıdan doğru mu? Yanlış veya yanıltıcı bilgi var mı?
  1 = tamamen yanlış veya tehlikeli
  5 = kısmen doğru, eksikler var
  10 = tam ve doğru tıbbi bilgi

Sadece tek bir tam sayı yaz (1-10). Başka hiçbir şey yazma."""

PROMPT_COMPLETENESS = """Bir Türk tıbbi soru-cevap sisteminin cevabını değerlendiriyorsun.

Hasta sorusu: {question}

Doktor cevabı: {answer}

SADECE ŞU KRİTERE GÖRE puanla (1-10):
Tamlık — Cevap hastanın sorusunu tam olarak yanıtlıyor mu? Önemli bir şey eksik mi?
  1 = soruyu hiç yanıtlamıyor
  5 = kısmen yanıtlıyor, önemli eksikler var
  10 = sorunun tüm yönlerini eksiksiz yanıtlıyor

Sadece tek bir tam sayı yaz (1-10). Başka hiçbir şey yazma."""

PROMPT_PRACTICALITY = """Bir Türk tıbbi soru-cevap sisteminin cevabını değerlendiriyorsun.

Hasta sorusu: {question}

Doktor cevabı: {answer}

SADECE ŞU KRİTERE GÖRE puanla (1-10):
Pratiklik — Cevap hastanın hayatına uygulanabilir, somut öneriler içeriyor mu?
  1 = tamamen teorik veya belirsiz, hiçbir pratik öneri yok
  5 = bazı pratik bilgiler var ama yetersiz
  10 = net, uygulanabilir ve hastanın durumuna özel öneriler içeriyor

Sadece tek bir tam sayı yaz (1-10). Başka hiçbir şey yazma."""

DIMENSIONS = [
    ("accuracy",      PROMPT_ACCURACY),
    ("completeness",  PROMPT_COMPLETENESS),
    ("practicality",  PROMPT_PRACTICALITY),
]


def _parse_score(text: str) -> float | None:
    """Extracts the first integer 1-10 from Gemini's response."""
    try:
        score = float(text.strip().split()[0])
        return score if 1.0 <= score <= 10.0 else None
    except Exception:
        return None


def score_with_gemini(
    gemini_model,
    question: str,
    answer: str,
) -> dict | None:
    """
    Scores a medical answer on three dimensions (accuracy, completeness,
    practicality) using separate Gemini calls, then computes an overall mean.

    Returns a dict with keys: accuracy, completeness, practicality, overall.
    Returns None if any dimension call fails.
    Each call is separated by RATE_LIMIT_DELAY to respect the free-tier cap.
    """
    scores = {}
    for dim_name, prompt_template in DIMENSIONS:
        prompt = prompt_template.format(question=question, answer=answer)
        try:
            response = gemini_model.generate_content(prompt)
            score = _parse_score(response.text)
            if score is None:
                print(f"    [Gemini] invalid score for {dim_name}: {response.text!r}")
                return None
            scores[dim_name] = score
        except Exception as e:
            print(f"    [Gemini error] {dim_name}: {e}")
            return None
        time.sleep(RATE_LIMIT_DELAY)

    scores["overall"] = round(np.mean(list(scores.values())), 4)
    return scores


# ──────────────────────────────────────────────
# Statistical test
# ──────────────────────────────────────────────

def paired_ttest_dim(
    qlora_scores: list[float],
    rag_scores: list[float],
    dim: str,
) -> dict:
    """Paired t-test for a single dimension."""
    t_stat, p_value = stats.ttest_rel(rag_scores, qlora_scores)
    n          = len(qlora_scores)
    mean_diff  = np.mean(rag_scores) - np.mean(qlora_scores)
    se         = stats.sem(np.array(rag_scores) - np.array(qlora_scores))
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se)
    return {
        "dimension":   dim,
        "n":           n,
        "mean_qlora":  round(float(np.mean(qlora_scores)), 4),
        "mean_rag":    round(float(np.mean(rag_scores)),   4),
        "mean_diff":   round(float(mean_diff),             4),
        "t_stat":      round(float(t_stat),                4),
        "p_value":     round(float(p_value),               4),
        "ci_95_low":   round(float(ci_low),                4),
        "ci_95_high":  round(float(ci_high),               4),
        "significant": bool(p_value < 0.05),
    }


def run_all_ttests(rows: list[dict]) -> list[dict]:
    """Runs paired t-tests for all four score dimensions."""
    results = []
    for dim in ("accuracy", "completeness", "practicality", "overall"):
        qlora = [r[f"qlora_{dim}"] for r in rows]
        rag   = [r[f"rag_{dim}"]   for r in rows]
        results.append(paired_ttest_dim(qlora, rag, dim))
    return results


def print_stats(ttest_results: list[dict]):
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISON — QLoRA vs RAG (Gemini scoring, paired t-test)")
    print(f"{'='*70}")
    print(f"  {'Dimension':<14} {'QLoRA':>6} {'RAG':>6} {'Diff':>7} {'95% CI':>18} {'p':>7}  Result")
    print(f"  {'-'*65}")
    for r in ttest_results:
        ci = f"[{r['ci_95_low']:+.3f}, {r['ci_95_high']:+.3f}]"
        sig = "p<0.05 ✓" if r["significant"] else "n.s."
        direction = ""
        if r["significant"]:
            direction = " RAG better" if r["mean_diff"] > 0 else " QLoRA better"
        print(
            f"  {r['dimension']:<14} "
            f"{r['mean_qlora']:>6.3f} "
            f"{r['mean_rag']:>6.3f} "
            f"{r['mean_diff']:>+7.3f} "
            f"{ci:>18} "
            f"{r['p_value']:>7.4f}  {sig}{direction}"
        )
    print(f"{'='*70}\n")


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
    model_type = "merged SLERP model" if args.merged else "QLoRA adapter"
    print(f"Loading {model_type}: {args.adapter_dir}")
    model, tokenizer = load_model(args.adapter_dir, args.base_model, merged=args.merged)

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
    # 3 dimension calls per answer × 2 answers per sample
    calls_per_sample = len(DIMENSIONS) * 2
    est_min = total * calls_per_sample * RATE_LIMIT_DELAY / 60
    print(f"\nScoring with Gemini ({GEMINI_MODEL})")
    print(f"  {calls_per_sample} calls/sample × {total} samples = {total * calls_per_sample} calls")
    print(f"  Estimated time: ~{est_min:.0f} min at {RATE_LIMIT_DELAY}s between calls")

    rows = []

    for i in tqdm(range(total), desc="Gemini scoring"):
        q       = all_questions[i]
        qlora_a = all_qlora[i]
        rag_a   = all_rag[i]
        ref     = all_references[i]

        scores_qlora = score_with_gemini(gemini_model, q, qlora_a)
        scores_rag   = score_with_gemini(gemini_model, q, rag_a)

        if scores_qlora is None or scores_rag is None:
            print(f"  Skipping sample {i+1} — Gemini returned invalid score")
            continue

        rows.append({
            "question":             q,
            "reference":            ref,
            "qlora_answer":         qlora_a,
            "rag_answer":           rag_a,
            # per-dimension scores
            "qlora_accuracy":       scores_qlora["accuracy"],
            "qlora_completeness":   scores_qlora["completeness"],
            "qlora_practicality":   scores_qlora["practicality"],
            "qlora_overall":        scores_qlora["overall"],
            "rag_accuracy":         scores_rag["accuracy"],
            "rag_completeness":     scores_rag["completeness"],
            "rag_practicality":     scores_rag["practicality"],
            "rag_overall":          scores_rag["overall"],
            # deltas
            "delta_accuracy":       scores_rag["accuracy"]     - scores_qlora["accuracy"],
            "delta_completeness":   scores_rag["completeness"] - scores_qlora["completeness"],
            "delta_practicality":   scores_rag["practicality"] - scores_qlora["practicality"],
            "delta_overall":        scores_rag["overall"]      - scores_qlora["overall"],
        })

    # ── Statistical tests ─────────────────────
    ttest_results = run_all_ttests(rows)
    print_stats(ttest_results)

    # ── Save results ──────────────────────────
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_csv(args.save_path, index=False)
    print(f"Detailed results saved to {args.save_path}")

    stats_path = args.save_path.replace(".csv", "_stats.csv")
    pd.DataFrame(ttest_results).to_csv(stats_path, index=False)
    print(f"Stats summary saved to {stats_path}")


if __name__ == "__main__":
    main()
