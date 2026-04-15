"""
scripts/score_answers.py — Score pre-generated answers with a strong LLM + paired t-test

Runs on Colab (or any machine with a strong GPU). Takes the CSV produced by
generate_answers.py and scores each answer pair on three medical quality
dimensions using a configurable scorer model.

Recommended scorer models (in order of quality):
    meta-llama/Meta-Llama-3-70B-Instruct   (best, needs A100 80GB)
    meta-llama/Meta-Llama-3-8B-Instruct    (fast, weaker judgment)
    --gemini                               (Gemini Flash API, free tier)

Output:
    <save_path>         — per-sample scores CSV
    <save_path>_stats   — paired t-test results per dimension

Usage (Colab):
    # Score with LLaMA 3 70B
    python scripts/score_answers.py \
        --answers_csv  outputs/eval_results/answers.csv \
        --scorer_model meta-llama/Meta-Llama-3-70B-Instruct \
        --save_path    outputs/eval_results/scores_70b.csv

    # Score with Gemini API
    python scripts/score_answers.py \
        --answers_csv  outputs/eval_results/answers.csv \
        --gemini \
        --save_path    outputs/eval_results/scores_gemini.csv

Requires:
    HF_TOKEN env var (for gated models)
    GEMINI_API_KEY env var (only if using --gemini)
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


GEMINI_MODEL     = "gemini-2.0-flash"
RATE_LIMIT_DELAY = 4.5


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Score pre-generated answers with a strong LLM")
    parser.add_argument("--answers_csv",   required=True,
                        help="CSV produced by generate_answers.py")
    parser.add_argument("--scorer_model",  default="meta-llama/Meta-Llama-3-70B-Instruct",
                        help="HF model ID or local path to use as scorer")
    parser.add_argument("--gemini",        action="store_true",
                        help="Use Gemini API instead of a local scorer model")
    parser.add_argument("--load_in_4bit",  action="store_true", default=True,
                        help="Load scorer in 4-bit (recommended for 70B on A100 40GB)")
    parser.add_argument("--save_path",     default="outputs/eval_results/scores.csv")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Scoring prompts (same as generate_answers.py)
# ──────────────────────────────────────────────

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
    ("accuracy",     PROMPT_ACCURACY),
    ("completeness", PROMPT_COMPLETENESS),
    ("practicality", PROMPT_PRACTICALITY),
]


# ──────────────────────────────────────────────
# Score parsing
# ──────────────────────────────────────────────

def _parse_score(text: str) -> float | None:
    try:
        score = float(text.strip().split()[0])
        return score if 1.0 <= score <= 10.0 else None
    except Exception:
        return None


# ──────────────────────────────────────────────
# Scorer loading
# ──────────────────────────────────────────────

def load_scorer(model_id: str, load_in_4bit: bool = True):
    """Loads a HuggingFace model as the scorer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────
# Scoring functions
# ──────────────────────────────────────────────

def score_with_local(model, tokenizer, question: str, answer: str) -> dict | None:
    """Scores an answer on all three dimensions using a local model."""
    scores = {}
    for dim_name, prompt_template in DIMENSIONS:
        prompt  = prompt_template.format(question=question, answer=answer)
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text    = tokenizer.decode(gen_ids, skip_special_tokens=True)
        score   = _parse_score(text)
        if score is None:
            print(f"    [scorer] invalid response for {dim_name}: {text!r}")
            return None
        scores[dim_name] = score

    scores["overall"] = round(np.mean(list(scores.values())), 4)
    return scores


def score_with_gemini(gemini_model, question: str, answer: str) -> dict | None:
    """Scores an answer on all three dimensions using Gemini API."""
    import google.generativeai as genai  # only imported when needed
    scores = {}
    for dim_name, prompt_template in DIMENSIONS:
        prompt = prompt_template.format(question=question, answer=answer)
        try:
            response = gemini_model.generate_content(prompt)
            score    = _parse_score(response.text)
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
# Statistical tests
# ──────────────────────────────────────────────

def paired_ttest_dim(qlora_scores: list[float], rag_scores: list[float], dim: str) -> dict:
    t_stat, p_value = stats.ttest_rel(rag_scores, qlora_scores)
    n         = len(qlora_scores)
    mean_diff = float(np.mean(rag_scores) - np.mean(qlora_scores))
    se        = stats.sem(np.array(rag_scores) - np.array(qlora_scores))
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se)
    return {
        "dimension":   dim,
        "n":           n,
        "mean_qlora":  round(float(np.mean(qlora_scores)), 4),
        "mean_rag":    round(float(np.mean(rag_scores)),   4),
        "mean_diff":   round(mean_diff, 4),
        "t_stat":      round(float(t_stat),   4),
        "p_value":     round(float(p_value),  4),
        "ci_95_low":   round(float(ci_low),   4),
        "ci_95_high":  round(float(ci_high),  4),
        "significant": bool(p_value < 0.05),
    }


def run_all_ttests(rows: list[dict]) -> list[dict]:
    return [
        paired_ttest_dim(
            [r[f"qlora_{dim}"] for r in rows],
            [r[f"rag_{dim}"]   for r in rows],
            dim,
        )
        for dim in ("accuracy", "completeness", "practicality", "overall")
    ]


def print_stats(ttest_results: list[dict], scorer_label: str):
    print(f"\n{'='*70}")
    print(f"STATISTICAL COMPARISON — QLoRA vs RAG  (scorer: {scorer_label})")
    print(f"{'='*70}")
    print(f"  {'Dimension':<14} {'QLoRA':>6} {'RAG':>6} {'Diff':>7} {'95% CI':>18} {'p':>7}  Result")
    print(f"  {'-'*65}")
    for r in ttest_results:
        ci  = f"[{r['ci_95_low']:+.3f}, {r['ci_95_high']:+.3f}]"
        sig = "p<0.05 ✓" if r["significant"] else "n.s."
        direction = (" RAG better" if r["mean_diff"] > 0 else " QLoRA better") if r["significant"] else ""
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

    print("=" * 60)
    print("SCORE ANSWERS")
    print("=" * 60)
    print(f"  Answers CSV : {args.answers_csv}")

    # ── Load scorer ───────────────────────────
    gemini_model = None
    if args.gemini:
        import google.generativeai as genai
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=gemini_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        scorer_label = f"Gemini {GEMINI_MODEL}"
        scorer_model, scorer_tokenizer = None, None
    else:
        scorer_label = args.scorer_model
        print(f"  Scorer      : {scorer_label}")
        print(f"  4-bit       : {args.load_in_4bit}")
        print()
        print("Loading scorer model...")
        scorer_model, scorer_tokenizer = load_scorer(args.scorer_model, args.load_in_4bit)

    print(f"  Scorer      : {scorer_label}")
    print()

    # ── Load answers ──────────────────────────
    df_in = pd.read_csv(args.answers_csv)
    total = len(df_in)
    print(f"Loaded {total} answer pairs from {args.answers_csv}")

    # ── Score each pair ───────────────────────
    rows = []
    for i, row in tqdm(df_in.iterrows(), total=total, desc="Scoring"):
        q       = row["question"]
        ref     = row["reference"]
        qlora_a = row["qlora_answer"]
        rag_a   = row["rag_answer"]

        if args.gemini:
            scores_qlora = score_with_gemini(gemini_model, q, qlora_a)
            scores_rag   = score_with_gemini(gemini_model, q, rag_a)
        else:
            scores_qlora = score_with_local(scorer_model, scorer_tokenizer, q, qlora_a)
            scores_rag   = score_with_local(scorer_model, scorer_tokenizer, q, rag_a)

        if scores_qlora is None or scores_rag is None:
            print(f"  Skipping sample {i+1} — invalid score returned")
            continue

        rows.append({
            "question":           q,
            "reference":          ref,
            "qlora_answer":       qlora_a,
            "rag_answer":         rag_a,
            "qlora_accuracy":     scores_qlora["accuracy"],
            "qlora_completeness": scores_qlora["completeness"],
            "qlora_practicality": scores_qlora["practicality"],
            "qlora_overall":      scores_qlora["overall"],
            "rag_accuracy":       scores_rag["accuracy"],
            "rag_completeness":   scores_rag["completeness"],
            "rag_practicality":   scores_rag["practicality"],
            "rag_overall":        scores_rag["overall"],
            "delta_accuracy":     scores_rag["accuracy"]     - scores_qlora["accuracy"],
            "delta_completeness": scores_rag["completeness"] - scores_qlora["completeness"],
            "delta_practicality": scores_rag["practicality"] - scores_qlora["practicality"],
            "delta_overall":      scores_rag["overall"]      - scores_qlora["overall"],
        })

    # ── Statistical tests ─────────────────────
    ttest_results = run_all_ttests(rows)
    print_stats(ttest_results, scorer_label)

    # ── Save ──────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.save_path, index=False)
    print(f"Scores saved to {args.save_path}")

    stats_path = args.save_path.replace(".csv", "_stats.csv")
    pd.DataFrame(ttest_results).to_csv(stats_path, index=False)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
