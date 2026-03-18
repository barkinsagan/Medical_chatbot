
"""
src/evaluate.py — Evaluation pipeline

Metrics (from paper):
  1. Cosine similarity via SBERT (paraphrase-multilingual-mpnet-base-v2)
     - Paper baseline: 0.4627 (base LLaMA 3)
     - Paper fine-tuned: 0.5151 (+11.33%)
  2. GPT-based scoring (1-10 scale, optional)
     - Paper expert answers: 7.692
     - Paper fine-tuned: 7.132

Usage in notebook:
    from src.evaluate import generate_responses, compute_similarity, evaluate
    results = evaluate(model, tokenizer, ds["test"], n_samples=100)
"""

import torch
import time
import pandas as pd
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. Generate responses on test set
# ──────────────────────────────────────────────

def generate_responses(
    model,
    tokenizer,
    test_dataset,
    n_samples: int = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """
    Generates model responses for the test set prompts.

    Each test row has a 'prompt' (the question formatted for inference)
    and a 'reference' (the real doctor's answer). This function feeds
    the prompt to the model and collects the generated text.

    Args:
        model          : the fine-tuned model (or base model for comparison)
        tokenizer      : the model's tokenizer
        test_dataset   : HF Dataset with 'prompt' and 'reference' columns
        n_samples      : how many test samples to evaluate (None = all)
        max_new_tokens : max tokens to generate per response
        temperature    : sampling temperature (0.7 matches typical medical LLM evals)

    Returns:
        DataFrame with columns: prompt, reference, generated, n_tokens, time_sec
    """
    model.eval()

    # Subset if requested
    if n_samples is not None:
        test_dataset = test_dataset.select(range(min(n_samples, len(test_dataset))))

    results = []
    for i in tqdm(range(len(test_dataset)), desc="Generating"):
        row = test_dataset[i]
        prompt = row["prompt"]
        reference = row["reference"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t0

        # Decode only the newly generated tokens (not the prompt)
        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
            "n_tokens": len(gen_ids),
            "time_sec": round(elapsed, 2),
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 2. Compute cosine similarity via SBERT
# ──────────────────────────────────────────────

def compute_similarity(
    df: pd.DataFrame,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Computes cosine similarity between generated and reference answers
    using Sentence-BERT, matching the paper's evaluation method.

    The SBERT model encodes both texts into dense vectors, then cosine
    similarity measures how close they are in meaning (not exact wording).
    A score of 1.0 = identical meaning, 0.0 = completely unrelated.

    Args:
        df         : DataFrame with 'reference' and 'generated' columns
        model_name : SBERT model (paper uses paraphrase-multilingual-mpnet-base-v2)
        batch_size : encoding batch size (doesn't affect results, just speed)

    Returns:
        Same DataFrame with a new 'similarity' column added
    """
    from sentence_transformers import SentenceTransformer, util

    sbert = SentenceTransformer(model_name)

    # Encode all references and generations in batches
    print("Encoding references...")
    ref_embeddings = sbert.encode(
        df["reference"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    print("Encoding generated responses...")
    gen_embeddings = sbert.encode(
        df["generated"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    # Cosine similarity for each pair (not all-vs-all, just row-by-row)
    similarities = util.cos_sim(ref_embeddings, gen_embeddings).diagonal().cpu().numpy()
    df = df.copy()
    df["similarity"] = similarities

    return df


# ──────────────────────────────────────────────
# 3. Summary statistics
# ──────────────────────────────────────────────

def print_summary(df: pd.DataFrame, label: str = "Model"):
    """
    Prints evaluation summary matching the paper's reporting format.

    Paper benchmarks for reference:
      - Base LLaMA 3 similarity:  0.4627
      - Fine-tuned similarity:    0.5151
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — {label}")
    print(f"{'='*60}")
    print(f"   Samples evaluated:    {len(df):,}")
    print(f"   Avg similarity:       {df['similarity'].mean():.4f}")
    print(f"   Median similarity:    {df['similarity'].median():.4f}")
    print(f"   Std similarity:       {df['similarity'].std():.4f}")
    print(f"   Min / Max:            {df['similarity'].min():.4f} / {df['similarity'].max():.4f}")
    print(f"   Avg tokens generated: {df['n_tokens'].mean():.1f}")
    print(f"   Avg generation time:  {df['time_sec'].mean():.2f}s")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 4. Save results
# ──────────────────────────────────────────────

def save_results(df: pd.DataFrame, path: str):
    """Saves evaluation results to CSV."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")


# ──────────────────────────────────────────────
# 5. Full evaluation pipeline
# ──────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    test_dataset,
    n_samples: int = None,
    label: str = "Model",
    save_path: str = None,
) -> pd.DataFrame:
    """
    Runs the full evaluation pipeline: generate → similarity → summary.

    Args:
        model        : the model to evaluate
        tokenizer    : the model's tokenizer
        test_dataset : HF Dataset with 'prompt' and 'reference' columns
        n_samples    : how many samples (None = all 37,527)
        label        : name for this eval run (used in summary printout)
        save_path    : if provided, saves results CSV here

    Returns:
        DataFrame with columns: prompt, reference, generated, n_tokens, time_sec, similarity
    """
    # Step 1: Generate responses
    print(f"\n--- Generating responses ({n_samples or len(test_dataset)} samples) ---")
    df = generate_responses(model, tokenizer, test_dataset, n_samples=n_samples)

    # Step 2: Compute SBERT similarity
    print(f"\n--- Computing SBERT cosine similarity ---")
    df = compute_similarity(df)

    # Step 3: Print summary
    print_summary(df, label=label)

    # Step 4: Save if path provided
    if save_path:
        save_results(df, save_path)

    return df


# ──────────────────────────────────────────────
# Pipeline validation
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test: generates and evaluates 5 samples to verify
    the entire evaluation pipeline works end-to-end.
    """
    import os
    from src.data import load_doktorsitesi
    from src.train import load_model

    # ── 1. Load data and model ────────────────
    print("=" * 60)
    print("EVALUATE.PY — PIPELINE TEST")
    print("=" * 60)

    ds = load_doktorsitesi()
    model, tokenizer = load_model()

    # ── 2. Run eval on small subset ───────────
    df = evaluate(
        model, tokenizer, ds["test"],
        n_samples=5,
        label="Base Model (5 samples)",
        save_path="outputs/eval_results/test_eval.csv",
    )

    # ── 3. Inspect results ────────────────────
    print("Sample results:")
    for i, row in df.iterrows():
        print(f"\n--- Sample {i+1} (similarity: {row['similarity']:.4f}) ---")
        print(f"   Generated: {row['generated'][:150]}")
        print(f"   Reference: {row['reference'][:150]}")

    # ── 4. Verify CSV was saved ───────────────
    assert os.path.exists("outputs/eval_results/test_eval.csv"), "CSV not saved!"
    saved = pd.read_csv("outputs/eval_results/test_eval.csv")
    assert len(saved) == 5
    assert "similarity" in saved.columns
    print(f"\n✅ EVALUATE PIPELINE TEST PASSED")
    print(f"   CSV saved with {len(saved)} rows and similarity scores")
