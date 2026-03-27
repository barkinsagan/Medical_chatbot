"""
run_slerp.py — Slerp merge: blend a LoRA-adapted model back with the base model.

Slerp (Spherical Linear Interpolation) interpolates between base and fine-tuned
weights to recover general language ability lost to catastrophic forgetting.
t=0.0 → pure base model, t=1.0 → pure fine-tuned model, t=0.5 → 50/50 blend.

Paper baseline (Bayram et al., 2025, DOI: 10.1145/3772000):
  t=0.5 → TR-MMLU score: 53/100  (vs 19/100 fine-tuned only)

Usage:
    python run_slerp.py --adapter outputs/checkpoints/lora_baseline --t 0.3 0.5 0.7
    python run_slerp.py --adapter outputs/checkpoints/qlora --t 0.5
    python run_slerp.py --adapter outputs/checkpoints/lora_baseline --t 0.5 --out_dir outputs/merged
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ──────────────────────────────────────────────
# Slerp
# ──────────────────────────────────────────────

def slerp_tensor(w0: torch.Tensor, w1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two weight tensors.
    Falls back to linear interpolation when the angle between them is near zero.

    Args:
        w0: base model weight
        w1: fine-tuned model weight
        t:  interpolation factor (0.0 = base, 1.0 = fine-tuned)
    """
    orig_dtype = w0.dtype
    w0_f = w0.float().flatten()
    w1_f = w1.float().flatten()

    norm0 = w0_f.norm()
    norm1 = w1_f.norm()

    if norm0 < 1e-8 or norm1 < 1e-8:
        return ((1 - t) * w0 + t * w1).to(orig_dtype)

    dot = (w0_f / norm0 * (w1_f / norm1)).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if sin_omega.abs() < 1e-6:
        # Nearly parallel — fall back to lerp
        return ((1 - t) * w0 + t * w1).to(orig_dtype)

    result = (
        torch.sin((1 - t) * omega) / sin_omega * w0_f
        + torch.sin(t * omega) / sin_omega * w1_f
    )
    return result.reshape(w0.shape).to(orig_dtype)


def slerp_state_dicts(base_sd: dict, tuned_sd: dict, t: float) -> dict:
    """
    Applies slerp_tensor to every matching float parameter.
    Non-float buffers (e.g. position ids, layer norm counters) are taken from base.
    """
    merged = {}
    for key in base_sd:
        if key not in tuned_sd:
            merged[key] = base_sd[key]
            continue

        w0 = base_sd[key]
        w1 = tuned_sd[key]

        if w0.dtype in (torch.float32, torch.float16, torch.bfloat16) and w0.shape == w1.shape:
            merged[key] = slerp_tensor(w0, w1, t)
        else:
            merged[key] = w0  # keep base for non-float or mismatched tensors

    return merged


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_base(model_name: str):
    print(f"  Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # keep on CPU for weight manipulation
    )
    return model, tokenizer


# ──────────────────────────────────────────────
# Main merge pipeline
# ──────────────────────────────────────────────

def merge_one(adapter_path: str, base_model_name: str, t: float, out_dir: str):
    adapter_name = os.path.basename(adapter_path.rstrip("/"))
    output_path  = os.path.join(out_dir, f"{adapter_name}_t{t:.2f}")

    print(f"\n{'='*60}")
    print(f"Slerp merge: {adapter_name}  t={t}")
    print(f"Output:      {output_path}")
    print(f"{'='*60}")

    # 1. Load base model (CPU to allow weight surgery)
    base_model, tokenizer = load_base(base_model_name)
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}

    # 2. Load adapter on top and merge LoRA weights into a plain model
    print(f"  Loading adapter + merging LoRA weights...")
    tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    tuned_model = tuned_model.merge_and_unload()  # returns plain HF model
    tuned_sd = tuned_model.state_dict()

    # 3. Apply Slerp between base and merged weights
    print(f"  Applying Slerp (t={t})...")
    merged_sd = slerp_state_dicts(base_sd, tuned_sd, t)

    # 4. Load merged weights back into the model
    tuned_model.load_state_dict(merged_sd)

    # 5. Save as a standard HF model (config.json + safetensors)
    print(f"  Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    tuned_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"  Done: {output_path}")

    del base_model, tuned_model, base_sd, tuned_sd, merged_sd
    torch.cuda.empty_cache()

    return output_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Slerp merge of a LoRA adapter with its base model")
    parser.add_argument("--adapter", required=True,
                        help="Path to the LoRA adapter directory (must contain adapter_config.json)")
    parser.add_argument("--t", nargs="+", type=float, default=[0.5], metavar="T",
                        help="Interpolation value(s) to sweep, e.g. --t 0.3 0.5 0.7")
    parser.add_argument("--base_model", default=None,
                        help="Base model HF path (default: read from adapter_config.json)")
    parser.add_argument("--out_dir", default="outputs/merged",
                        help="Directory to save merged models (default: outputs/merged)")
    args = parser.parse_args()

    # Resolve base model from adapter config if not provided
    base_model = args.base_model
    if base_model is None:
        config_path = os.path.join(args.adapter, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No adapter_config.json found at {args.adapter}")
        with open(config_path) as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model_name_or_path", "meta-llama/Meta-Llama-3-8B-Instruct")
        base_model = base_model.replace("unsloth/llama-3-8b-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")
        print(f"Base model (from adapter config): {base_model}")

    os.makedirs(args.out_dir, exist_ok=True)

    saved = []
    for t in args.t:
        if not 0.0 <= t <= 1.0:
            print(f"  Skipping t={t} — must be between 0.0 and 1.0")
            continue
        path = merge_one(args.adapter, base_model, t, args.out_dir)
        saved.append((t, path))

    print(f"\n{'='*60}")
    print(f"Slerp merge complete — {len(saved)} model(s) saved:")
    for t, path in saved:
        print(f"  t={t:.2f}  →  {path}")
    print(f"\nEvaluate all with:")
    print(f"  python eval_trmmlu.py --slerp_dir {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
