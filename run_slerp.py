"""
run_slerp.py — Slerp merge: blend a LoRA-adapted model back with the base model.

Slerp (Spherical Linear Interpolation) interpolates between base and fine-tuned
weights to recover general language ability lost to catastrophic forgetting.
t=0.0 → pure base model, t=1.0 → pure fine-tuned model, t=0.5 → 50/50 blend.

Paper baseline (Bayram et al., 2025, DOI: 10.1145/3772000):
  t=0.5 → TR-MMLU score: 53/100  (vs 19/100 fine-tuned only)

Memory-efficient design: base weights are read from the HF cache via memory-mapped
safetensors — only the tuned model (~16GB) is held in RAM at once.

Usage:
    python run_slerp.py --adapter outputs/checkpoints/lora_baseline --t 0.3 0.5 0.7
    python run_slerp.py --adapter outputs/checkpoints/qlora --t 0.5
    python run_slerp.py --adapter outputs/checkpoints/lora_baseline --t 0.5 --out_dir outputs/merged
"""

import argparse
import gc
import json
import os
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
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
        return ((1 - t) * w0 + t * w1).to(orig_dtype)

    result = (
        torch.sin((1 - t) * omega) / sin_omega * w0_f
        + torch.sin(t * omega) / sin_omega * w1_f
    )
    return result.reshape(w0.shape).to(orig_dtype)


# ──────────────────────────────────────────────
# Memory-mapped base weight reader
# ──────────────────────────────────────────────

class BaseWeightReader:
    """
    Reads base model weights on demand from memory-mapped safetensors files
    in the HuggingFace cache. Uses near-zero RAM — tensors are loaded one at a time.
    """

    def __init__(self, model_name: str):
        print(f"  Locating base model in HF cache: {model_name}")
        cache_dir = snapshot_download(model_name)

        index_path = os.path.join(cache_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self._weight_map = json.load(f)["weight_map"]
            self._handles = {
                fname: safe_open(os.path.join(cache_dir, fname), framework="pt", device="cpu")
                for fname in set(self._weight_map.values())
            }
            self._sharded = True
        else:
            self._handle = safe_open(
                os.path.join(cache_dir, "model.safetensors"), framework="pt", device="cpu"
            )
            self._sharded = False

    def get(self, name: str) -> torch.Tensor | None:
        if self._sharded:
            if name not in self._weight_map:
                return None
            return self._handles[self._weight_map[name]].get_tensor(name)
        else:
            if name not in self._handle.keys():
                return None
            return self._handle.get_tensor(name)


# ──────────────────────────────────────────────
# Main merge pipeline
# ──────────────────────────────────────────────

def merge_one(
    adapter_path: str,
    base_model_name: str,
    t: float,
    out_dir: str,
    base_reader: BaseWeightReader,
    tokenizer: AutoTokenizer,
):
    adapter_name = os.path.basename(adapter_path.rstrip("/"))
    output_path  = os.path.join(out_dir, f"{adapter_name}_t{t:.2f}")

    print(f"\n{'='*60}")
    print(f"Slerp merge: {adapter_name}  t={t}")
    print(f"Output:      {output_path}")
    print(f"{'='*60}")

    # 1. Load base model + adapter, merge LoRA weights in-place
    print(f"  Loading adapter + merging LoRA weights...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )
    tuned = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    # 2. Apply Slerp in-place — read base weights one tensor at a time from cache
    print(f"  Applying Slerp (t={t})...")
    float_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    with torch.no_grad():
        for name, param in tuned.named_parameters():
            base_w = base_reader.get(name)
            if base_w is not None and param.dtype in float_dtypes:
                param.data.copy_(
                    slerp_tensor(base_w.to(param.dtype), param.data, t)
                )

    # 3. Save as standard HF model (config.json + safetensors)
    print(f"  Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    tuned.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"  Done: {output_path}")

    del tuned, base
    gc.collect()
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
    base_model_name = args.base_model
    if base_model_name is None:
        config_path = os.path.join(args.adapter, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No adapter_config.json found at {args.adapter}")
        with open(config_path) as f:
            cfg = json.load(f)
        base_model_name = cfg.get("base_model_name_or_path", "meta-llama/Meta-Llama-3-8B-Instruct")
        base_model_name = base_model_name.replace(
            "unsloth/llama-3-8b-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        print(f"Base model (from adapter config): {base_model_name}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer and base weight reader once — reused across all t values
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_reader = BaseWeightReader(base_model_name)

    saved = []
    for t in args.t:
        if not 0.0 <= t <= 1.0:
            print(f"  Skipping t={t} — must be between 0.0 and 1.0")
            continue
        path = merge_one(args.adapter, base_model_name, t, args.out_dir, base_reader, tokenizer)
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
