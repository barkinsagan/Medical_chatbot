"""
run_train.py — CLI entry point for LoRA / QLoRA training

Usage:
    python run_train.py                                      # defaults (qlora.yaml)
    python run_train.py --config configs/lora_baseline.yaml
    python run_train.py --epochs 3 --lr 2e-5 --batch 4
    python run_train.py --config configs/qlora.yaml --epochs 5 --output outputs/my_run

HF token is read from the HF_TOKEN environment variable.
"""

import argparse
import os

from huggingface_hub import login
from src.data import load_doktorsitesi
from src.train import load_config, load_model, train, save_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA / QLoRA adapter on doktorsitesi")

    # Config file (all defaults come from here)
    parser.add_argument("--config", type=str, default="configs/qlora.yaml",
                        help="Path to YAML config file (default: configs/qlora.yaml)")

    # Model
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--max-seq-len", type=int, help="Max sequence length")

    # Quantization
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (use full LoRA instead of QLoRA)")

    # LoRA
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, help="LoRA dropout")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch", type=int, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, help="Warmup ratio")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Output
    parser.add_argument("--output", type=str, help="Output directory for the adapter")

    return parser.parse_args()


def apply_overrides(config: dict, args) -> dict:
    """Applies CLI arguments on top of the loaded YAML config."""
    if args.model:          config["model_name"] = args.model
    if args.max_seq_len:    config["max_seq_length"] = args.max_seq_len
    if args.no_4bit:        config["load_in_4bit"] = False
    if args.lora_r:         config["lora_r"] = args.lora_r
    if args.lora_alpha:     config["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None: config["lora_dropout"] = args.lora_dropout
    if args.epochs:         config["num_train_epochs"] = args.epochs
    if args.lr:             config["learning_rate"] = args.lr
    if args.batch:          config["per_device_train_batch_size"] = args.batch
    if args.grad_accum:     config["gradient_accumulation_steps"] = args.grad_accum
    if args.warmup_ratio is not None: config["warmup_ratio"] = args.warmup_ratio
    if args.seed:           config["seed"] = args.seed
    if args.output:         config["output_dir"] = args.output
    return config


def main():
    args = parse_args()

    # Auth
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    login(token=token)

    # Config
    config = load_config(args.config)
    config = apply_overrides(config, args)

    eff_batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    print(f"\n{'='*60}")
    print(f"  Model:           {config['model_name']}")
    print(f"  Config:          {args.config}")
    print(f"  4-bit QLoRA:     {config['load_in_4bit']}")
    print(f"  LoRA rank:       {config['lora_r']}")
    print(f"  Epochs:          {config['num_train_epochs']}")
    print(f"  Learning rate:   {config['learning_rate']}")
    print(f"  Effective batch: {eff_batch} ({config['per_device_train_batch_size']} x {config['gradient_accumulation_steps']})")
    print(f"  Output:          {config['output_dir']}")
    print(f"{'='*60}\n")

    # Data
    print("Loading dataset...")
    ds = load_doktorsitesi()

    # Model
    print("Loading model...")
    model, tokenizer = load_model(config)

    # Train
    print("Starting training...")
    trainer = train(model, tokenizer, ds["train"], config)

    # Save
    save_adapter(model, tokenizer, config["output_dir"])


if __name__ == "__main__":
    main()
