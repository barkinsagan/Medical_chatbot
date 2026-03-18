"""
src/train.py — Training loop using HuggingFace transformers + PEFT + TRL

Works for both LoRA (Phase 1) and QLoRA (Phase 2) based on config.

Paper hyperparams:
  - LoRA rank 8, lr 3e-5, batch 32, 5 epochs
  - Base model: meta-llama/Meta-Llama-3-8B-Instruct
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# ──────────────────────────────────────────────
# Default config (matches paper)
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "max_seq_length": 512,

    # Quantization (set load_in_4bit=True for QLoRA)
    "load_in_4bit": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,

    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # effective batch = 32
    "num_train_epochs": 1,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "seed": 42,

    # Output
    "output_dir": "outputs/checkpoints/lora_baseline",
}


# ──────────────────────────────────────────────
# Load config from YAML
# ──────────────────────────────────────────────

def load_config(yaml_path: str = None) -> dict:
    """
    Loads a YAML config file and merges it with DEFAULT_CONFIG.
    Values in the YAML override the defaults.

    Args:
        yaml_path: path to YAML file (e.g. "configs/lora_baseline.yaml")
                   If None, returns DEFAULT_CONFIG as-is.
    Returns:
        dict of merged config values
    """
    if yaml_path is None:
        return dict(DEFAULT_CONFIG)

    import yaml
    with open(yaml_path, "r") as f:
        overrides = yaml.safe_load(f)

    return {**DEFAULT_CONFIG, **(overrides or {})}


# ──────────────────────────────────────────────
# Load model + tokenizer
# ──────────────────────────────────────────────

def load_model(config: dict = None):
    """
    Loads the base model and tokenizer using HuggingFace transformers + PEFT.
    Applies LoRA adapters. Uses BitsAndBytes for QLoRA if load_in_4bit=True.

    Args:
        config: dict of hyperparams (uses DEFAULT_CONFIG if None)

    Returns:
        (model, tokenizer) — model has LoRA adapters attached
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg["max_seq_length"]

    # Quantization config for QLoRA
    bnb_config = None
    if cfg["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────

def train(model, tokenizer, train_dataset, config: dict = None):
    """
    Runs SFTTrainer on the formatted training dataset.

    Args:
        model        : model with LoRA adapters (from load_model)
        tokenizer    : tokenizer (from load_model)
        train_dataset: HF Dataset with a 'text' column
        config       : dict of hyperparams (uses DEFAULT_CONFIG if None)

    Returns:
        trainer: the SFTTrainer instance (for accessing logs, saving, etc.)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        seed=cfg["seed"],
        gradient_checkpointing=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    trainer.train()

    return trainer


# ──────────────────────────────────────────────
# Save adapter
# ──────────────────────────────────────────────

def save_adapter(model, tokenizer, output_dir: str):
    """Saves the LoRA adapter weights (not the full model)."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")
