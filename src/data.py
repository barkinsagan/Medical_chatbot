"""
src/data.py — Dataset loading and prompt formatting

Dataset : alibayram/doktorsitesi
Splits  : train (150,105) / test (37,527) — pre-defined on HuggingFace
Fields  : doctor_title, doctor_speciality, question_content, question_answer

Prompt template follows the paper's instruction-tuning format:
  "Sen {specialty} alanında uzman bir Türk {title} olarak,
   aşağıdaki hastanın sorusuna profesyonel ve empatik bir şekilde yanıt ver."

Requires: huggingface_hub.login() called beforehand (dataset is gated).
"""

from datasets import load_dataset, DatasetDict


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

# Training — full example with answer (for SFTTrainer)
TRAIN_TEMPLATE = """Sen {doctor_speciality} alanında uzman bir Türk {doctor_title} olarak, aşağıdaki hastanın sorusuna profesyonel ve empatik bir şekilde yanıt ver.

### Soru:
{question_content}

### Yanıt:
{question_answer}"""

# Inference — answer left open (for generation)
INFERENCE_TEMPLATE = """Sen {doctor_speciality} alanında uzman bir Türk {doctor_title} olarak, aşağıdaki hastanın sorusuna profesyonel ve empatik bir şekilde yanıt ver.

### Soru:
{question_content}

### Yanıt:
"""


# ──────────────────────────────────────────────
# Format functions
# ──────────────────────────────────────────────

def format_train(example: dict) -> dict:
    """Formats a row for training (includes answer). Returns 'text' field."""
    text = TRAIN_TEMPLATE.format(
        doctor_speciality=example["doctor_speciality"],
        doctor_title=example["doctor_title"],
        question_content=example["question_content"],
        question_answer=example["question_answer"],
    )
    return {"text": text}


def format_inference(example: dict) -> dict:
    """Formats a row for inference (answer left blank). Returns 'prompt' + 'reference'."""
    prompt = INFERENCE_TEMPLATE.format(
        doctor_speciality=example["doctor_speciality"],
        doctor_title=example["doctor_title"],
        question_content=example["question_content"],
    )
    return {"prompt": prompt, "reference": example["question_answer"]}


# ──────────────────────────────────────────────
# Load and prepare the dataset
# ──────────────────────────────────────────────

def load_doktorsitesi() -> DatasetDict:
    """
    Loads alibayram/doktorsitesi from HuggingFace using the
    pre-defined train/test splits.

    Returns:
        DatasetDict with 'train' and 'test' splits.
        - train: 150,105 rows with 'text' column (for SFTTrainer)
        - test:  37,527 rows with 'prompt' + 'reference' columns (for eval)
    """
    ds = load_dataset("alibayram/doktorsitesi")

    train = ds["train"].map(format_train)
    test = ds["test"].map(format_inference)

    return DatasetDict({"train": train, "test": test})
