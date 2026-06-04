#!/usr/bin/env python3
"""Run a tiny local DNABERT-2 smoke test on Mac.

This script checks that the CSV files, tokenizer, model loading, Trainer setup,
device selection, and model saving all work before attempting a larger run.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "zhihan1996/DNABERT-2-117M"
OUTPUT_DIR = Path("training/outputs/smoke_test")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
LOCAL_DNABERT2_PATCH_DIR = Path("training/local_dnabert2_patch")

TRAIN_FILENAME = "train_with_sequences.csv"
VAL_FILENAME = "val_with_sequences.csv"

MAX_TRAIN_ROWS = 64
MAX_VAL_ROWS = 32
MIN_SEQUENCE_LENGTH = 50
MAX_LENGTH = 512

DROP_CLNSIG_VALUES = (
    "Conflicting_classifications_of_pathogenicity",
    "Conflicting_interpretations",
    "Uncertain_significance",
    "not_provided",
    "risk_factor",
    "association",
    "drug_response",
    "protective",
)

FLASH_DISABLE_FLAGS = {
    "use_flash_attn": False,
    "use_flash_attention": False,
    "flash_attn": False,
    "attn_implementation": "eager",
    "attention_implementation": "eager",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_dataset_dir(root: Path) -> Path:
    """Prefer data/processed, then fall back to training/csv_files."""
    candidates = [
        root / "data" / "processed",
        root / "training" / "csv_files",
    ]

    for directory in candidates:
        train_path = directory / TRAIN_FILENAME
        val_path = directory / VAL_FILENAME
        if train_path.exists() and val_path.exists():
            return directory

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find train/validation CSV files. Searched:\n"
        f"{searched}\n"
        f"Required files: {TRAIN_FILENAME}, {VAL_FILENAME}"
    )


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def disable_flash_attention_on_config(config):
    """Set common config flags that disable flash/triton attention."""
    for field_name, value in FLASH_DISABLE_FLAGS.items():
        setattr(config, field_name, value)
    config.pad_token_id = getattr(config, "pad_token_id", None) or 3
    config.num_labels = 2
    config.id2label = {0: "benign_or_likely_benign", 1: "pathogenic"}
    config.label2id = {"benign_or_likely_benign": 0, "pathogenic": 1}
    return config


def normalize_text(value: object) -> str:
    return unquote(str(value)).strip().lower()


def should_drop_clnsig(value: object) -> bool:
    text = normalize_text(value)
    return any(drop_value.lower() in text for drop_value in DROP_CLNSIG_VALUES)


def clean_sequence(value: object) -> str:
    return str(value).strip().upper()


def load_and_filter_csv(path: Path, split_name: str, max_rows: int) -> Dataset:
    df = pd.read_csv(path)
    before_rows = len(df)

    required_columns = {"sequence", "label"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    df = df.copy()

    if "CLNSIG" in df.columns:
        df = df.loc[~df["CLNSIG"].fillna("").apply(should_drop_clnsig)].copy()

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.loc[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    df["sequence"] = df["sequence"].fillna("").apply(clean_sequence)
    df = df.loc[df["sequence"] != ""].copy()
    df = df.loc[df["sequence"].str.len() >= MIN_SEQUENCE_LENGTH].copy()

    after_filter_rows = len(df)
    df = df.head(max_rows).copy()

    print(f"{split_name} CSV: {path}")
    print(f"{split_name} rows before filtering: {before_rows:,}")
    print(f"{split_name} rows after filtering: {after_filter_rows:,}")
    print(f"{split_name} rows used for smoke test: {len(df):,}")
    print(f"{split_name} label distribution:")
    print(df["label"].value_counts().sort_index().to_string())
    print()

    if df.empty:
        raise ValueError(f"No usable rows remain for {split_name}.")

    dataset_df = df[["sequence", "label"]].rename(columns={"label": "labels"})
    return Dataset.from_pandas(dataset_df, preserve_index=False)


def tokenize_datasets(tokenizer, train_dataset: Dataset, val_dataset: Dataset) -> tuple[Dataset, Dataset]:
    def tokenize_batch(batch):
        return tokenizer(
            batch["sequence"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
    val_dataset = val_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, (tuple, list)):
        for item in logits:
            candidate = np.asarray(item)
            if candidate.ndim >= 2 and candidate.shape[-1] == 2:
                logits = candidate
                break
        else:
            logits = np.asarray(logits[0])
    else:
        logits = np.asarray(logits)

    labels = np.asarray(labels)
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(labels, predictions),
    }


def make_training_arguments() -> TrainingArguments:
    base_kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "num_train_epochs": 1,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_strategy": "epoch",
        "logging_steps": 5,
        "save_total_limit": 1,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "fp16": False,
        "bf16": False,
    }

    try:
        return TrainingArguments(**base_kwargs, eval_strategy="epoch")
    except TypeError:
        return TrainingArguments(**base_kwargs, evaluation_strategy="epoch")


def make_trainer(model, tokenizer, training_args, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def load_dnabert2_with_eager_attention():
    """Approach A: ask DNABERT-2 remote code to use eager attention."""
    print("Trying DNABERT-2 with eager attention...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = disable_flash_attention_on_config(config)
    print("Triton/flash attention disabled for Mac.")

    model = load_sequence_classification_model(MODEL_NAME, config)
    print("Model loaded successfully.")
    return model


def load_sequence_classification_model(model_source: str, config):
    """Load a classifier, retrying without num_labels for older custom classes."""
    common_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,
    }
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=2,
            **common_kwargs,
        )
    except TypeError as error:
        if "num_labels" not in str(error):
            raise
        return AutoModelForSequenceClassification.from_pretrained(
            model_source,
            **common_kwargs,
        )


def patch_bert_layers_for_mac(source_path: Path, destination_path: Path) -> None:
    """Remove the flash_attn_triton import so Transformers does not require Triton."""
    source_text = source_path.read_text(encoding="utf-8")
    flash_import_block = """try:
    from .flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None
"""
    patched_text = source_text.replace(
        flash_import_block,
        "# Mac-safe local patch: always use the PyTorch attention fallback.\n"
        "flash_attn_qkvpacked_func = None\n",
    )

    if "from .flash_attn_triton import" in patched_text:
        raise RuntimeError("Could not patch flash_attn_triton import from bert_layers.py.")

    # Newer Transformers versions may instantiate custom models under a meta
    # device context. The original ALiBi code can then multiply a CPU tensor by
    # a meta tensor. Keep both tensors on the same device.
    patched_text = patched_text.replace(
        "        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)\n"
        "        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position\n",
        "        slope_device = device if device is not None else relative_position.device\n"
        "        slopes = torch.tensor(_get_alibi_head_slopes(n_heads), device=slope_device)\n"
        "        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position\n",
    )
    patched_text = patched_text.replace(
        "        elif self.alibi.device != hidden_states.device:\n"
        "            # Device catch-up\n"
        "            self.alibi = self.alibi.to(hidden_states.device)\n",
        "        elif getattr(self.alibi, 'is_meta', False) or self.alibi.device != hidden_states.device:\n"
        "            # Device catch-up. Under newer Transformers, the buffer may be created\n"
        "            # on the meta device during low-level loading, so rebuild it instead of\n"
        "            # copying it.\n"
        "            self.rebuild_alibi_tensor(size=self._current_alibi_size, device=hidden_states.device)\n",
    )

    destination_path.write_text(patched_text, encoding="utf-8")


def clear_local_patch_module_cache() -> None:
    """Clear Transformers' cached dynamic module for the local patch."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "local_dnabert2_patch"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def create_local_dnabert2_patch() -> Path:
    """Approach B: create a local DNABERT-2 copy without the Triton import."""
    print("Creating local Mac-safe DNABERT-2 patch...")
    snapshot_path = Path(
        snapshot_download(
            MODEL_NAME,
            allow_patterns=[
                "config.json",
                "configuration_bert.py",
                "bert_layers.py",
                "bert_padding.py",
                "tokenizer.json",
                "tokenizer_config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "generation_config.json",
            ],
        )
    )

    if LOCAL_DNABERT2_PATCH_DIR.exists():
        shutil.rmtree(LOCAL_DNABERT2_PATCH_DIR)
    LOCAL_DNABERT2_PATCH_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "config.json",
        "configuration_bert.py",
        "bert_padding.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    for filename in files_to_copy:
        source = snapshot_path / filename
        if source.exists():
            shutil.copy2(source, LOCAL_DNABERT2_PATCH_DIR / filename)

    patch_bert_layers_for_mac(snapshot_path / "bert_layers.py", LOCAL_DNABERT2_PATCH_DIR / "bert_layers.py")
    clear_local_patch_module_cache()

    config_path = LOCAL_DNABERT2_PATCH_DIR / "config.json"
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    config_json.update(FLASH_DISABLE_FLAGS)
    config_json["pad_token_id"] = config_json.get("pad_token_id") or 3
    config_json["num_labels"] = 2
    config_json["id2label"] = {"0": "benign_or_likely_benign", "1": "pathogenic"}
    config_json["label2id"] = {"benign_or_likely_benign": 0, "pathogenic": 1}
    config_path.write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    print(f"Local Mac-safe DNABERT-2 patch ready: {LOCAL_DNABERT2_PATCH_DIR}")
    return LOCAL_DNABERT2_PATCH_DIR


def load_dnabert2_from_local_patch():
    """Load the local patched copy that avoids flash_attn_triton."""
    patch_dir = create_local_dnabert2_patch()
    config = AutoConfig.from_pretrained(str(patch_dir), trust_remote_code=True)
    config = disable_flash_attention_on_config(config)

    model = load_sequence_classification_model(str(patch_dir), config)
    print("Triton/flash attention disabled for Mac.")
    print("Model loaded successfully.")
    return model


def load_mac_safe_dnabert2_model():
    """Load DNABERT-2 without requiring Triton on Mac."""
    try:
        return load_dnabert2_with_eager_attention()
    except Exception as eager_error:
        print("Approach A failed. Falling back to local Mac-safe DNABERT-2 patch.")
        print(f"Approach A error: {eager_error}")

    try:
        return load_dnabert2_from_local_patch()
    except Exception as patch_error:
        raise RuntimeError(
            "DNABERT-2 could not be loaded without Triton. "
            "The Mac-safe eager and local patch strategies both failed."
        ) from patch_error


def main() -> None:
    root = project_root()
    dataset_dir = find_dataset_dir(root)
    train_csv = dataset_dir / TRAIN_FILENAME
    val_csv = dataset_dir / VAL_FILENAME
    device = choose_device()

    print("DNABERT-2 local smoke test")
    print(f"Selected CSV directory: {dataset_dir}")
    print(f"Selected train CSV path: {train_csv}")
    print(f"Selected validation CSV path: {val_csv}")
    print(f"Selected device: {device}")
    if device == "cpu":
        print("WARNING: CPU training will be slow. This script uses a tiny subset only.")
    print()

    train_dataset = load_and_filter_csv(train_csv, "train", MAX_TRAIN_ROWS)
    val_dataset = load_and_filter_csv(val_csv, "validation", MAX_VAL_ROWS)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Loading model: {MODEL_NAME}")
    model = load_mac_safe_dnabert2_model()

    train_dataset, val_dataset = tokenize_datasets(tokenizer, train_dataset, val_dataset)

    training_args = make_training_arguments()
    trainer = make_trainer(model, tokenizer, training_args, train_dataset, val_dataset)

    print("Starting 1-epoch smoke-test training.")
    trainer.train()

    print("Running validation evaluation.")
    metrics = trainer.evaluate()
    print(metrics)

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print(f"Saved final model to: {FINAL_MODEL_DIR}")
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
