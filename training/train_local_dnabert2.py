#!/usr/bin/env python3
"""Fine-tune DNABERT-2 on prepared ClinVar sequence CSV files.

This script works on CUDA, Mac MPS, and CPU. CUDA tries the standard
Hugging Face DNABERT-2 model first; MPS and fallback paths use the local
no-Triton patch from train_smoke_test.py.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_smoke_test import (  # noqa: E402
    LOCAL_DNABERT2_PATCH_DIR,
    MODEL_NAME,
    choose_device,
    clear_local_patch_module_cache,
    create_local_dnabert2_patch,
    disable_flash_attention_on_config,
    load_sequence_classification_model,
    should_drop_clnsig,
)


ORIGINAL_SPLIT_FILES = {
    "train": "train_with_sequences.csv",
    "val": "val_with_sequences.csv",
    "test": "test_with_sequences.csv",
}

ALT_SPLIT_FILES = {
    "train": "train_with_alt_sequences.csv",
    "val": "val_with_alt_sequences.csv",
    "test": "test_with_alt_sequences.csv",
}

MIN_SEQUENCE_LENGTH = 50
RANDOM_STATE = 42


@dataclass
class CsvSelection:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    dataset_dir: Path
    is_alt_dataset: bool
    is_large_alt_dataset: bool
    is_10k_alt_dataset: bool = False
    is_20k_alt_dataset: bool = False
    dataset_name: str = ""


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--val_csv", type=Path, default=None)
    parser.add_argument("--test_csv", type=Path, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("training/outputs/dnabert2_clinvar"))
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--variant_center_index", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_subset_size", type=int, default=0)
    parser.add_argument("--save_eval_each_epoch", type=parse_bool, default=False)
    parser.add_argument("--freeze_encoder", type=parse_bool, default=True)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    parser.add_argument("--freeze_embeddings", type=parse_bool, default=True)
    parser.add_argument("--use_class_weights", type=parse_bool, default=True)
    parser.add_argument("--center_crop", type=parse_bool, default=True)
    parser.add_argument("--tune_threshold", type=parse_bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def has_all_split_files(directory: Path, split_files: dict[str, str]) -> bool:
    return all((directory / filename).exists() for filename in split_files.values())


def infer_alt_dataset_from_paths(paths: list[Path]) -> bool:
    return any("csv_files_alt" in path.parts or "with_alt_sequences" in path.name for path in paths)


def infer_large_alt_dataset_from_paths(paths: list[Path]) -> bool:
    return any(
        directory_name in path.parts
        for path in paths
        for directory_name in ["csv_files_large_alt", "csv_files_10k_alt", "csv_files_20k_alt"]
    )


def infer_10k_alt_dataset_from_paths(paths: list[Path]) -> bool:
    return any("csv_files_10k_alt" in path.parts for path in paths)


def infer_20k_alt_dataset_from_paths(paths: list[Path]) -> bool:
    return any("csv_files_20k_alt" in path.parts for path in paths)


def selection_from_directory(directory: Path, dataset_name: str = "") -> CsvSelection:
    if has_all_split_files(directory, ALT_SPLIT_FILES):
        selected_paths = [
            directory / ALT_SPLIT_FILES["train"],
            directory / ALT_SPLIT_FILES["val"],
            directory / ALT_SPLIT_FILES["test"],
        ]
        return CsvSelection(
            train_csv=selected_paths[0],
            val_csv=selected_paths[1],
            test_csv=selected_paths[2],
            dataset_dir=directory,
            is_alt_dataset=True,
            is_large_alt_dataset=infer_large_alt_dataset_from_paths(selected_paths),
            is_10k_alt_dataset=infer_10k_alt_dataset_from_paths(selected_paths),
            is_20k_alt_dataset=infer_20k_alt_dataset_from_paths(selected_paths),
            dataset_name=dataset_name or directory.name,
        )

    if has_all_split_files(directory, ORIGINAL_SPLIT_FILES):
        selected_paths = [
            directory / ORIGINAL_SPLIT_FILES["train"],
            directory / ORIGINAL_SPLIT_FILES["val"],
            directory / ORIGINAL_SPLIT_FILES["test"],
        ]
        return CsvSelection(
            train_csv=selected_paths[0],
            val_csv=selected_paths[1],
            test_csv=selected_paths[2],
            dataset_dir=directory,
            is_alt_dataset=False,
            is_large_alt_dataset=False,
            dataset_name=dataset_name or directory.name,
        )

    raise FileNotFoundError(
        "Dataset directory does not contain a complete alternate or reference split:\n"
        f"{directory}"
    )


def find_default_csv_selection() -> CsvSelection:
    """Prefer larger alternate-sequence datasets, then smaller fallbacks."""
    candidates = [
        (PROJECT_ROOT / "training" / "csv_files_20k_alt", ALT_SPLIT_FILES, "20k alternate-sequence dataset"),
        (PROJECT_ROOT / "training" / "csv_files_10k_alt", ALT_SPLIT_FILES, "10k alternate-sequence dataset"),
        (PROJECT_ROOT / "training" / "csv_files_large_alt", ALT_SPLIT_FILES, "large alternate-sequence dataset"),
        (PROJECT_ROOT / "training" / "csv_files_alt", ALT_SPLIT_FILES, "alternate-sequence dataset"),
        (PROJECT_ROOT / "data" / "processed", ORIGINAL_SPLIT_FILES, "data/processed reference dataset"),
        (PROJECT_ROOT / "training" / "csv_files", ORIGINAL_SPLIT_FILES, "reference-sequence dataset"),
        (PROJECT_ROOT / "training" / "csv_files_large", ORIGINAL_SPLIT_FILES, "large reference-sequence dataset"),
    ]

    for directory, split_files, dataset_name in candidates:
        if has_all_split_files(directory, split_files):
            return selection_from_directory(directory, dataset_name)

    searched = "\n".join(str(directory) for directory, _split_files, _dataset_name in candidates)
    raise FileNotFoundError(
        "Could not find train/val/test CSV files in the default locations.\n"
        f"Searched:\n{searched}"
    )


def resolve_csv_paths(args: argparse.Namespace) -> CsvSelection:
    default_selection = None
    if args.dataset_dir is not None and not (args.train_csv or args.val_csv or args.test_csv):
        return selection_from_directory(resolve_path(args.dataset_dir), args.dataset_dir.name)

    if not (args.train_csv and args.val_csv and args.test_csv):
        default_selection = find_default_csv_selection()

    train_csv = resolve_path(args.train_csv) if args.train_csv else default_selection.train_csv
    val_csv = resolve_path(args.val_csv) if args.val_csv else default_selection.val_csv
    test_csv = resolve_path(args.test_csv) if args.test_csv else default_selection.test_csv

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    selected_paths = [train_csv, val_csv, test_csv]
    is_alt_dataset = (
        default_selection.is_alt_dataset
        if default_selection is not None
        else infer_alt_dataset_from_paths(selected_paths)
    )

    if infer_alt_dataset_from_paths(selected_paths):
        is_alt_dataset = True

    is_large_alt_dataset = (
        default_selection.is_large_alt_dataset
        if default_selection is not None
        else infer_large_alt_dataset_from_paths(selected_paths)
    )

    if infer_large_alt_dataset_from_paths(selected_paths):
        is_large_alt_dataset = True

    is_10k_alt_dataset = (
        default_selection.is_10k_alt_dataset
        if default_selection is not None
        else infer_10k_alt_dataset_from_paths(selected_paths)
    )
    if infer_10k_alt_dataset_from_paths(selected_paths):
        is_10k_alt_dataset = True

    is_20k_alt_dataset = (
        default_selection.is_20k_alt_dataset
        if default_selection is not None
        else infer_20k_alt_dataset_from_paths(selected_paths)
    )
    if infer_20k_alt_dataset_from_paths(selected_paths):
        is_20k_alt_dataset = True

    common_parent = selected_paths[0].parent
    if any(path.parent != common_parent for path in selected_paths):
        common_parent = PROJECT_ROOT

    return CsvSelection(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        dataset_dir=common_parent,
        is_alt_dataset=is_alt_dataset,
        is_large_alt_dataset=is_large_alt_dataset,
        is_10k_alt_dataset=is_10k_alt_dataset,
        is_20k_alt_dataset=is_20k_alt_dataset,
        dataset_name=default_selection.dataset_name if default_selection is not None else common_parent.name,
    )


def clean_sequence(value: object) -> str:
    return str(value).strip().upper()


def crop_sequence_around_variant(sequence: str, max_length: int, variant_center_index: int) -> str:
    """Crop a long sequence while keeping the variant position in the window."""
    if len(sequence) <= max_length:
        return sequence

    start = max(0, variant_center_index - max_length // 2)
    end = start + max_length
    if end > len(sequence):
        end = len(sequence)
        start = max(0, end - max_length)

    return sequence[start:end]


def apply_center_crop(
    df: pd.DataFrame,
    max_length: int,
    variant_center_index: int,
    split_name: str,
    enabled: bool,
) -> pd.DataFrame:
    if not enabled:
        print(f"{split_name} variant-centered crop: disabled")
        return df.reset_index(drop=True)

    df = df.copy()
    original_lengths = df["sequence"].str.len()
    cropped_count = int((original_lengths > max_length).sum())
    df["sequence"] = df["sequence"].apply(
        lambda sequence: crop_sequence_around_variant(sequence, max_length, variant_center_index)
    )
    cropped_lengths = df["sequence"].str.len()

    print(
        f"{split_name} variant-centered crop: "
        f"{cropped_count:,}/{len(df):,} sequences cropped to {max_length} characters "
        f"around input index {variant_center_index}"
    )
    print(
        f"{split_name} sequence lengths after crop: "
        f"min={int(cropped_lengths.min())}, "
        f"mean={cropped_lengths.mean():.1f}, "
        f"max={int(cropped_lengths.max())}"
    )
    print()
    return df.reset_index(drop=True)


def load_and_filter_dataframe(csv_path: Path, split_name: str, sequence_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    before_rows = len(df)

    required_columns = {sequence_column, "label"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")

    df = df.copy()

    if "CLNSIG" in df.columns:
        df = df.loc[~df["CLNSIG"].fillna("").apply(should_drop_clnsig)].copy()

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.loc[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    df["sequence"] = df[sequence_column].fillna("").apply(clean_sequence)
    df = df.loc[df["sequence"] != ""].copy()
    df = df.loc[df["sequence"].str.len() >= MIN_SEQUENCE_LENGTH].copy()

    print(f"{split_name} CSV: {csv_path}")
    print(f"{split_name} rows before filtering: {before_rows:,}")
    print(f"{split_name} rows after filtering: {len(df):,}")
    print(f"{split_name} label distribution after filtering:")
    print(df["label"].value_counts().sort_index().to_string())
    print()

    if df.empty:
        raise ValueError(f"No usable rows remain in {split_name}.")

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sample_size: int, split_name: str) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True)

    label_counts = df["label"].value_counts()
    if df["label"].nunique() == 2 and label_counts.min() >= 2 and sample_size >= 2:
        try:
            sampled, _unused = train_test_split(
                df,
                train_size=sample_size,
                stratify=df["label"],
                random_state=RANDOM_STATE,
            )
            sampled = sampled.sort_index()
        except ValueError:
            sampled = df.sample(n=sample_size, random_state=RANDOM_STATE).sort_index()
    else:
        sampled = df.sample(n=sample_size, random_state=RANDOM_STATE).sort_index()

    print(f"{split_name} sampled rows: {len(sampled):,}/{len(df):,}")
    print(f"{split_name} label distribution after sampling:")
    print(sampled["label"].value_counts().sort_index().to_string())
    print()
    return sampled.reset_index(drop=True)


def make_eval_subset(df: pd.DataFrame, eval_subset_size: int, split_name: str) -> pd.DataFrame:
    if eval_subset_size <= 0 or len(df) <= eval_subset_size:
        print(f"{split_name} evaluation rows: {len(df):,}")
        print()
        return df.reset_index(drop=True)

    subset = stratified_sample(df, eval_subset_size, f"{split_name} evaluation")
    print(f"{split_name} evaluation subset enabled: {len(subset):,}/{len(df):,}")
    print()
    return subset


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    dataset_df = df[["sequence", "label"]].rename(columns={"label": "labels"})
    return Dataset.from_pandas(dataset_df, preserve_index=False)


def tokenize_dataset(tokenizer, dataset: Dataset, max_length: int, split_name: str) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(
            batch["sequence"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"], desc=f"Tokenizing {split_name}")
    return tokenized


def extract_logits(eval_pred) -> tuple[np.ndarray, np.ndarray]:
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

    return logits, np.asarray(labels)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def extract_predictions_and_labels(eval_pred) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred

    if isinstance(predictions, (tuple, list)):
        for item in predictions:
            candidate = np.asarray(item)
            if candidate.ndim == 1 or (candidate.ndim >= 2 and candidate.shape[-1] in {1, 2}):
                predictions = candidate
                break
        else:
            predictions = np.asarray(predictions[0])
    else:
        predictions = np.asarray(predictions)

    return predictions, np.asarray(labels)


def probabilities_from_predictions(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        return predictions.astype(np.float64)
    if predictions.ndim >= 2 and predictions.shape[-1] == 1:
        return predictions.reshape(-1).astype(np.float64)
    if predictions.ndim >= 2 and predictions.shape[-1] == 2:
        return softmax(predictions.astype(np.float64))[:, 1]
    raise ValueError(f"Unsupported prediction shape for binary classification: {predictions.shape}")


def class_predictions_from_scores(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions)
    if predictions.ndim == 1 or (predictions.ndim >= 2 and predictions.shape[-1] == 1):
        probabilities = probabilities_from_predictions(predictions)
        return (probabilities >= 0.5).astype(int)
    return np.argmax(predictions, axis=-1)


def positive_class_probabilities(logits: np.ndarray) -> np.ndarray:
    return probabilities_from_predictions(logits)


def preprocess_logits_for_metrics(logits, labels):
    """Keep only class-1 probabilities during Trainer evaluation to reduce memory."""
    logits = extract_torch_logits(logits)
    if logits.ndim >= 2 and logits.shape[-1] == 2:
        return torch.softmax(logits.float(), dim=-1)[:, 1]
    if logits.ndim >= 2 and logits.shape[-1] == 1:
        return logits.squeeze(-1)
    return logits


def compute_metrics(eval_pred):
    predictions, labels = extract_predictions_and_labels(eval_pred)
    class_predictions = class_predictions_from_scores(predictions)

    metrics = {
        "accuracy": float(accuracy_score(labels, class_predictions)),
        "precision": float(precision_score(labels, class_predictions, zero_division=0)),
        "recall": float(recall_score(labels, class_predictions, zero_division=0)),
        "f1": float(f1_score(labels, class_predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, class_predictions)),
    }

    if len(np.unique(labels)) == 2:
        probabilities = probabilities_from_predictions(predictions)
        try:
            metrics["auc_roc"] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def tune_threshold_for_mcc(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Try thresholds from 0.10 to 0.90 and keep the best validation MCC."""
    best_threshold = 0.5
    best_mcc = -1.0
    best_f1 = -1.0

    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)

        if mcc > best_mcc or (np.isclose(mcc, best_mcc) and f1 > best_f1):
            best_threshold = float(threshold)
            best_mcc = float(mcc)
            best_f1 = float(f1)

    return best_threshold, best_mcc


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(int)

    metrics = {
        "selected_threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc_roc": None,
        "confusion_matrix": matrix.tolist(),
    }

    if len(np.unique(labels)) == 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def predict_probabilities(trainer: Trainer, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    prediction_output = trainer.predict(dataset)
    logits, labels = extract_logits(prediction_output)
    probabilities = positive_class_probabilities(logits)
    return labels.astype(int), probabilities


def clear_mps_cache_if_needed(device: str) -> None:
    if device != "mps":
        return
    mps_backend = getattr(torch, "mps", None)
    if mps_backend is not None and hasattr(mps_backend, "empty_cache"):
        mps_backend.empty_cache()


def predict_in_small_batches(model, dataset: Dataset, batch_size: int = 1, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Memory-safe prediction loop that immediately moves scores to CPU."""
    device_object = torch.device(device)
    model.to(device_object)
    model.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Memory-safe prediction"):
        labels = batch.pop("labels")
        batch = {key: value.to(device_object) for key, value in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            logits = extract_torch_logits(outputs)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]

        all_probabilities.append(probabilities.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        del outputs, logits, probabilities, labels, batch
        clear_mps_cache_if_needed(device)

    gc.collect()
    clear_mps_cache_if_needed(device)

    return np.concatenate(all_probabilities), np.concatenate(all_labels).astype(int)


def print_confusion_matrix(split_name: str, matrix: list[list[int]]) -> None:
    print(f"{split_name} confusion matrix")
    print("                 predicted_0  predicted_1")
    print(f"actual_0         {matrix[0][0]:>11}  {matrix[0][1]:>11}")
    print(f"actual_1         {matrix[1][0]:>11}  {matrix[1][1]:>11}")


def print_threshold_metrics(split_name: str, metrics: dict) -> None:
    print(f"{split_name} metrics at threshold {metrics['selected_threshold']:.2f}")
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"]:
        value = metrics[key]
        if value is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {value:.4f}")
    print_confusion_matrix(split_name, metrics["confusion_matrix"])
    print()


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Balanced class weights: total_rows / (num_classes * class_count)."""
    counts = train_df["label"].value_counts().sort_index()
    if not {0, 1}.issubset(set(counts.index)):
        print("Class weights requested, but both classes are not present. Using equal weights.")
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    total = float(counts.sum())
    weights = [
        total / (2.0 * float(counts.loc[0])),
        total / (2.0 * float(counts.loc[1])),
    ]
    return torch.tensor(weights, dtype=torch.float32)


def maybe_compute_class_weights(train_df: pd.DataFrame, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        print("Class weights: disabled")
        print()
        return None

    class_weights = compute_class_weights(train_df)
    print("Class weights: enabled")
    print(f"label 0 weight: {class_weights[0].item():.4f}")
    print(f"label 1 weight: {class_weights[1].item():.4f}")
    print()
    return class_weights


def extract_torch_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.ndim >= 2 and item.shape[-1] == 2:
                return item
        return outputs[0]

    raise TypeError("Could not find logits in model outputs.")


class WeightedLossTrainer(Trainer):
    """Trainer that uses weighted cross entropy for imbalanced labels."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = extract_torch_logits(outputs)

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1).long())

        if return_outputs:
            return loss, outputs
        return loss


def local_patch_is_ready(patch_dir: Path) -> bool:
    required_files = [
        "config.json",
        "configuration_bert.py",
        "bert_layers.py",
        "bert_padding.py",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    if not all((patch_dir / filename).exists() for filename in required_files):
        return False

    bert_layers = (patch_dir / "bert_layers.py").read_text(encoding="utf-8")
    return "from .flash_attn_triton import" not in bert_layers and "getattr(self.alibi, 'is_meta', False)" in bert_layers


def create_patch_from_project_root() -> Path:
    """Run the smoke-test patch creator from the project root."""
    previous_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        patch_dir = create_local_dnabert2_patch()
    finally:
        os.chdir(previous_cwd)
    return resolve_path(patch_dir)


def load_model_from_source(model_source: str | Path):
    config = AutoConfig.from_pretrained(str(model_source), trust_remote_code=True)
    config = disable_flash_attention_on_config(config)
    return load_sequence_classification_model(str(model_source), config)


def load_mac_safe_model() -> tuple[object, Path | str]:
    patch_dir = resolve_path(LOCAL_DNABERT2_PATCH_DIR)

    if patch_dir.exists() and local_patch_is_ready(patch_dir):
        print(f"Using local patched DNABERT-2: {patch_dir}")
        clear_local_patch_module_cache()
        model = load_model_from_source(patch_dir)
        print("Triton/flash attention disabled for Mac.")
        print("Model loaded successfully.")
        return model, patch_dir

    print("No usable local patched DNABERT-2 found. Trying DNABERT-2 with eager attention...")
    try:
        model = load_model_from_source(MODEL_NAME)
        print("Triton/flash attention disabled for Mac.")
        print("Model loaded successfully.")
        return model, MODEL_NAME
    except Exception as eager_error:
        print("Direct eager DNABERT-2 load failed. Creating local Mac-safe patch.")
        print(f"Direct load error: {eager_error}")

    patch_dir = create_patch_from_project_root()
    print(f"Using local patched DNABERT-2: {patch_dir}")
    model = load_model_from_source(patch_dir)
    print("Triton/flash attention disabled for Mac.")
    print("Model loaded successfully.")
    return model, patch_dir


def load_dnabert2_model(device: str) -> tuple[object, Path | str]:
    """Load DNABERT-2 using the best strategy for the selected device."""
    if device == "cuda":
        print("CUDA detected. Trying standard Hugging Face DNABERT-2 first.")
        try:
            model = load_model_from_source(MODEL_NAME)
            print("Model loaded successfully from Hugging Face.")
            return model, MODEL_NAME
        except Exception as error:
            print("Standard Hugging Face DNABERT-2 load failed.")
            print(f"Direct load error: {error}")
            print("Falling back to the local no-Triton DNABERT-2 patch.")

    return load_mac_safe_model()


def get_module_by_path(model, path: str):
    current = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_encoder_layers(model) -> tuple[str | None, object | None]:
    layer_paths = [
        "bert.encoder.layer",
        "encoder.layer",
        "base_model.encoder.layer",
        "bert.encoder.layers",
    ]

    for path in layer_paths:
        layers = get_module_by_path(model, path)
        if layers is not None and hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
            return f"model.{path}", layers

    return None, None


def find_embedding_module(model):
    embedding_paths = [
        "bert.embeddings",
        "embeddings",
        "base_model.embeddings",
        "base_model.bert.embeddings",
    ]

    for path in embedding_paths:
        embeddings = get_module_by_path(model, path)
        if embeddings is not None and hasattr(embeddings, "parameters"):
            return f"model.{path}", embeddings

    return None, None


def unfreeze_classifier_parameters(model) -> list[str]:
    trainable_names = []
    for name, parameter in model.named_parameters():
        if name.startswith("classifier") or ".classifier" in name:
            parameter.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def unfreeze_pooler_parameters(model) -> list[str]:
    trainable_names = []
    for name, parameter in model.named_parameters():
        if name.startswith("pooler") or ".pooler" in name:
            parameter.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def print_trainable_parameter_summary(model) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_percentage = (trainable_params / total_params * 100.0) if total_params else 0.0
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_percentage:.4f}%")
    print("First 20 trainable parameter names:")
    for name in trainable_names[:20]:
        print(f"  {name}")
    if len(trainable_names) > 20:
        print(f"  ... {len(trainable_names) - 20:,} more")
    print()


def freeze_encoder_if_requested(
    model,
    freeze_encoder: bool,
    unfreeze_last_n_layers: int,
    freeze_embeddings: bool,
    device: str,
) -> None:
    if unfreeze_last_n_layers < 0:
        raise ValueError("--unfreeze_last_n_layers must be 0 or greater.")

    if not freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad = True
        print("Encoder frozen: false")
        print("All model parameters are trainable.")
        print_trainable_parameter_summary(model)
        return

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier_names = unfreeze_classifier_parameters(model)
    print("Encoder frozen: true")

    if unfreeze_last_n_layers == 0:
        print("Partial unfreezing: disabled")
        print("Trainable modules: classifier head only")
        if not classifier_names:
            print("WARNING: no classifier parameters were found to unfreeze.")
        print_trainable_parameter_summary(model)
        return

    if device == "mps":
        print("WARNING: unfreezing encoder layers on MPS may be slower and may use more memory.")

    print(f"Partial unfreezing: last {unfreeze_last_n_layers} encoder layer(s)")
    pooler_names = unfreeze_pooler_parameters(model)
    if pooler_names:
        print("Pooler unfrozen: true")
    else:
        print("Pooler unfrozen: false, no pooler parameters found")

    layer_path, layers = find_encoder_layers(model)
    if layers is None:
        print("Encoder layer path found: none")
        print("WARNING: no supported encoder layer path was found. Only classifier/pooler parameters are trainable.")
    else:
        layer_count = len(layers)
        layers_to_unfreeze = min(unfreeze_last_n_layers, layer_count)
        start_index = layer_count - layers_to_unfreeze
        print(f"Encoder layer path found: {layer_path}")
        print(f"Encoder layers found: {layer_count}")
        print(f"Encoder layer indexes unfrozen: {start_index} to {layer_count - 1}")

        for layer in list(layers)[start_index:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    if freeze_embeddings:
        print("Embeddings frozen: true")
    else:
        embedding_path, embeddings = find_embedding_module(model)
        if embeddings is None:
            print("Embeddings frozen: false requested, but no embedding module was found")
        else:
            for parameter in embeddings.parameters():
                parameter.requires_grad = True
            print(f"Embeddings frozen: false")
            print(f"Embedding path found: {embedding_path}")

    if not classifier_names:
        print("WARNING: no classifier parameters were found to unfreeze.")

    print_trainable_parameter_summary(model)


def make_training_arguments(args: argparse.Namespace, output_dir: Path, device: str) -> TrainingArguments:
    eval_batch_size = args.batch_size if device == "cuda" else 1
    base_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": args.save_eval_each_epoch,
        "save_total_limit": 2,
        "logging_steps": 20,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": device == "cuda",
        "remove_unused_columns": False,
        "fp16": device == "cuda",
        "bf16": False,
    }

    if args.save_eval_each_epoch:
        base_kwargs["metric_for_best_model"] = "eval_mcc"
        base_kwargs["greater_is_better"] = True
        eval_strategy = "epoch"
    else:
        eval_strategy = "no"

    try:
        return TrainingArguments(**base_kwargs, eval_strategy=eval_strategy)
    except TypeError:
        return TrainingArguments(**base_kwargs, evaluation_strategy=eval_strategy)


def make_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_weights: torch.Tensor | None,
) -> Trainer:
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }

    trainer_class = Trainer
    if class_weights is not None:
        trainer_class = WeightedLossTrainer
        trainer_kwargs["class_weights"] = class_weights

    trainer_signature = inspect.signature(Trainer.__init__)
    if "preprocess_logits_for_metrics" in trainer_signature.parameters:
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return trainer_class(**trainer_kwargs)


def save_metrics(output_dir: Path, metrics: dict) -> Path:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def zip_final_model(output_dir: Path, final_model_dir: Path) -> Path:
    zip_path = output_dir / "final_dnabert2_clinvar_model.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in final_model_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(final_model_dir.parent))

    return zip_path


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir = output_dir / "final_model"

    csv_selection = resolve_csv_paths(args)
    device = choose_device()

    print("DNABERT-2 ClinVar training")
    print(f"Selected dataset directory: {csv_selection.dataset_dir}")
    print(f"Selected dataset name: {csv_selection.dataset_name}")
    print(f"Using 20k alt-sequence dataset: {csv_selection.is_20k_alt_dataset}")
    print(f"Using 10k alt-sequence dataset: {csv_selection.is_10k_alt_dataset}")
    print(f"Using large alt-sequence dataset: {csv_selection.is_large_alt_dataset}")
    print(f"Selected train CSV path: {csv_selection.train_csv}")
    print(f"Selected validation CSV path: {csv_selection.val_csv}")
    print(f"Selected test CSV path: {csv_selection.test_csv}")
    print(f"Selected sequence column: {args.sequence_column}")
    print(f"Using alt-sequence dataset: {csv_selection.is_alt_dataset}")
    print(f"Selected device: {device}")
    if device == "cuda":
        print("CUDA training enabled. fp16 will be enabled in TrainingArguments.")
    if device == "mps":
        print("MPS training enabled. fp16/bf16 stay disabled for Mac compatibility.")
    if device == "cpu":
        print("WARNING: CPU training will be slow. Consider using sample_size > 0.")
    print(f"Output path: {output_dir}")
    print()

    train_df = load_and_filter_dataframe(csv_selection.train_csv, "train", args.sequence_column)
    val_df = load_and_filter_dataframe(csv_selection.val_csv, "validation", args.sequence_column)
    test_df = load_and_filter_dataframe(csv_selection.test_csv, "test", args.sequence_column)

    if args.sample_size > 0:
        train_df = stratified_sample(train_df, args.sample_size, "train")
        eval_sample_size = min(200, args.sample_size)
        val_df = stratified_sample(val_df, min(eval_sample_size, len(val_df)), "validation")
        test_df = stratified_sample(test_df, min(eval_sample_size, len(test_df)), "test")

    train_df = apply_center_crop(
        train_df,
        args.max_length,
        args.variant_center_index,
        "train",
        args.center_crop,
    )
    val_df = apply_center_crop(
        val_df,
        args.max_length,
        args.variant_center_index,
        "validation",
        args.center_crop,
    )
    test_df = apply_center_crop(
        test_df,
        args.max_length,
        args.variant_center_index,
        "test",
        args.center_crop,
    )
    class_weights = maybe_compute_class_weights(train_df, args.use_class_weights)
    eval_val_df = make_eval_subset(val_df, args.eval_subset_size, "validation")
    eval_test_df = make_eval_subset(test_df, args.eval_subset_size, "test")

    print(f"Final train rows: {len(train_df):,}")
    print(f"Final validation rows: {len(val_df):,}")
    print(f"Final test rows: {len(test_df):,}")
    print(f"Validation rows used for evaluation: {len(eval_val_df):,}")
    print(f"Test rows used for evaluation: {len(eval_test_df):,}")
    print(f"Save/evaluate each epoch: {args.save_eval_each_epoch}")
    print(f"Eval accumulation steps: {args.eval_accumulation_steps}")
    print("Using memory-safe evaluation mode.")
    print()

    model, model_path = load_dnabert2_model(device)
    freeze_encoder_if_requested(
        model,
        args.freeze_encoder,
        args.unfreeze_last_n_layers,
        args.freeze_embeddings,
        device,
    )

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    train_dataset = tokenize_dataset(tokenizer, dataframe_to_dataset(train_df), args.max_length, "train")
    val_dataset = tokenize_dataset(tokenizer, dataframe_to_dataset(eval_val_df), args.max_length, "validation")
    test_dataset = tokenize_dataset(tokenizer, dataframe_to_dataset(eval_test_df), args.max_length, "test")

    training_args = make_training_arguments(args, output_dir, device)
    trainer = make_trainer(model, tokenizer, training_args, train_dataset, val_dataset, class_weights)

    print("Starting training.")
    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Predicting validation split for threshold tuning.")
    validation_probabilities, validation_labels = predict_in_small_batches(
        trainer.model,
        val_dataset,
        batch_size=1,
        device=device,
    )
    if args.tune_threshold:
        selected_threshold, best_validation_mcc = tune_threshold_for_mcc(validation_labels, validation_probabilities)
        print(f"Selected threshold from validation MCC: {selected_threshold:.2f}")
        print(f"Best validation MCC during threshold tuning: {best_validation_mcc:.4f}")
    else:
        selected_threshold = 0.5
        best_validation_mcc = None
        print("Threshold tuning disabled. Using threshold: 0.50")
    print()

    validation_metrics = metrics_at_threshold(validation_labels, validation_probabilities, selected_threshold)
    print_threshold_metrics("validation", validation_metrics)

    print("Predicting test split with selected threshold.")
    test_probabilities, test_labels = predict_in_small_batches(
        trainer.model,
        test_dataset,
        batch_size=1,
        device=device,
    )
    test_metrics = metrics_at_threshold(test_labels, test_probabilities, selected_threshold)
    print_threshold_metrics("test", test_metrics)

    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    metrics = {
        "model_name": MODEL_NAME,
        "model_path": str(model_path),
        "train_csv": str(csv_selection.train_csv),
        "validation_csv": str(csv_selection.val_csv),
        "test_csv": str(csv_selection.test_csv),
        "dataset_dir": str(csv_selection.dataset_dir),
        "dataset_name": csv_selection.dataset_name,
        "sequence_column": args.sequence_column,
        "is_alt_sequence_dataset": csv_selection.is_alt_dataset,
        "is_large_alt_sequence_dataset": csv_selection.is_large_alt_dataset,
        "is_10k_alt_sequence_dataset": csv_selection.is_10k_alt_dataset,
        "is_20k_alt_sequence_dataset": csv_selection.is_20k_alt_dataset,
        "device": device,
        "freeze_encoder": args.freeze_encoder,
        "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "freeze_embeddings": args.freeze_embeddings,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "center_crop": args.center_crop,
        "tune_threshold": args.tune_threshold,
        "save_eval_each_epoch": args.save_eval_each_epoch,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "eval_subset_size": args.eval_subset_size,
        "selected_threshold": selected_threshold,
        "best_validation_mcc_for_threshold": best_validation_mcc,
        "sample_size": args.sample_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "variant_center_index": args.variant_center_index,
        "train_metrics": train_output.metrics,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_eval_rows": len(eval_val_df),
        "test_eval_rows": len(eval_test_df),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    metrics_path = save_metrics(output_dir, metrics)
    zip_path = zip_final_model(output_dir, final_model_dir)

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved final model to: {final_model_dir}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Created zip file: {zip_path}")
    print("Local DNABERT-2 training completed successfully.")


if __name__ == "__main__":
    main()
