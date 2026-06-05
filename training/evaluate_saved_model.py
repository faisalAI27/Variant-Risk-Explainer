#!/usr/bin/env python3
"""Memory-safe full evaluation for a saved DNABERT-2 ClinVar model.

This script does not train. It loads the saved model, evaluates the full
validation and test CSV files in small batches, and writes metrics to JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors_file
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_smoke_test import (  # noqa: E402
    LOCAL_DNABERT2_PATCH_DIR,
    clear_local_patch_module_cache,
    create_local_dnabert2_patch,
    disable_flash_attention_on_config,
    load_sequence_classification_model,
)

OUTPUT_DIR = PROJECT_ROOT / "training" / "outputs" / "dnabert2_clinvar"
MODEL_DIR = OUTPUT_DIR / "final_model"
UPLOADED_MODEL_DIR = PROJECT_ROOT / "training" / "training_model_files"
TRAINING_METRICS_PATH = OUTPUT_DIR / "metrics.json"
FULL_EVAL_METRICS_PATH = OUTPUT_DIR / "full_eval_metrics.json"

ALT_SPLIT_FILES = {
    "validation": "val_with_alt_sequences.csv",
    "test": "test_with_alt_sequences.csv",
}

DATASET_CANDIDATES = [
    PROJECT_ROOT / "training" / "csv_files_20k_alt",
    PROJECT_ROOT / "training" / "csv_files_10k_alt",
    PROJECT_ROOT / "training" / "csv_files_large_alt",
    PROJECT_ROOT / "training" / "csv_files_alt",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "training" / "csv_files",
]

SEQUENCE_COLUMN = "sequence"
LABEL_COLUMN = "label"
MAX_LENGTH = 512
VARIANT_CENTER_INDEX = 512
BATCH_SIZE = 1
MPS_CACHE_EVERY = 25


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Use true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-safe full validation/test evaluation for the saved DNABERT-2 ClinVar model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tune_threshold",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Tune the decision threshold on the full validation set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use this fixed decision threshold instead of tuning.",
    )
    parser.add_argument(
        "--threshold_min",
        type=float,
        default=0.1,
        help="Minimum threshold to test when tuning.",
    )
    parser.add_argument(
        "--threshold_max",
        type=float,
        default=0.9,
        help="Maximum threshold to test when tuning.",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
        help="Threshold step size when tuning.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help=(
            "Saved model folder to evaluate. Defaults to training/outputs/dnabert2_clinvar/final_model; "
            "if that is missing, falls back to training/training_model_files."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be greater than 0.")

    if args.threshold_min > args.threshold_max:
        raise ValueError("--threshold_min must be less than or equal to --threshold_max.")


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def choose_model_dir(requested_model_dir: Path | None) -> Path:
    if requested_model_dir is not None:
        model_dir = resolve_project_path(requested_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Requested saved model directory not found: {model_dir}")
        return model_dir

    if MODEL_DIR.exists():
        return MODEL_DIR

    if UPLOADED_MODEL_DIR.exists():
        return UPLOADED_MODEL_DIR

    raise FileNotFoundError(
        "Saved model directory not found. Searched:\n"
        f"{MODEL_DIR}\n"
        f"{UPLOADED_MODEL_DIR}"
    )


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def find_dataset_dir() -> Path:
    for directory in DATASET_CANDIDATES:
        if all((directory / filename).exists() for filename in ALT_SPLIT_FILES.values()):
            return directory

    searched = "\n".join(str(directory) for directory in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not find validation/test alternate-sequence CSV files.\n"
        f"Searched:\n{searched}"
    )


def load_threshold() -> float:
    if not TRAINING_METRICS_PATH.exists():
        return 0.5

    try:
        metrics = json.loads(TRAINING_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.5

    threshold = metrics.get("selected_threshold")
    if threshold is None:
        return 0.5

    try:
        return float(threshold)
    except (TypeError, ValueError):
        return 0.5


def clean_sequence(value: object) -> str:
    return str(value).strip().upper()


def crop_sequence_around_variant(sequence: str, max_length: int, variant_center_index: int) -> str:
    if len(sequence) <= max_length:
        return sequence

    start = max(0, variant_center_index - max_length // 2)
    end = start + max_length
    if end > len(sequence):
        end = len(sequence)
        start = max(0, end - max_length)

    return sequence[start:end]


def load_eval_dataframe(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {SEQUENCE_COLUMN, LABEL_COLUMN}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")

    df = df.copy()
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df.loc[df[LABEL_COLUMN].isin([0, 1])].copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    df[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].fillna("").apply(clean_sequence)
    df = df.loc[df[SEQUENCE_COLUMN] != ""].copy()
    df[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].apply(
        lambda sequence: crop_sequence_around_variant(sequence, MAX_LENGTH, VARIANT_CENTER_INDEX)
    )

    print(f"{split_name} CSV: {csv_path}")
    print(f"{split_name} rows loaded for full evaluation: {len(df):,}")
    print(f"{split_name} label distribution:")
    print(df[LABEL_COLUMN].value_counts().sort_index().to_string())
    print()

    if df.empty:
        raise ValueError(f"No usable rows remain for {split_name}.")

    return df.reset_index(drop=True)


def clear_mps_cache_if_needed(device: str) -> None:
    if device != "mps":
        return
    mps_backend = getattr(torch, "mps", None)
    if mps_backend is not None and hasattr(mps_backend, "empty_cache"):
        mps_backend.empty_cache()


def extract_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.ndim >= 2 and item.shape[-1] == 2:
                return item
        return outputs[0]

    raise TypeError("Could not find logits in model outputs.")


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

    bert_layers_text = (patch_dir / "bert_layers.py").read_text(encoding="utf-8")
    return "from .flash_attn_triton import" not in bert_layers_text


def create_patch_from_project_root() -> Path:
    previous_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        patch_dir = create_local_dnabert2_patch()
    finally:
        os.chdir(previous_cwd)

    return resolve_project_path(patch_dir)


def get_local_patch_dir() -> Path:
    patch_dir = resolve_project_path(LOCAL_DNABERT2_PATCH_DIR)
    if patch_dir.exists() and local_patch_is_ready(patch_dir):
        return patch_dir

    print("Local Mac-safe DNABERT-2 patch was not found or is incomplete.")
    return create_patch_from_project_root()


def load_saved_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    pytorch_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        print(f"Loading fine-tuned weights: {safetensors_path}")
        return load_safetensors_file(str(safetensors_path), device="cpu")

    if pytorch_path.exists():
        print(f"Loading fine-tuned weights: {pytorch_path}")
        return torch.load(pytorch_path, map_location="cpu")

    raise FileNotFoundError(
        "Could not find saved model weights. Expected model.safetensors or pytorch_model.bin in "
        f"{model_dir}"
    )


def load_saved_model_with_local_patch(model_dir: Path):
    """Load Mac-safe DNABERT-2 code, then load the saved fine-tuned weights."""
    patch_dir = get_local_patch_dir()
    print(f"Using local Mac-safe DNABERT-2 code: {patch_dir}")
    print("Triton/flash attention disabled for Mac.")

    clear_local_patch_module_cache()
    config = AutoConfig.from_pretrained(str(patch_dir), trust_remote_code=True)
    config = disable_flash_attention_on_config(config)

    saved_config_path = model_dir / "config.json"
    if saved_config_path.exists():
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
        if saved_config.get("id2label"):
            config.id2label = {int(key): value for key, value in saved_config["id2label"].items()}
        if saved_config.get("label2id"):
            config.label2id = saved_config["label2id"]

    model = load_sequence_classification_model(str(patch_dir), config)
    state_dict = load_saved_state_dict(model_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: missing keys while loading saved weights: {len(missing_keys)}")
        print(missing_keys[:10])
    if unexpected_keys:
        print(f"Warning: unexpected keys while loading saved weights: {len(unexpected_keys)}")
        print(unexpected_keys[:10])

    print("Model loaded successfully.")
    return model


def load_saved_model(model_dir: Path, device: str):
    print(f"Trying to load saved model directly from: {model_dir}")
    print(f"Selected device: {device}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        print("Saved model loaded cleanly")
        return model
    except Exception as error:
        print("Direct saved-model loading failed.")
        print(f"Reason: {type(error).__name__}: {error}")
        print("Falling back to training/local_dnabert2_patch.")

    return load_saved_model_with_local_patch(model_dir)


def predict_in_small_batches(model, tokenizer, df: pd.DataFrame, device: str) -> tuple[np.ndarray, np.ndarray]:
    device_object = torch.device(device)
    model.to(device_object)
    model.eval()

    probabilities: list[float] = []
    labels: list[int] = []

    for row_number, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Evaluating"), start=1):
        sequence = getattr(row, SEQUENCE_COLUMN)
        label = int(getattr(row, LABEL_COLUMN))

        encoded = tokenizer(
            sequence,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device_object) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = extract_logits(outputs)
            probability = torch.softmax(logits.float(), dim=-1)[0, 1].detach().cpu().item()

        probabilities.append(float(probability))
        labels.append(label)

        del encoded, outputs, logits
        if row_number % MPS_CACHE_EVERY == 0:
            clear_mps_cache_if_needed(device)

    gc.collect()
    clear_mps_cache_if_needed(device)

    return np.asarray(probabilities, dtype=np.float64), np.asarray(labels, dtype=int)


def metrics_at_threshold(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc_roc": None,
        "confusion_matrix": matrix.tolist(),
        "rows": int(len(labels)),
    }

    if len(np.unique(labels)) == 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def build_threshold_grid(threshold_min: float, threshold_max: float, threshold_step: float) -> np.ndarray:
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)
    thresholds = thresholds[thresholds <= threshold_max + 1e-12]
    return np.round(thresholds, 10)


def tune_threshold_on_validation(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, dict]:
    thresholds = build_threshold_grid(threshold_min, threshold_max, threshold_step)
    if len(thresholds) == 0:
        raise ValueError("No thresholds were generated. Check threshold_min, threshold_max, and threshold_step.")

    best_threshold = float(thresholds[0])
    best_metrics = metrics_at_threshold(probabilities, labels, best_threshold)

    for threshold in thresholds[1:]:
        candidate_metrics = metrics_at_threshold(probabilities, labels, float(threshold))
        if candidate_metrics["mcc"] > best_metrics["mcc"]:
            best_threshold = float(threshold)
            best_metrics = candidate_metrics

    tuning_summary = {
        "threshold_min": float(threshold_min),
        "threshold_max": float(threshold_max),
        "threshold_step": float(threshold_step),
        "thresholds_tested": int(len(thresholds)),
        "best_threshold": best_threshold,
        "best_validation_mcc": float(best_metrics["mcc"]),
    }

    print(f"Best full-validation threshold: {best_threshold:.4f}")
    print(f"Best full-validation MCC: {best_metrics['mcc']:.4f}")
    print()

    return best_threshold, tuning_summary


def choose_threshold(args: argparse.Namespace, probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    if args.threshold is not None:
        print(f"Using threshold provided by --threshold: {args.threshold:.4f}")
        print()
        return float(args.threshold), {
            "mode": "manual",
            "selected_threshold": float(args.threshold),
        }

    if args.tune_threshold:
        print("Tuning threshold on the full validation set.")
        threshold, tuning_summary = tune_threshold_on_validation(
            probabilities,
            labels,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
        )
        tuning_summary["mode"] = "full_validation_mcc"
        tuning_summary["selected_threshold"] = threshold
        return threshold, tuning_summary

    saved_threshold = load_threshold()
    print("--tune_threshold is false and no --threshold was provided.")
    print(f"Falling back to threshold from metrics.json/default: {saved_threshold:.4f}")
    print()
    return saved_threshold, {
        "mode": "saved_or_default",
        "selected_threshold": float(saved_threshold),
    }


def print_metrics(split_name: str, metrics: dict) -> None:
    print("=" * 80)
    print(f"{split_name.upper()} FULL EVALUATION")
    print("=" * 80)
    print(f"Rows: {metrics['rows']:,}")
    print(f"Threshold used: {metrics['threshold']:.4f}")
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"]:
        value = metrics[key]
        if value is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {value:.4f}")

    matrix = metrics["confusion_matrix"]
    print("Confusion matrix:")
    print("                 predicted_0  predicted_1")
    print(f"actual_0         {matrix[0][0]:>11}  {matrix[0][1]:>11}")
    print(f"actual_1         {matrix[1][0]:>11}  {matrix[1][1]:>11}")
    print()


def main() -> None:
    args = parse_args()
    validate_args(args)

    model_dir = choose_model_dir(args.model_dir)
    dataset_dir = find_dataset_dir()
    device = choose_device()

    print("Memory-safe saved model evaluation")
    print(f"Saved model directory: {model_dir}")
    print(f"Selected dataset directory: {dataset_dir}")
    print(f"Selected device: {device}")
    print(f"Tune threshold on full validation set: {args.tune_threshold}")
    if args.threshold is not None:
        print(f"Manual threshold requested: {args.threshold:.4f}")
    else:
        print(
            "Threshold search range: "
            f"{args.threshold_min:.4f} to {args.threshold_max:.4f} "
            f"by {args.threshold_step:.4f}"
        )
    print("Using manual small-batch evaluation. No retraining. No HuggingFace Trainer evaluation.")
    print()

    print("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = load_saved_model(model_dir, device)

    all_metrics = {
        "model_dir": str(model_dir),
        "dataset_dir": str(dataset_dir),
        "device": device,
        "max_length": MAX_LENGTH,
        "variant_center_index": VARIANT_CENTER_INDEX,
        "batch_size": BATCH_SIZE,
        "threshold_args": {
            "tune_threshold": bool(args.tune_threshold),
            "threshold": args.threshold,
            "threshold_min": float(args.threshold_min),
            "threshold_max": float(args.threshold_max),
            "threshold_step": float(args.threshold_step),
        },
    }

    predictions_by_split = {}
    for split_name, filename in ALT_SPLIT_FILES.items():
        csv_path = dataset_dir / filename
        df = load_eval_dataframe(csv_path, split_name)
        probabilities, labels = predict_in_small_batches(model, tokenizer, df, device)
        predictions_by_split[split_name] = {
            "probabilities": probabilities,
            "labels": labels,
        }

    validation_predictions = predictions_by_split["validation"]
    threshold, threshold_selection = choose_threshold(
        args,
        validation_predictions["probabilities"],
        validation_predictions["labels"],
    )

    all_metrics["threshold"] = threshold
    all_metrics["selected_threshold"] = threshold
    all_metrics["threshold_selection"] = threshold_selection

    for split_name, prediction_data in predictions_by_split.items():
        probabilities = prediction_data["probabilities"]
        labels = prediction_data["labels"]
        split_metrics = metrics_at_threshold(probabilities, labels, threshold)
        all_metrics[f"{split_name}_metrics"] = split_metrics
        print_metrics(split_name, split_metrics)

    FULL_EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FULL_EVAL_METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Saved full evaluation metrics to: {FULL_EVAL_METRICS_PATH}")


if __name__ == "__main__":
    main()
