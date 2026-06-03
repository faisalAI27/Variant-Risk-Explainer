#!/usr/bin/env python
"""Fine-tune DNABERT-2 for research-only ClinVar sequence classification.

Run this script in Google Colab or another GPU notebook environment.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from training.utils.label_utils import assign_binary_label


ID_TO_LABEL = {
    0: "benign_or_likely_benign",
    1: "pathogenic",
}
LABEL_TO_ID = {label: idx for idx, label in ID_TO_LABEL.items()}
VALID_BASES = set("ACGTN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", help="Legacy prepared JSONL from prepare_clinvar_dataset.py.")
    parser.add_argument("--train-csv", help="CSV containing sequence and label columns for training.")
    parser.add_argument("--val-csv", help="CSV containing sequence and label columns for validation.")
    parser.add_argument("--test-csv", help="CSV containing sequence and label columns for final evaluation.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved model artifacts.")
    parser.add_argument("--model-name", default="zhihan1996/DNABERT-2-117M", help="Hugging Face base model.")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--min-sequence-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clean-clnsig", action="store_true", help="Do not re-filter CLNSIG labels in CSV inputs.")
    return parser.parse_args()


def load_examples(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append(
                {
                    "sequence": record["sequence"],
                    "labels": int(record["label_id"]),
                    "chromosome": record.get("chromosome"),
                    "position": record.get("position"),
                    "gene": record.get("gene"),
                }
            )

    if not records:
        raise ValueError("No examples found. Check ClinVar, FASTA, and label filtering settings.")
    return Dataset.from_list(records)


def clean_sequence(value: object) -> str:
    sequence = "".join(base for base in str(value).upper() if base in VALID_BASES)
    return sequence


def load_csv_split(
    path: str,
    sequence_column: str,
    label_column: str,
    min_sequence_length: int,
    clean_clnsig: bool,
) -> Dataset:
    df = pd.read_csv(path)
    missing_columns = [column for column in [sequence_column, label_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    start_rows = len(df)
    df = df.copy()
    df[sequence_column] = df[sequence_column].map(clean_sequence)
    df = df[df[sequence_column].str.len() >= min_sequence_length].copy()

    if clean_clnsig and "CLNSIG" in df.columns:
        df["_clean_label"] = df["CLNSIG"].apply(assign_binary_label)
        df = df[df["_clean_label"].notna()].copy()
        df[label_column] = df["_clean_label"].astype(int)

    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df[df[label_column].isin([0, 1])].copy()
    df[label_column] = df[label_column].astype(int)
    if df.empty:
        raise ValueError(f"No usable rows remain after cleaning {path}.")

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "sequence": row[sequence_column],
                "labels": int(row[label_column]),
                "variant_id": row.get("variant_id"),
                "chromosome": row.get("CHROM"),
                "position": row.get("POS"),
                "gene": row.get("gene_symbol"),
                "clnsig": row.get("CLNSIG"),
            }
        )

    print(
        f"{path}: kept {len(records):,}/{start_rows:,} rows "
        f"with label counts {df[label_column].value_counts().sort_index().to_dict()}"
    )
    return Dataset.from_list(records)


def load_csv_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    required = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--test-csv": args.test_csv,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"CSV training requires: {', '.join(missing)}")

    clean_clnsig = not args.no_clean_clnsig
    train_dataset = load_csv_split(
        args.train_csv,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        min_sequence_length=args.min_sequence_length,
        clean_clnsig=clean_clnsig,
    )
    eval_dataset = load_csv_split(
        args.val_csv,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        min_sequence_length=args.min_sequence_length,
        clean_clnsig=clean_clnsig,
    )
    test_dataset = load_csv_split(
        args.test_csv,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        min_sequence_length=args.min_sequence_length,
        clean_clnsig=clean_clnsig,
    )
    return train_dataset, eval_dataset, test_dataset


def split_dataset(dataset: Dataset, test_size: float, eval_size: float, seed: int):
    train_eval = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="labels")
    eval_fraction = eval_size / (1.0 - test_size)
    train_valid = train_eval["train"].train_test_split(
        test_size=eval_fraction,
        seed=seed,
        stratify_by_column="labels",
    )
    return train_valid["train"], train_valid["test"], train_eval["test"]


def compute_metrics(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv or args.val_csv or args.test_csv:
        train_dataset, eval_dataset, test_dataset = load_csv_splits(args)
        dataset_source = {
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
        }
    elif args.dataset_jsonl:
        dataset = load_examples(args.dataset_jsonl)
        train_dataset, eval_dataset, test_dataset = split_dataset(
            dataset,
            test_size=args.test_size,
            eval_size=args.eval_size,
            seed=args.seed,
        )
        dataset_source = {"dataset_jsonl": args.dataset_jsonl}
    else:
        raise ValueError("Provide either --dataset-jsonl or all of --train-csv, --val-csv, and --test-csv.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    def tokenize(batch):
        return tokenizer(batch["sequence"], truncation=True, max_length=args.max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        trust_remote_code=True,
    )

    training_args = build_training_args(args, output_dir)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    metadata = {
        "base_model": args.model_name,
        "genome_build": "GRCh38",
        "labels": ID_TO_LABEL,
        "dataset_source": dataset_source,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "test_rows": len(test_dataset),
        "max_length": args.max_length,
        "test_metrics": test_metrics,
        "research_only": True,
        "disclaimer": "For research and education only. Not for medical diagnosis.",
    }
    (final_model_dir / "variant_risk_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved final model to {final_model_dir}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
