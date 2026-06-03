#!/usr/bin/env python
"""Fine-tune DNABERT-2 for research-only ClinVar sequence classification.

Run this script in Google Colab or another GPU notebook environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ID_TO_LABEL = {
    0: "likely_benign",
    1: "likely_pathogenic",
}
LABEL_TO_ID = {label: idx for idx, label in ID_TO_LABEL.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", required=True, help="Prepared JSONL from prepare_clinvar_dataset.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved model artifacts.")
    parser.add_argument("--model-name", default="zhihan1996/DNABERT-2-117M", help="Hugging Face base model.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_examples(args.dataset_jsonl)
    train_dataset, eval_dataset, test_dataset = split_dataset(
        dataset,
        test_size=args.test_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )

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

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    metadata = {
        "base_model": args.model_name,
        "genome_build": "GRCh38",
        "labels": ID_TO_LABEL,
        "test_metrics": test_metrics,
        "research_only": True,
        "disclaimer": "For research and education only. Not for medical diagnosis.",
    }
    (final_model_dir / "variant_risk_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved final model to {final_model_dir}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
