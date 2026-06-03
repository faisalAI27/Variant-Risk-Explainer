# Model Card

## Model Name

Variant Risk Explainer DNABERT-2 research classifier.

## Intended Use

This model is intended for education and research demonstrations that explore how genomic sequence models can be fine-tuned on ClinVar-derived labels.

It is not intended for:

- Medical diagnosis.
- Treatment decisions.
- Patient risk stratification.
- Clinical reporting.
- Replacing expert variant interpretation.

## Base Model

The training pipeline is designed for DNABERT-2 through Hugging Face Transformers.

## Genome Build

GRCh38 is used consistently for ClinVar coordinates and reference sequence extraction.

## Training Data

The pipeline prepares examples from ClinVar GRCh38 VCF records. It focuses on single nucleotide variants with clinical significance labels that can be mapped into binary or compact research classes.

Example label mapping:

- `Pathogenic` and `Likely_pathogenic` -> `likely_pathogenic`
- `Benign` and `Likely_benign` -> `likely_benign`
- conflicting, uncertain, or unsupported labels -> skipped or held out depending on notebook settings

## Inputs

The model receives sequence windows centered on a submitted variant. The demo scripts use alternate-allele sequence windows for classification.

## Outputs

The backend maps model scores into demo labels:

- `likely_benign`
- `uncertain`
- `likely_pathogenic`

## Evaluation

Recommended evaluation:

- Stratified train, validation, and test splits.
- Accuracy, precision, recall, F1, ROC-AUC where appropriate.
- Confusion matrix.
- Per-class metrics.
- Label distribution audit.
- Gene and chromosome holdout experiments for stronger leakage checks.

## Limitations

- ClinVar labels can change over time.
- ClinVar records may be conflicting, incomplete, or biased toward heavily studied genes.
- Sequence-only models omit clinical, segregation, population frequency, functional assay, inheritance, and literature evidence.
- A small Colab fine-tune is not sufficient for clinical validity.
- Performance on rare variant classes, indels, structural variants, and non-human references is not established.

## Ethical and Safety Notes

All UI and API responses must state that this is research-only software. Outputs should be treated as exploratory model behavior, not medical evidence.
