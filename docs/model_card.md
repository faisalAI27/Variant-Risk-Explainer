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

Final label mapping:

- `Pathogenic` and `Likely_pathogenic` -> `1`
- `Benign` and `Likely_benign` -> `0`
- conflicting, uncertain, or unsupported labels -> skipped or held out depending on notebook settings

## Inputs

The model receives sequence windows centered on a submitted variant. The demo scripts use alternate-allele sequence windows for classification.

## Outputs

The backend maps model probabilities into demo labels:

- `0`: Benign / Likely benign
- `1`: Pathogenic / Likely pathogenic

The current research threshold for class `1` is `0.16`.

## Evaluation

Final confirmed 20k alternate-sequence evaluation:

- Accuracy: `0.5537`
- Precision: `0.5384`
- Recall: `0.7533`
- F1: `0.6280`
- MCC: `0.1171`
- AUC ROC: `0.5928`

Recommended future evaluation:

- Larger independent test sets.
- Gene and chromosome holdout experiments.
- Leakage checks across related variants.
- Per-variant-type reporting for SNVs and indels.
- Calibration analysis across thresholds.

## Limitations

- ClinVar labels can change over time.
- ClinVar records may be conflicting, incomplete, or biased toward heavily studied genes.
- Sequence-only models omit clinical, segregation, population frequency, functional assay, inheritance, and literature evidence.
- A small Colab fine-tune is not sufficient for clinical validity.
- Performance on rare variant classes, indels, structural variants, and non-human references is not established.

## Ethical and Safety Notes

All UI and API responses must state that this is research-only software. Outputs should be treated as exploratory model behavior, not medical evidence.
