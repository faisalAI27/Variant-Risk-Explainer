# Training

This folder contains the Google Colab training pipeline for fine-tuning DNABERT-2 on ClinVar-derived GRCh38 examples.

Training should be run in Google Colab or another dedicated GPU notebook environment, not on the local development machine.

## Contents

- `01_prepare_clinvar_dataset.ipynb`: Colab notebook for downloading ClinVar GRCh38 VCF data and preparing binary SNV/small-indel CSV splits.
- `colab_dnabert2_clinvar_finetune.ipynb`: notebook entry point for Colab.
- `requirements-colab.txt`: Python packages for the notebook.
- `scripts/prepare_clinvar_dataset.py`: converts ClinVar GRCh38 VCF records into sequence classification examples.
- `scripts/train_dnabert2_classifier.py`: fine-tunes DNABERT-2 with Hugging Face Transformers.
- `utils/clinvar_parser.py`: manual gzip VCF parsing and variant filtering helpers.
- `utils/label_utils.py`: ClinVar clinical-significance label mapping helpers.
- `utils/sequence_fetcher.py`: UCSC API and optional local FASTA sequence extraction helpers.

## Data Requirements

Use GRCh38 consistently:

- ClinVar GRCh38 VCF: `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz`
- GRCh38 reference FASTA: provide a local or Google Drive path in Colab.

The preparation script validates reference alleles against the FASTA and skips records that do not match.

## Colab Flow

1. Open `colab_dnabert2_clinvar_finetune.ipynb` in Google Colab.
2. Install `requirements-colab.txt`.
3. Download ClinVar GRCh38 VCF.
4. Mount Google Drive or upload a GRCh38 FASTA.
5. Run dataset preparation.
6. Run DNABERT-2 fine-tuning.
7. Export the saved model directory.

## Safety Notice

This pipeline creates research-only model artifacts. A trained model from this folder is not clinically validated and must not be used for diagnosis or treatment decisions.
