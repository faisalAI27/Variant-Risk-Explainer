# Training

This folder contains the training pipeline for fine-tuning DNABERT-2 on ClinVar-derived GRCh38 examples.

The original path is Google Colab, but this project also includes Mac-friendly local scripts for small research runs and smoke tests.

## Contents

- `01_prepare_clinvar_dataset.ipynb`: Colab notebook for downloading ClinVar GRCh38 VCF data and preparing binary SNV/small-indel CSV splits with sequence columns.
- `colab_dnabert2_clinvar_finetune.ipynb`: Colab notebook for fine-tuning DNABERT-2 from `train_with_sequences.csv`, `val_with_sequences.csv`, and `test_with_sequences.csv`.
- `requirements-colab.txt`: Python packages for the notebook.
- `requirements-mac.txt`: Python packages for local Mac training.
- `scripts/prepare_clinvar_dataset.py`: converts ClinVar GRCh38 VCF records into sequence classification examples.
- `scripts/train_dnabert2_classifier.py`: fine-tunes DNABERT-2 with Hugging Face Transformers.
- `train_smoke_test.py`: runs a tiny DNABERT-2 local smoke test.
- `train_local_dnabert2.py`: runs Mac-friendly local DNABERT-2 fine-tuning.
- `utils/clinvar_parser.py`: manual gzip VCF parsing and variant filtering helpers.
- `utils/label_utils.py`: ClinVar clinical-significance label mapping helpers.
- `utils/sequence_fetcher.py`: UCSC API and optional local FASTA sequence extraction helpers.

## Data Requirements

Use GRCh38 consistently:

- ClinVar GRCh38 VCF: `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz`
- Sequence extraction uses the UCSC hg38 API by default.
- Optional GRCh38 reference FASTA: provide a local or Google Drive path in Colab if using local FASTA sequence extraction.

The fine-tuning notebook expects the sequence CSV files from the preparation notebook.

## Dataset Check

Before local training, verify the prepared CSV files:

```bash
python training/check_dataset.py
```

The checker expects `train_with_sequences.csv`, `val_with_sequences.csv`, and `test_with_sequences.csv` in `data/processed/`. It also falls back to `training/csv_files/` for sample files.

## Mac Local Setup

For local Mac training setup, see:

```text
training/setup_mac.md
```

Check the available PyTorch device with:

```bash
python training/check_device.py
```

Run a tiny smoke test first:

```bash
python training/train_smoke_test.py
```

Then run local DNABERT-2 fine-tuning:

```bash
python training/train_local_dnabert2.py
```

By default, local training uses `training/csv_files_large_alt/` when the 5,000-row alternate-sequence CSVs exist. It uses all available rows, trains for 5 epochs, freezes the DNABERT-2 encoder, applies class weights, crops long sequences around variant index 512, and tunes the final classification threshold on the validation split.

Local Mac evaluation is memory-safe by default: epoch evaluation is disabled, validation/test prediction runs in small batches, and final metrics use an evaluation subset of 300 rows per split. To evaluate all validation/test rows after training, pass `--eval_subset_size 0`.

To train the classifier plus the last encoder layer on Mac, use:

```bash
python training/train_local_dnabert2.py --unfreeze_last_n_layers 1
```

To build a larger balanced ClinVar dataset before training, use:

```bash
python training/prepare_larger_clinvar_dataset.py
```

Check the larger dataset before training:

```bash
python training/check_large_dataset.py
```

Then train from the larger alternate-sequence CSVs explicitly:

```bash
python training/train_local_dnabert2.py --train_csv training/csv_files_large_alt/train_with_alt_sequences.csv --val_csv training/csv_files_large_alt/val_with_alt_sequences.csv --test_csv training/csv_files_large_alt/test_with_alt_sequences.csv
```

## Colab Flow

1. Open `01_prepare_clinvar_dataset.ipynb` in Google Colab.
2. Run dataset preparation and sequence extraction.
3. Save or download `train_with_sequences.csv`, `val_with_sequences.csv`, and `test_with_sequences.csv`.
4. Upload those CSV files into `training/csv_files/` or `data/processed/`.
5. Open `colab_dnabert2_clinvar_finetune.ipynb`.
6. Run DNABERT-2 fine-tuning.
7. Export the saved model directory.

## Safety Notice

This pipeline creates research-only model artifacts. A trained model from this folder is not clinically validated and must not be used for diagnosis or treatment decisions.
