# Mac Local Training Setup

This guide sets up a Mac-friendly Python environment for local DNABERT-2 experiments.

Full DNABERT-2 training can be slow on a Mac, especially on CPU. Start with small smoke tests before running larger jobs.

## CSV File Locations

Training scripts should look for sequence CSVs in this order:

1. `data/processed/`
2. `training/csv_files/`

Your current CSV files are in:

```text
training/csv_files/train_with_sequences.csv
training/csv_files/val_with_sequences.csv
training/csv_files/test_with_sequences.csv
```

That location is supported.

## 1. Create Virtual Environment

Run this from the project root:

```bash
python3 -m venv .venv
```

## 2. Activate It

```bash
source .venv/bin/activate
```

Your terminal prompt should now show `.venv`.

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 4. Install Requirements

```bash
pip install -r training/requirements-mac.txt
```

## 5. Check PyTorch Device

```bash
python training/check_device.py
```

Expected output includes one of:

```text
Using device: mps
```

or:

```text
Using device: cpu
```

If you see `Using device: cpu`, the project will still work, but local training will be slow. Use very small smoke tests locally and use a GPU environment for full model training.

## 6. Check Dataset

Before training, verify the CSV files:

```bash
python training/check_dataset.py
```

This confirms that the `sequence` and `label` columns exist, labels are encoded as `0` and `1`, sequence values are present, and class balance is reasonable.
