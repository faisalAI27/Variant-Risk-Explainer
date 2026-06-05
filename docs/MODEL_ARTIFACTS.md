# Model Artifacts

Large trained models are not stored in GitHub.

The final cleaned model folder is kept locally at:

```text
training/training_model_files/
```

This folder is ignored by Git because it contains large model files such as
`model.safetensors` and `training_args.bin`.

The current best model is the 20k DNABERT-2 alternate-sequence model. Model
artifacts should be stored in Google Drive or downloaded local storage rather
than committed to the repository.

For backend use, either:

- place the final model folder at `backend/models/final_model/`, or
- configure the backend model path environment variable to point to
  `training/training_model_files/`.

Keep these small metadata files with the model artifact:

- `metrics.json`
- `full_eval_metrics.json`

The 20k CSV dataset is also ignored by Git:

```text
training/csv_files_20k_alt/
```

Store that dataset externally or regenerate it with the training scripts and
Colab notebook.

This project is for research and education only. It is not a clinical
diagnostic tool.
