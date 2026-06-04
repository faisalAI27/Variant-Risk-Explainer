# Colab Heavy Training Guide

This workflow is for research and education only. The model output is not a
medical diagnosis and should not be used for clinical decisions.

## Steps

1. Push this repository to GitHub.
2. Open `training/colab_dnabert2_heavy_training.ipynb` in Google Colab.
3. In the repo clone cell, replace:

   ```python
   REPO_URL = "PASTE_YOUR_GITHUB_REPO_URL_HERE"
   ```

   with your GitHub repository URL.

4. Run the notebook cells one by one.
5. Start with the 10k dataset:

   ```bash
   python training/prepare_larger_clinvar_dataset.py --target_total 10000
   ```

6. Only try the 20k dataset after the 10k dataset prepares and trains correctly.
7. Save the final model and metrics to Google Drive from the notebook.

## Notes

- Use a Colab GPU runtime.
- Do not add API keys or secrets to the notebook.
- Dataset preparation uses caching and progress files, so reruns can resume.
- The training script uses CUDA when available, MPS on Mac, and CPU as a slow fallback.
