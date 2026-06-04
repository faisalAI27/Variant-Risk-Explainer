#!/usr/bin/env python3
"""Check the best available PyTorch device for local Mac training."""

from __future__ import annotations

import torch


def choose_device() -> str:
    """Select CUDA first, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def main() -> None:
    device = choose_device()

    print("PyTorch device check")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    print(f"MPS available: {mps_available}")
    print(f"Using device: {device}")

    if device == "mps":
        print("Apple Silicon acceleration is available through MPS.")
    elif device == "cuda":
        print("CUDA GPU acceleration is available.")
    else:
        print("WARNING: CPU training will be slow. Use small smoke tests locally, or use a GPU machine for full DNABERT-2 training.")


if __name__ == "__main__":
    main()
