# Optional Local Deployment Model

For a self-contained Hugging Face Space, place the exported model files in:

```text
models/final_model/
```

Then configure:

```bash
MODEL_DIR=./models/final_model
```

The recommended deployment is to keep model weights in a separate Hugging Face
model repository and set `MODEL_DIR` to that repository ID instead.
