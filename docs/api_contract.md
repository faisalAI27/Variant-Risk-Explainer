# API Contract

Base URL for local development:

```text
http://localhost:8000
```

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "mps",
  "model_dir": "../training/training_model_files",
  "threshold": 0.16,
  "model_name": "DNABERT-2 ClinVar 20k",
  "load_error": null
}
```

## Analyze Variant Sequence

```http
POST /analyze
Content-Type: application/json
```

### Request Body

```json
{
  "sequence": "ACGTACGTACGTACGT",
  "variant_name": "Demo Variant A",
  "gene": "BRCA1",
  "notes": "Synthetic demo request"
}
```

### Request Fields

- `sequence`: required DNA sequence using only `A`, `C`, `G`, `T`, or `N`.
- `variant_name`: optional display name.
- `gene`: optional gene symbol for display/explanation context.
- `notes`: optional demo notes.

### Response Body

```json
{
  "variant_name": "Demo Variant A",
  "gene": "BRCA1",
  "prediction_class": 1,
  "prediction_label": "Pathogenic / Likely pathogenic",
  "risk_level": "Elevated",
  "benign_probability": 0.671463,
  "pathogenic_probability": 0.328537,
  "threshold": 0.16,
  "model_name": "DNABERT-2 ClinVar 20k",
  "sequence_length_used": 64,
  "explanation": "The model estimated...",
  "explanation_source": "openai",
  "confidence_level": "Low model confidence",
  "recommendation": "This result is for research/demo use only...",
  "limitations": [
    "The model uses DNA sequence patterns and does not replace clinical interpretation.",
    "The model performance is limited, with test AUC around 0.5928.",
    "The prediction does not include full clinical evidence, family history, population frequency, or functional studies.",
    "The result should not be used for diagnosis or treatment decisions."
  ],
  "disclaimer": "Research/demo use only. This model is not a clinical diagnostic system and must not be used for medical decisions."
}
```

### Explanation Sources

- `openai`: OpenAI successfully rewrote the explanation paragraph.
- `rule-based`: local rule-based explanation was used because AI explanation is disabled.
- `rule-based-fallback`: AI explanation was enabled, but the key was missing or the OpenAI request failed.

### Error Example

Invalid DNA characters return a clear error:

```json
{
  "detail": "sequence contains invalid DNA characters: XZ"
}
```

## Safety Boundary

All outputs are research/demo outputs only. They are not clinical classifications and must not be used for diagnosis, treatment, or medical decision-making.
