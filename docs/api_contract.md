# API Contract

Base URL for local development:

```text
http://localhost:8000
```

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "service": "variant-risk-explainer",
  "model_mode": "mock"
}
```

## Analyze Variant

```http
POST /analyze
Content-Type: application/json
```

### Request Body

```json
{
  "chromosome": "7",
  "position": 140753336,
  "reference": "A",
  "alternate": "T",
  "gene": "BRAF",
  "sequence_context": "ACGTACGTACGT"
}
```

### Fields

- `chromosome`: GRCh38 chromosome name. Accepts `1` through `22`, `X`, `Y`, `MT`, and optional `chr` prefix.
- `position`: 1-based GRCh38 genomic coordinate.
- `reference`: reference allele using `A`, `C`, `G`, `T`, or `N`.
- `alternate`: alternate allele using `A`, `C`, `G`, `T`, or `N`.
- `gene`: optional gene symbol for display and explanation context.
- `sequence_context`: optional GRCh38 sequence context around the variant.

### Response Body

```json
{
  "request_id": "0e7c3d55-6f46-4c19-a1fd-860bc4f8a88d",
  "submitted_at": "2026-06-04T10:00:00Z",
  "input": {
    "chromosome": "7",
    "position": 140753336,
    "reference": "A",
    "alternate": "T",
    "gene": "BRAF",
    "sequence_context": "ACGTACGTACGT"
  },
  "grch_build": "GRCh38",
  "risk_label": "uncertain",
  "confidence": 0.54,
  "model_mode": "mock",
  "explanation": "Mock mode produced a deterministic research-only score from variant features. No clinical meaning should be inferred.",
  "limitations": [
    "Research demo only.",
    "Not validated for diagnosis or treatment decisions.",
    "ClinVar labels may be incomplete, conflicting, or biased."
  ],
  "disclaimer": "For research and education only. Not for medical diagnosis."
}
```

### Risk Labels

- `likely_benign`
- `uncertain`
- `likely_pathogenic`

These labels are demo categories only and are not clinical classifications.

### Error Example

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "position"],
      "msg": "Value error, position must be a positive GRCh38 coordinate"
    }
  ]
}
```
