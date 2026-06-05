# Demo Examples

These examples use synthetic/demo DNA sequences only. They are not real clinical examples and should not be interpreted as medical evidence.

## Example 1: Demo Variant A

Request:

```json
{
  "variant_name": "Demo Variant A",
  "gene": "BRCA1",
  "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
  "notes": "Synthetic short sequence for endpoint testing."
}
```

Example curl:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "variant_name": "Demo Variant A",
    "gene": "BRCA1",
    "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
    "notes": "Synthetic short sequence for endpoint testing."
  }'
```

## Example 2: Demo Variant B

Request:

```json
{
  "variant_name": "Demo Variant B",
  "gene": "TP53",
  "sequence": "TTGCAAGCTTAGGCTAACCGTTGCAAGCTTAGGCTAACCGTTGCAAGCTTAGGCTAACCGTTGCAAGC",
  "notes": "Second synthetic sequence for frontend demo testing."
}
```

Example curl:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "variant_name": "Demo Variant B",
    "gene": "TP53",
    "sequence": "TTGCAAGCTTAGGCTAACCGTTGCAAGCTTAGGCTAACCGTTGCAAGCTTAGGCTAACCGTTGCAAGC",
    "notes": "Second synthetic sequence for frontend demo testing."
  }'
```

## Example 3: Invalid Sequence

Request:

```json
{
  "variant_name": "Invalid Demo Variant",
  "gene": "TP53",
  "sequence": "ACGTXYZACGT",
  "notes": "This should fail validation because X, Y, and Z are not valid DNA bases."
}
```

Expected behavior:

- HTTP status: `400`
- Error message includes invalid DNA characters.

Example response:

```json
{
  "detail": "sequence contains invalid DNA characters: XYZ"
}
```
