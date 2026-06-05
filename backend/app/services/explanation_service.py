from __future__ import annotations

import json
import re


LIMITATIONS = [
    "The model uses DNA sequence patterns and does not replace clinical interpretation.",
    "The model performance is limited, with test AUC around 0.5928.",
    "The prediction does not include full clinical evidence, family history, population frequency, or functional studies.",
    "The result should not be used for diagnosis or treatment decisions.",
]

RECOMMENDATION = (
    "This result is for research/demo use only. For any real genetic or medical decision, "
    "consult a qualified clinical genetics professional and use validated clinical databases/testing."
)


OPENAI_SYSTEM_INSTRUCTIONS = """
You explain a research-only DNABERT-2 variant risk demo to non-expert users.
Rules:
- Do not make clinical claims.
- Do not say a variant causes disease.
- Use cautious wording like "the model estimated" and "this may indicate".
- State that the result is not for diagnosis.
- Use only the model output values provided in the prompt.
- Keep the response to one short paragraph.
- Return JSON only with this shape: {"explanation": "..."}
""".strip()


def _confidence_level(prediction_class: int, benign_probability: float, pathogenic_probability: float, threshold: float) -> str:
    if prediction_class == 1:
        if pathogenic_probability >= 0.80:
            return "High model confidence"
        if pathogenic_probability >= 0.60:
            return "Moderate model confidence"
        if pathogenic_probability >= threshold:
            return "Low model confidence"
        return "Low model confidence"

    if benign_probability >= 0.80:
        return "High model confidence"
    if benign_probability >= 0.60:
        return "Moderate model confidence"
    return "Low model confidence"


def _context_text(variant_name: str | None, gene: str | None, sequence_length_used: int | None) -> str:
    details: list[str] = []
    if variant_name:
        details.append(f"variant {variant_name}")
    if gene:
        details.append(f"gene {gene}")
    if sequence_length_used is not None:
        details.append(f"{sequence_length_used} bases used by the model")

    if not details:
        return ""

    return " Context: " + "; ".join(details) + "."


def _generate_rule_based_explanation(
    prediction_class: int,
    prediction_label: str,
    risk_level: str,
    benign_probability: float,
    pathogenic_probability: float,
    threshold: float,
    variant_name: str | None = None,
    gene: str | None = None,
    sequence_length_used: int | None = None,
) -> dict:
    pathogenic_percent = pathogenic_probability * 100.0
    confidence_level = _confidence_level(
        prediction_class=prediction_class,
        benign_probability=benign_probability,
        pathogenic_probability=pathogenic_probability,
        threshold=threshold,
    )
    context = _context_text(variant_name, gene, sequence_length_used)

    if prediction_class == 1:
        explanation = (
            "The DNABERT-2 model estimated this sequence as more similar to pathogenic or likely pathogenic "
            f"variants in the training data. The pathogenic probability is {pathogenic_percent:.1f}%. "
            f"Because this is above the selected threshold of {threshold:.2f}, the model labels it as "
            f"{prediction_label} with an {risk_level.lower()} research-demo risk level."
            f"{context}"
        )
    else:
        explanation = (
            "The DNABERT-2 model estimated this sequence as more similar to benign or likely benign "
            f"variants in the training data. The pathogenic probability is {pathogenic_percent:.1f}%, "
            f"which is below the selected threshold of {threshold:.2f}. The model labels it as "
            f"{prediction_label} with a {risk_level.lower()} research-demo risk level."
            f"{context}"
        )

    return {
        "explanation": explanation,
        "confidence_level": confidence_level,
        "recommendation": RECOMMENDATION,
        "limitations": LIMITATIONS,
    }


def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _generate_openai_explanation(
    fallback: dict,
    prediction_class: int,
    prediction_label: str,
    risk_level: str,
    benign_probability: float,
    pathogenic_probability: float,
    threshold: float,
    openai_api_key: str,
    openai_model: str,
    openai_timeout: float,
    variant_name: str | None,
    gene: str | None,
    sequence_length_used: int | None,
) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI package is not installed. Using rule-based explanation.")
        return fallback

    context = {
        "variant_name": variant_name,
        "gene": gene,
        "prediction_class": prediction_class,
        "prediction_label": prediction_label,
        "risk_level": risk_level,
        "benign_probability": round(benign_probability, 6),
        "pathogenic_probability": round(pathogenic_probability, 6),
        "pathogenic_probability_percent": round(pathogenic_probability * 100.0, 1),
        "threshold": threshold,
        "sequence_length_used": sequence_length_used,
        "model": "DNABERT-2 fine-tuned on a ClinVar alternate-sequence research dataset",
        "test_auc_roc": 0.5928,
    }
    user_prompt = (
        "Write the explanation paragraph for this model output. "
        "The explanation must be understandable to a beginner and must stay research/demo-only.\n\n"
        f"Model output JSON:\n{json.dumps(context, indent=2)}"
    )

    try:
        client = OpenAI(api_key=openai_api_key, timeout=openai_timeout)
        response = client.responses.create(
            model=openai_model,
            instructions=OPENAI_SYSTEM_INSTRUCTIONS,
            input=user_prompt,
            max_output_tokens=300,
        )
        output_text = str(getattr(response, "output_text", "")).strip()
        if not output_text:
            print("OpenAI explanation response was empty. Using rule-based explanation.")
            return fallback

        parsed = _extract_json_object(output_text)
        explanation = str(parsed.get("explanation", "")).strip()
        if not explanation:
            print("OpenAI explanation JSON did not include explanation. Using rule-based explanation.")
            return fallback

        enhanced = dict(fallback)
        enhanced["explanation"] = explanation
        return enhanced
    except Exception as exc:  # pragma: no cover - network/API failures vary.
        print(f"OpenAI explanation failed: {type(exc).__name__}: {exc}. Using rule-based explanation.")
        return fallback


def generate_explanation(
    prediction_class: int,
    prediction_label: str,
    risk_level: str,
    benign_probability: float,
    pathogenic_probability: float,
    threshold: float,
    variant_name: str | None = None,
    gene: str | None = None,
    sequence_length_used: int | None = None,
    use_openai: bool = False,
    openai_api_key: str = "",
    openai_model: str = "gpt-4.1-mini",
    openai_timeout: float = 12.0,
) -> dict:
    fallback = _generate_rule_based_explanation(
        prediction_class=prediction_class,
        prediction_label=prediction_label,
        risk_level=risk_level,
        benign_probability=benign_probability,
        pathogenic_probability=pathogenic_probability,
        threshold=threshold,
        variant_name=variant_name,
        gene=gene,
        sequence_length_used=sequence_length_used,
    )

    if not use_openai:
        return fallback
    if not openai_api_key:
        print("USE_OPENAI_EXPLANATION is true, but OPENAI_API_KEY is missing. Using rule-based explanation.")
        return fallback

    return _generate_openai_explanation(
        fallback=fallback,
        prediction_class=prediction_class,
        prediction_label=prediction_label,
        risk_level=risk_level,
        benign_probability=benign_probability,
        pathogenic_probability=pathogenic_probability,
        threshold=threshold,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_timeout=openai_timeout,
        variant_name=variant_name,
        gene=gene,
        sequence_length_used=sequence_length_used,
    )
