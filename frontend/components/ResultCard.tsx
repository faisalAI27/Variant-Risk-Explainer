"use client";

import { AlertTriangle, BadgeCheck, FlaskConical, Loader2 } from "lucide-react";
import type { AnalyzeResponse } from "@/types";

type ResultCardProps = {
  error: string | null;
  isLoading: boolean;
  result: AnalyzeResponse | null;
};

function asPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function explanationSourceLabel(source: string): string {
  if (source === "openai") {
    return "OpenAI";
  }
  if (source === "rule-based-fallback") {
    return "Rule-based fallback";
  }
  return "Rule-based";
}

export function ResultCard({ error, isLoading, result }: ResultCardProps) {
  if (isLoading) {
    return (
      <section className="resultPanel statePanel" aria-live="polite">
        <Loader2 aria-hidden className="spin" size={24} />
        <p>Analyzing sequence with DNABERT-2...</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="resultPanel errorPanel" role="alert">
        <AlertTriangle aria-hidden size={22} />
        <div>
          <h2>Request Error</h2>
          <p>{error}</p>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="resultPanel statePanel">
        <FlaskConical aria-hidden size={24} />
        <p>Submit a DNA sequence to see the research demo output.</p>
      </section>
    );
  }

  const riskClass = result.prediction_class === 1 ? "pathogenic" : "benign";

  return (
    <section className="resultPanel">
      <div className="panelHeader">
        <h2>Result</h2>
        <span className="buildPill">Research demo</span>
      </div>

      <div className={`riskBadge ${riskClass}`}>
        <BadgeCheck aria-hidden size={20} />
        <span>{result.prediction_label}</span>
      </div>

      <dl className="resultDetails">
        <div>
          <dt>Risk level</dt>
          <dd>{result.risk_level}</dd>
        </div>
        <div>
          <dt>Pathogenic probability</dt>
          <dd>{asPercent(result.pathogenic_probability)}</dd>
        </div>
        <div>
          <dt>Benign probability</dt>
          <dd>{asPercent(result.benign_probability)}</dd>
        </div>
        <div>
          <dt>Threshold</dt>
          <dd>{result.threshold.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Sequence length used</dt>
          <dd>{result.sequence_length_used.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{result.model_name}</dd>
        </div>
      </dl>

      <div className="probabilityBlock">
        <div className="confidenceTopline">
          <span>Pathogenic probability</span>
          <strong>{asPercent(result.pathogenic_probability)}</strong>
        </div>
        <div className="confidenceTrack" aria-hidden>
          <div className={`confidenceFill ${riskClass}`} style={{ width: asPercent(result.pathogenic_probability) }} />
        </div>
      </div>

      <div className="explanation">
        <h3>Explanation</h3>
        <p>{result.explanation}</p>
        <p>
          <strong>Confidence level:</strong> {result.confidence_level}
        </p>
        <p>
          <strong>Explanation source:</strong> {explanationSourceLabel(result.explanation_source)}
        </p>
      </div>

      <div className="recommendationBlock">
        <h3>Recommendation</h3>
        <p>{result.recommendation}</p>
      </div>

      <div className="limitations">
        <h3>Limitations</h3>
        <ul>
          {result.limitations.map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
      </div>

      <p className="disclaimer">
        <strong>Research/demo only. Not for clinical diagnosis.</strong>
        <br />
        {result.disclaimer}
      </p>
    </section>
  );
}
