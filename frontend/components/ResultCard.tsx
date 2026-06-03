"use client";

import { AlertTriangle, BadgeCheck, FlaskConical, Loader2 } from "lucide-react";
import type { RiskLabel, VariantAnalysisResponse } from "@/types";

type ResultCardProps = {
  error: string | null;
  isLoading: boolean;
  result: VariantAnalysisResponse | null;
};

const LABEL_TEXT: Record<RiskLabel, string> = {
  likely_benign: "Likely benign",
  uncertain: "Uncertain",
  likely_pathogenic: "Likely pathogenic"
};

export function ResultCard({ error, isLoading, result }: ResultCardProps) {
  if (isLoading) {
    return (
      <section className="resultPanel statePanel" aria-live="polite">
        <Loader2 aria-hidden className="spin" size={24} />
        <p>Analyzing variant</p>
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
        <p>No result yet</p>
      </section>
    );
  }

  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <section className="resultPanel">
      <div className="panelHeader">
        <h2>Result</h2>
        <span className="buildPill">{result.model_mode}</span>
      </div>

      <div className={`riskBadge ${result.risk_label}`}>
        <BadgeCheck aria-hidden size={20} />
        <span>{LABEL_TEXT[result.risk_label]}</span>
      </div>

      <div className="confidenceBlock">
        <div className="confidenceTopline">
          <span>Confidence</span>
          <strong>{confidencePercent}%</strong>
        </div>
        <div className="confidenceTrack" aria-hidden>
          <div className="confidenceFill" style={{ width: `${confidencePercent}%` }} />
        </div>
      </div>

      <dl className="resultDetails">
        <div>
          <dt>Variant</dt>
          <dd>
            {result.input.chromosome}:{result.input.position} {result.input.reference}&gt;{result.input.alternate}
          </dd>
        </div>
        <div>
          <dt>Gene</dt>
          <dd>{result.input.gene || "Not provided"}</dd>
        </div>
        <div>
          <dt>Build</dt>
          <dd>{result.grch_build}</dd>
        </div>
      </dl>

      <p className="explanation">{result.explanation}</p>

      <div className="limitations">
        <h3>Limitations</h3>
        <ul>
          {result.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <p className="disclaimer">{result.disclaimer}</p>
    </section>
  );
}
