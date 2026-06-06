"use client";

import {
  AlertTriangle,
  BadgeCheck,
  ChevronDown,
  CircleGauge,
  FlaskConical,
  Lightbulb,
  Loader2,
  SlidersHorizontal
} from "lucide-react";
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
    return "AI-assisted";
  }
  if (source === "rule-based-fallback") {
    return "Rule-based fallback";
  }
  return "Rule-based";
}

function professionalizeText(value: string): string {
  return value
    .replace(/\bresearch[/-]demo\b/gi, "research-oriented")
    .replace(/\bdemo\b/gi, "analysis");
}

export function ResultCard({ error, isLoading, result }: ResultCardProps) {
  if (isLoading) {
    return (
      <section className="resultPanel statePanel loadingState" aria-live="polite">
        <div className="stateIcon">
          <Loader2 aria-hidden className="spin" size={24} />
        </div>
        <div>
          <h2>Analyzing genomic sequence...</h2>
          <p>Evaluating sequence patterns and preparing an interpretation.</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="resultPanel errorPanel" role="alert">
        <AlertTriangle aria-hidden size={22} />
        <div>
          <h2>Analysis could not be completed</h2>
          <p>{error}</p>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="resultPanel statePanel">
        <div className="stateIcon">
          <FlaskConical aria-hidden size={24} />
        </div>
        <div>
          <h2>Ready for analysis</h2>
          <p>Enter a DNA sequence to generate a model-based variant risk assessment.</p>
        </div>
      </section>
    );
  }

  const riskClass = result.prediction_class === 1 ? "pathogenic" : "benign";
  const riskCategory = result.prediction_class === 1 ? "Elevated risk" : "Lower risk";

  return (
    <section className={`resultPanel resultComplete ${riskClass}`}>
      <div className="panelHeader resultHeader">
        <div>
          <p className="sectionEyebrow">Analysis output</p>
          <h2>Variant Assessment</h2>
        </div>
        <span className="resultStatus">
          <BadgeCheck aria-hidden size={16} />
          Complete
        </span>
      </div>

      <div className="assessmentSummary">
        <div>
          <span className="assessmentLabel">Overall assessment</span>
          <h3>{result.prediction_label}</h3>
          {(result.variant_name || result.gene) && (
            <p className="assessmentContext">
              {[result.variant_name, result.gene].filter(Boolean).join(" · ")}
            </p>
          )}
        </div>
        <span className={`riskBadge ${riskClass}`}>{riskCategory}</span>
      </div>

      <div className="probabilityGrid" aria-label="Prediction likelihoods">
        <div className="probabilityCard pathogenicProbability">
          <div className="probabilityHeading">
            <span>Pathogenic likelihood</span>
            <strong>{asPercent(result.pathogenic_probability)}</strong>
          </div>
          <div
            className="probabilityTrack"
            role="progressbar"
            aria-label="Pathogenic likelihood"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(result.pathogenic_probability * 100)}
          >
            <div className="probabilityFill pathogenic" style={{ width: asPercent(result.pathogenic_probability) }} />
          </div>
        </div>

        <div className="probabilityCard benignProbability">
          <div className="probabilityHeading">
            <span>Benign likelihood</span>
            <strong>{asPercent(result.benign_probability)}</strong>
          </div>
          <div
            className="probabilityTrack"
            role="progressbar"
            aria-label="Benign likelihood"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(result.benign_probability * 100)}
          >
            <div className="probabilityFill benign" style={{ width: asPercent(result.benign_probability) }} />
          </div>
        </div>
      </div>

      <div className="confidenceSummary">
        <CircleGauge aria-hidden size={20} />
        <div>
          <span>Confidence level</span>
          <strong>{result.confidence_level}</strong>
        </div>
      </div>

      <div className="interpretationSection">
        <h3>Explanation</h3>
        <p>{professionalizeText(result.explanation)}</p>
      </div>

      <div className="recommendationBlock">
        <Lightbulb aria-hidden size={20} />
        <div>
          <h3>Recommendation</h3>
          <p>{professionalizeText(result.recommendation)}</p>
        </div>
      </div>

      <details className="technicalDetails">
        <summary>
          <span>
            <SlidersHorizontal aria-hidden size={18} />
            Technical details
          </span>
          <ChevronDown aria-hidden className="detailsChevron" size={18} />
        </summary>
        <dl className="technicalGrid">
          <div>
            <dt>Model</dt>
            <dd>{result.model_name}</dd>
          </div>
          <div>
            <dt>Decision threshold</dt>
            <dd>{result.threshold.toFixed(2)}</dd>
          </div>
          <div>
            <dt>Sequence length used</dt>
            <dd>{result.sequence_length_used.toLocaleString()} bases</dd>
          </div>
          <div>
            <dt>Prediction class</dt>
            <dd>{result.prediction_class}</dd>
          </div>
          <div>
            <dt>Test AUC</dt>
            <dd>0.5928</dd>
          </div>
          <div>
            <dt>Test F1</dt>
            <dd>0.6280</dd>
          </div>
          <div>
            <dt>Test MCC</dt>
            <dd>0.1171</dd>
          </div>
          <div>
            <dt>Explanation source</dt>
            <dd>{explanationSourceLabel(result.explanation_source)}</dd>
          </div>
        </dl>
      </details>
    </section>
  );
}
