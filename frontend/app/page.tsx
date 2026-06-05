"use client";

import { Activity, AlertCircle, CheckCircle2 } from "lucide-react";
import { useEffect, useState } from "react";
import { HistoryPanel } from "@/components/HistoryPanel";
import { ResultCard } from "@/components/ResultCard";
import { VariantForm } from "@/components/VariantForm";
import { analyzeVariant, healthCheck } from "@/lib/api";
import type { AnalysisHistoryItem, AnalyzeRequest, AnalyzeResponse, HealthResponse } from "@/types";

function makeHistoryItem(result: AnalyzeResponse, input: AnalyzeRequest): AnalysisHistoryItem {
  const cleanedSequence = input.sequence.replace(/\s+/g, "").toUpperCase();
  return {
    ...result,
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    submitted_at: new Date().toISOString(),
    sequence_preview: cleanedSequence.length > 26 ? `${cleanedSequence.slice(0, 26)}...` : cleanedSequence
  };
}

export default function Home() {
  const [result, setResult] = useState<AnalysisHistoryItem | null>(null);
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const nextHealth = await healthCheck();
        if (!cancelled) {
          setHealth(nextHealth);
          setHealthError(null);
        }
      } catch (caught) {
        if (!cancelled) {
          setHealth(null);
          setHealthError(caught instanceof Error ? caught.message : "Backend is not running. Start FastAPI backend first.");
        }
      }
    }

    loadHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleAnalyze(input: AnalyzeRequest) {
    setIsLoading(true);
    setError(null);

    try {
      const response = await analyzeVariant(input);
      const nextResult = makeHistoryItem(response, input);
      setResult(nextResult);
      setHistory((current) => [nextResult, ...current].slice(0, 8));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis request failed.");
    } finally {
      setIsLoading(false);
    }
  }

  function clearCurrentResult() {
    setResult(null);
    setError(null);
  }

  return (
    <main className="appShell">
      <header className="appHeader">
        <div>
          <p className="eyebrow">DNABERT-2 research demo</p>
          <h1>Variant Risk Explainer</h1>
          <p className="subtitle">
            A research/demo tool using DNABERT-2 to estimate whether a DNA variant sequence looks benign or pathogenic.
          </p>
        </div>
        <p className="safetyBanner">Research only. Not for medical diagnosis.</p>
      </header>

      <section className="statusPanel" aria-label="Backend status">
        {health ? (
          <>
            <CheckCircle2 aria-hidden size={20} />
            <div>
              <strong>Backend connected</strong>
              <span>
                Model loaded: {health.model_loaded ? "yes" : "no"} · Device: {health.device} · Threshold:{" "}
                {health.threshold.toFixed(2)}
              </span>
              {!health.model_loaded && health.load_error ? <span className="statusWarning">{health.load_error}</span> : null}
            </div>
          </>
        ) : (
          <>
            <AlertCircle aria-hidden size={20} />
            <div>
              <strong>Backend unavailable</strong>
              <span>{healthError || "Backend is not running. Start FastAPI backend first."}</span>
            </div>
          </>
        )}
      </section>

      <section className="workspace" aria-label="Variant analysis workspace">
        <div className="mainColumn">
          <VariantForm isLoading={isLoading} onAnalyze={handleAnalyze} onClearResult={clearCurrentResult} />
          <ResultCard error={error} isLoading={isLoading} result={result} />
          <section className="infoPanel">
            <div className="panelHeader">
              <h2>Model information</h2>
              <Activity aria-hidden size={20} />
            </div>
            <dl className="modelStats">
              <div>
                <dt>Model</dt>
                <dd>DNABERT-2 fine-tuned on 20k ClinVar alternate-sequence dataset</dd>
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
                <dt>Threshold</dt>
                <dd>0.16</dd>
              </div>
            </dl>
            <p className="disclaimer">
              This tool is for research/demo purposes only and is not a clinical diagnostic system.
            </p>
          </section>
        </div>

        <HistoryPanel history={history} selectedId={result?.id} onClear={() => setHistory([])} onSelect={setResult} />
      </section>
    </main>
  );
}
