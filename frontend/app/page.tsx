"use client";

import { AlertCircle, CheckCircle2, Dna, ShieldCheck } from "lucide-react";
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
        <div className="brandLine">
          <div className="brandMark" aria-hidden>
            <Dna size={24} />
          </div>
          <p className="eyebrow">AI Genomics Platform</p>
        </div>
        <div className="heroCopy">
          <h1>Variant Risk Explainer</h1>
          <p className="subtitle">
            AI-powered genomic sequence analysis for variant risk interpretation.
          </p>
          <p className="heroDescription">
            Enter a DNA sequence to receive a model-based variant risk assessment and explanation.
          </p>
        </div>
      </header>

      <section className="statusPanel" aria-label="Backend status">
        {health ? (
          <>
            <CheckCircle2 aria-hidden size={20} />
            <div>
              <strong>Analysis service connected</strong>
              <span>{health.model_loaded ? "Genomic analysis model is ready." : "Model is not currently available."}</span>
              {!health.model_loaded && health.load_error ? <span className="statusWarning">{health.load_error}</span> : null}
            </div>
          </>
        ) : (
          <>
            <AlertCircle aria-hidden size={20} />
            <div>
              <strong>Analysis service unavailable</strong>
              <span>{healthError || "Backend is not running. Start FastAPI backend first."}</span>
            </div>
          </>
        )}
      </section>

      <section className="workspace" aria-label="Variant analysis workspace">
        <div className="mainColumn">
          <VariantForm isLoading={isLoading} onAnalyze={handleAnalyze} onClearResult={clearCurrentResult} />
          <ResultCard error={error} isLoading={isLoading} result={result} />
        </div>

        <HistoryPanel history={history} selectedId={result?.id} onClear={() => setHistory([])} onSelect={setResult} />
      </section>

      <footer className="appFooter">
        <div className="noticeHeading">
          <ShieldCheck aria-hidden size={18} />
          <strong>Important notice</strong>
        </div>
        <p>
          This system provides AI-assisted variant risk interpretation and is intended for educational and
          research-oriented analysis. It does not replace clinical genetic testing or professional medical evaluation.
        </p>
        <details className="noticeDetails">
          <summary>Review system limitations</summary>
          <ul>
            <li>Sequence patterns are evaluated without full clinical evidence or family history.</li>
            <li>Population frequency and functional study evidence are not included.</li>
            <li>Results should be reviewed alongside validated genetics resources and qualified professional guidance.</li>
          </ul>
        </details>
      </footer>
    </main>
  );
}
