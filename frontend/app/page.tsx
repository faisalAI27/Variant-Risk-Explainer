"use client";

import { useState } from "react";
import { HistoryPanel } from "@/components/HistoryPanel";
import { ResultCard } from "@/components/ResultCard";
import { VariantForm } from "@/components/VariantForm";
import { analyzeVariant } from "@/lib/api";
import type { VariantAnalysisResponse, VariantRequest } from "@/types";

export default function Home() {
  const [result, setResult] = useState<VariantAnalysisResponse | null>(null);
  const [history, setHistory] = useState<VariantAnalysisResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleAnalyze(input: VariantRequest) {
    setIsLoading(true);
    setError(null);

    try {
      const nextResult = await analyzeVariant(input);
      setResult(nextResult);
      setHistory((current) => [nextResult, ...current.filter((item) => item.request_id !== nextResult.request_id)].slice(0, 8));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis request failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="appShell">
      <header className="appHeader">
        <div>
          <p className="eyebrow">GRCh38 research demo</p>
          <h1>Variant Risk Explainer</h1>
        </div>
        <p className="safetyBanner">Research only. Not for medical diagnosis.</p>
      </header>

      <section className="workspace" aria-label="Variant analysis workspace">
        <div className="mainColumn">
          <VariantForm isLoading={isLoading} onAnalyze={handleAnalyze} />
          <ResultCard error={error} isLoading={isLoading} result={result} />
        </div>

        <HistoryPanel
          history={history}
          selectedId={result?.request_id}
          onClear={() => setHistory([])}
          onSelect={setResult}
        />
      </section>
    </main>
  );
}
