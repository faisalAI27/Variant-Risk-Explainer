export type AnalyzeRequest = {
  sequence: string;
  variant_name?: string | null;
  gene?: string | null;
  notes?: string | null;
};

export type AnalyzeResponse = {
  variant_name: string | null;
  gene: string | null;
  prediction_class: 0 | 1;
  prediction_label: string;
  risk_level: "Lower" | "Elevated";
  benign_probability: number;
  pathogenic_probability: number;
  threshold: number;
  model_name: string;
  sequence_length_used: number;
  explanation: string;
  explanation_source: "rule-based" | "openai" | "rule-based-fallback" | string;
  confidence_level: string;
  recommendation: string;
  limitations: string[];
  disclaimer: string;
};

export type HealthResponse = {
  status: "ok" | "degraded";
  model_loaded: boolean;
  device: string;
  model_dir: string;
  threshold: number;
  model_name: string;
  explanation_mode: "openai" | "rule-based";
  ai_explanation_enabled: boolean;
  openai_configured: boolean;
  load_error?: string | null;
};

export type AnalysisHistoryItem = AnalyzeResponse & {
  id: string;
  submitted_at: string;
  sequence_preview: string;
};
