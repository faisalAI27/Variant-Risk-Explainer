export type RiskLabel = "likely_benign" | "uncertain" | "likely_pathogenic";

export type VariantRequest = {
  chromosome: string;
  position: number;
  reference: string;
  alternate: string;
  gene?: string | null;
  sequence_context?: string | null;
};

export type VariantAnalysisResponse = {
  request_id: string;
  submitted_at: string;
  input: VariantRequest;
  grch_build: "GRCh38";
  risk_label: RiskLabel;
  confidence: number;
  model_mode: "mock" | "trained";
  explanation: string;
  limitations: string[];
  disclaimer: string;
};
