import type { VariantAnalysisResponse, VariantRequest } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function formatApiError(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return "Analysis request failed.";
  }

  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return String(item);
      })
      .join(" ");
  }

  return "Analysis request failed.";
}

export async function analyzeVariant(input: VariantRequest): Promise<VariantAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(input)
  });

  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`Analysis request failed with status ${response.status}.`);
    }
    throw new Error(formatApiError(payload));
  }

  return response.json();
}
