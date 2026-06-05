import type { AnalyzeRequest, AnalyzeResponse, HealthResponse } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

function formatApiError(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
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

  return fallback;
}

async function parseResponse<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`${fallback} Status ${response.status}.`);
    }
    throw new Error(formatApiError(payload, fallback));
  }

  return response.json() as Promise<T>;
}

export async function analyzeVariant(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
  } catch {
    throw new Error("Backend is not running. Start FastAPI backend first.");
  }

  return parseResponse<AnalyzeResponse>(response, "Analysis request failed.");
}

export async function healthCheck(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/health`, {
      cache: "no-store"
    });
  } catch {
    throw new Error("Backend is not running. Start FastAPI backend first.");
  }

  return parseResponse<HealthResponse>(response, "Health check failed.");
}
