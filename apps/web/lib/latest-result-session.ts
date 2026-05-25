import { trimResultForStorage } from "./trim-extracted-text";
import type { CVAnalysisResponse } from "./types";

const SESSION_KEY = "cv-analyzer-latest-result-v1";

export type LatestResultPayload = {
  fileName: string;
  result: CVAnalysisResponse;
};

function isPayload(x: unknown): x is LatestResultPayload {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.fileName === "string" &&
    o.result !== null &&
    typeof o.result === "object" &&
    typeof (o.result as CVAnalysisResponse).success === "boolean"
  );
}

export function saveLatestResultSession(fileName: string, result: CVAnalysisResponse): boolean {
  if (typeof window === "undefined") return false;
  try {
    const payload: LatestResultPayload = {
      fileName,
      result: trimResultForStorage(result),
    };
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

export function loadLatestResultSession(): LatestResultPayload | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    return isPayload(parsed) ? parsed : null;
  } catch {
    return null;
  }
}
