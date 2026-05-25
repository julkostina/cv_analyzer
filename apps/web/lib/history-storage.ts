import { trimResultForStorage } from "./trim-extracted-text";
import type { CVAnalysisResponse } from "./types";

const STORAGE_KEY = "cv-analyzer-history-v1";
const MAX_ENTRIES = 50;

export type HistoryEntry = {
  id: string;
  savedAt: string;
  fileName: string;
  jobUrl?: string | null;
  jobDescriptionPreview?: string | null;
  result: CVAnalysisResponse;
};

function isHistoryEntry(x: unknown): x is HistoryEntry {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.id === "string" &&
    typeof o.savedAt === "string" &&
    typeof o.fileName === "string" &&
    o.result !== null &&
    typeof o.result === "object" &&
    typeof (o.result as CVAnalysisResponse).success === "boolean"
  );
}

export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isHistoryEntry);
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

export function addHistoryEntry(params: {
  fileName: string;
  result: CVAnalysisResponse;
  jobUrl?: string | null;
  jobDescription?: string | null;
}): void {
  if (typeof window === "undefined") return;
  if (!params.result.success) return;
  const url = params.jobUrl?.trim() || null;
  const desc = params.jobDescription?.trim() || null;
  const entry: HistoryEntry = {
    id: crypto.randomUUID(),
    savedAt: new Date().toISOString(),
    fileName: params.fileName,
    jobUrl: url,
    jobDescriptionPreview: url ? null : desc ? desc.slice(0, 400) : null,
    result: trimResultForStorage(params.result),
  };
  const prev = loadHistory();
  saveHistory([entry, ...prev].slice(0, MAX_ENTRIES));
}

export function removeHistoryEntry(id: string): void {
  if (typeof window === "undefined") return;
  saveHistory(loadHistory().filter((e) => e.id !== id));
}

export function clearHistory(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(STORAGE_KEY);
}
