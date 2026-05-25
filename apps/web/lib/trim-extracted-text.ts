import type { CVAnalysisResponse } from "./types";

const MAX_EXTRACTED_TEXT_LEN = 12000;

export function trimResultForStorage(r: CVAnalysisResponse): CVAnalysisResponse {
  const t = r.extracted_text;
  if (!t || t.length <= MAX_EXTRACTED_TEXT_LEN) return r;
  return {
    ...r,
    extracted_text: `${t.slice(0, MAX_EXTRACTED_TEXT_LEN)}\n[truncated for storage]`,
  };
}
