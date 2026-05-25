export function formatPercent(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return "—";
  return `${Math.round(score * 1000) / 10}%`;
}

/** Whole units 0–100 for a score in 0…1 (no percent symbol). */
export function matchScoreUnits(score: number | null | undefined): number | null {
  if (score == null || Number.isNaN(score)) return null;
  return Math.round(Math.max(0, Math.min(1, score)) * 100);
}

const PLACEHOLDER_REC = /^див\.?\s*підсумок\s*вище\.?$/i;

const COACHING_SECTION_LABEL_RE =
  /(?:^|[\n.]\s*)(?:Сильні сторони|Очікування роботодавця|Відповідність і прогалини|Прогалини|Слабкі сторони|Висновок(?:\s+і\s+поради)?|Висновки)\s*:\s*/gi;

/** If the model returned a JSON blob, extract the summary field. */
export function unwrapCoachingSummary(text: string): string {
  const t = text.trim();
  if (!t.startsWith("{") || !t.includes("summary")) return t;
  try {
    const obj = JSON.parse(t) as { summary?: unknown };
    if (typeof obj.summary === "string" && obj.summary.trim()) {
      return obj.summary.trim();
    }
  } catch {
    /* keep original */
  }
  return t;
}

/** Remove «Сильні сторони:»-style labels; merge into flowing paragraphs. */
export function polishCoachingSummary(text: string): string {
  const t = unwrapCoachingSummary(text).trim();
  if (!t) return t;
  let out = t.replace(COACHING_SECTION_LABEL_RE, ". ");
  if (out === t) return t;
  out = out.replace(/\.\s*\./g, ".").replace(/\s+/g, " ").trim();
  const sentences = out.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  if (sentences.length <= 5) return out;
  const per = Math.max(2, Math.ceil(sentences.length / 3));
  const paras: string[] = [];
  for (let i = 0; i < sentences.length; i += per) {
    paras.push(sentences.slice(i, i + per).join(" "));
  }
  return paras.join("\n\n");
}

function toProseFragment(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") {
    const t = value.trim();
    return t.length > 0 ? t : null;
  }
  if (Array.isArray(value)) {
    const parts = value.map((item) => String(item).trim()).filter(Boolean);
    if (parts.length === 0) return null;
    return parts
      .map((part) => (part.endsWith(".") || part.endsWith("!") || part.endsWith("?") ? part : `${part}.`))
      .join(" ");
  }
  const t = String(value).trim();
  return t.length > 0 ? t : null;
}

/** One coaching-style block: summary first; legacy fields merged if summary is short or missing. */
export function buildUnifiedNarrativeParagraphs(
  summary: string | null | undefined,
  strengths: unknown,
  weaknesses: unknown,
  recommendations: string[] | null | undefined,
): string[] {
  const main = polishCoachingSummary(summary?.trim() ?? "");
  if (main.length >= 200) {
    return main
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  const chunks: string[] = [];
  if (main) chunks.push(main);

  const s = toProseFragment(strengths);
  const w = toProseFragment(weaknesses);
  if (s) chunks.push(s);
  if (w) chunks.push(w);

  const recs = (recommendations ?? [])
    .map((r) => r.trim())
    .filter((r) => r.length > 0 && !PLACEHOLDER_REC.test(r));
  if (recs.length > 0) {
    chunks.push(
      recs
        .map((part) => (part.endsWith(".") || part.endsWith("!") || part.endsWith("?") ? part : `${part}.`))
        .join(" "),
    );
  }

  if (chunks.length === 0) return [];
  if (chunks.length === 1) {
    return chunks[0]!
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  return chunks;
}

export function triggerBlobDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
