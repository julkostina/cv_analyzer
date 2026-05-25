import { getApiBaseUrl } from "./config";
import { uk } from "./strings-uk";
import type { CVAnalysisResponse } from "./types";

type AnalyzeOptions = {
  jobDescription?: string;
  jobUrl?: string;
};

function isCvAnalysisResponse(data: unknown): data is CVAnalysisResponse {
  if (typeof data !== "object" || data === null) return false;
  return typeof (data as { success?: unknown }).success === "boolean";
}

function formatApiErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (d && typeof d === "object" && "msg" in d) {
          return String((d as { msg: unknown }).msg);
        }
        return JSON.stringify(d);
      })
      .join("; ");
  }
  return JSON.stringify(detail ?? fallback);
}

function buildFormData(file: File, options: AnalyzeOptions): FormData {
  const fd = new FormData();
  fd.append("file", file);
  const jd = options.jobDescription?.trim();
  const ju = options.jobUrl?.trim();
  if (jd) fd.append("job_description", jd);
  if (ju) fd.append("job_description_url", ju);
  return fd;
}

export async function analyzeCv(
  file: File,
  options: AnalyzeOptions,
): Promise<CVAnalysisResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/v1/analyze`, {
    method: "POST",
    body: buildFormData(file, options),
  });

  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("application/json")) {
    const text = await res.text();
    throw new Error(
      res.ok
        ? uk.analyzeClient.unexpectedJson
        : uk.analyzeClient.errorStatus(res.status, text.slice(0, 200)),
    );
  }

  const data: unknown = await res.json();

  if (!res.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? (data as { detail: unknown }).detail
        : data;
    throw new Error(
      formatApiErrorDetail(detail, uk.analyzeClient.requestFailed(res.status)),
    );
  }

  if (!isCvAnalysisResponse(data)) {
    throw new Error(uk.analyzeClient.unexpectedJson);
  }
  return data;
}

export async function downloadReportPdfFromResult(result: CVAnalysisResponse): Promise<Blob> {
  const res = await fetch(`${getApiBaseUrl()}/api/v1/report/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result),
  });

  if (!res.ok) {
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const err: unknown = await res.json();
      const detail =
        typeof err === "object" && err !== null && "detail" in err
          ? (err as { detail: unknown }).detail
          : err;
      const msg =
        typeof detail === "string" ? detail : uk.analyzeClient.requestFailed(res.status);
      throw new Error(msg);
    }
    throw new Error(uk.analyzeClient.pdfFailed(res.status));
  }

  return res.blob();
}
