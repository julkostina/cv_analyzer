import { getApiBaseUrl } from "./config";
import { uk } from "./strings-uk";
import type { CVAnalysisResponse } from "./types";

export type AnalyzeOptions = {
  jobDescription?: string;
  jobUrl?: string;
  returnPdf?: boolean;
};

function buildFormData(file: File, options: AnalyzeOptions): FormData {
  const fd = new FormData();
  fd.append("file", file);
  const jd = options.jobDescription?.trim();
  const ju = options.jobUrl?.trim();
  if (jd) fd.append("job_description", jd);
  if (ju) fd.append("job_description_url", ju);
  if (options.returnPdf) fd.append("return_pdf", "true");
  return fd;
}

export async function analyzeCv(
  file: File,
  options: AnalyzeOptions,
): Promise<CVAnalysisResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/v1/analyze`, {
    method: "POST",
    body: buildFormData(file, { ...options, returnPdf: false }),
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

  const data = (await res.json()) as CVAnalysisResponse | { detail?: unknown };

  if (!res.ok) {
    const detail = (data as { detail?: unknown }).detail;
    let msg: string;
    if (typeof detail === "string") {
      msg = detail;
    } else if (Array.isArray(detail)) {
      msg = detail
        .map((d) => {
          if (d && typeof d === "object" && "msg" in d) return String((d as { msg: unknown }).msg);
          return JSON.stringify(d);
        })
        .join("; ");
    } else {
      msg = JSON.stringify(detail ?? data);
    }
    throw new Error(msg || uk.analyzeClient.requestFailed(res.status));
  }

  return data as CVAnalysisResponse;
}

export async function downloadAnalysisPdf(
  file: File,
  options: AnalyzeOptions,
): Promise<Blob> {
  const res = await fetch(`${getApiBaseUrl()}/api/v1/analyze`, {
    method: "POST",
    body: buildFormData(file, { ...options, returnPdf: true }),
  });

  if (!res.ok) {
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const err = (await res.json()) as { detail?: string };
      throw new Error(err.detail ?? uk.analyzeClient.requestFailed(res.status));
    }
    throw new Error(uk.analyzeClient.pdfFailed(res.status));
  }

  return res.blob();
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
      const err = (await res.json()) as { detail?: string };
      throw new Error(err.detail ?? uk.analyzeClient.requestFailed(res.status));
    }
    throw new Error(uk.analyzeClient.pdfFailed(res.status));
  }

  return res.blob();
}
