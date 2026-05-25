"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AnalysisResults } from "../../components/home/AnalysisResults";
import { PageHeader } from "../../components/page-header/PageHeader";
import styles from "../../components/home/HomeAnalyzer.module.css";
import { downloadReportPdfFromResult } from "../../lib/analyze-client";
import { triggerBlobDownload } from "../../lib/analysis-format";
import { loadLatestResultSession, type LatestResultPayload } from "../../lib/latest-result-session";
import { uk } from "../../lib/strings-uk";
import layoutStyles from "./results.module.css";

export default function ResultsPage() {
  const [payload, setPayload] = useState<LatestResultPayload | null | undefined>(undefined);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const r = uk.results;
  const t = uk.homeAnalyzer;

  useEffect(() => {
    setPayload(loadLatestResultSession());
  }, []);

  const onDownloadPdf = useCallback(async () => {
    if (!payload?.result.success) return;
    setPdfLoading(true);
    setPdfError(null);
    try {
      const blob = await downloadReportPdfFromResult(payload.result);
      triggerBlobDownload(blob, "zvit-analiz-rezyume.pdf");
    } catch (err) {
      setPdfError(err instanceof Error ? err.message : t.errors.pdfDownload);
    } finally {
      setPdfLoading(false);
    }
  }, [payload, t.errors.pdfDownload]);

  if (payload === undefined) {
    return (
      <div className={layoutStyles.page}>
        <main className={layoutStyles.content}>
          <PageHeader title={r.title} subtitle={r.loading} />
        </main>
      </div>
    );
  }

  if (!payload) {
    return (
      <div className={layoutStyles.page}>
        <main className={layoutStyles.content}>
          <PageHeader
            title={r.title}
            subtitle={
              <>
                {r.emptyBeforeLink}
                <Link href="/">{r.runOnHome}</Link>.
              </>
            }
          />
        </main>
      </div>
    );
  }

  const { result } = payload;

  return (
    <div className={layoutStyles.page}>
      <main className={layoutStyles.content}>
        <PageHeader title={r.title} subtitle={payload.fileName} />

        {result.success === false && result.error ? (
          <div className={styles.error} role="alert">
            {result.error}
          </div>
        ) : null}

        {pdfError ? (
          <div className={styles.error} role="alert">
            {pdfError}
          </div>
        ) : null}

        <AnalysisResults
          result={result}
          showPdfDownload={result.success}
          pdfLoading={pdfLoading}
          onDownloadPdf={onDownloadPdf}
        />
      </main>
    </div>
  );
}
