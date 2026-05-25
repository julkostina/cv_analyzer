"use client";

import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { analyzeCv } from "../../lib/analyze-client";
import { addHistoryEntry } from "../../lib/history-storage";
import { saveLatestResultSession } from "../../lib/latest-result-session";
import { uk } from "../../lib/strings-uk";
import type { CVAnalysisResponse } from "../../lib/types";
import { AnalysisResults } from "./AnalysisResults";
import { AnalyzerForm } from "./AnalyzerForm";
import styles from "./HomeAnalyzer.module.css";

export function HomeAnalyzer() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inlineResult, setInlineResult] = useState<CVAnalysisResponse | null>(null);
  const t = uk.homeAnalyzer;

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      setInlineResult(null);
      if (!file) {
        setError(t.errors.noFile);
        return;
      }
      setLoading(true);
      try {
        const data = await analyzeCv(file, {
          jobDescription: jobDescription || undefined,
          jobUrl: jobUrl || undefined,
        });
        const saved = saveLatestResultSession(file.name, data);
        if (data.success) {
          addHistoryEntry({
            fileName: file.name,
            result: data,
            jobUrl: jobUrl || undefined,
            jobDescription: jobDescription || undefined,
          });
        }
        if (!saved) {
          setInlineResult(data);
          setError(t.errors.sessionBlocked);
          return;
        }
        router.push("/results");
      } catch (err) {
        setError(err instanceof Error ? err.message : t.errors.unknown);
      } finally {
        setLoading(false);
      }
    },
    [file, jobDescription, jobUrl, router, t.errors],
  );

  return (
    <section className={styles.service} aria-labelledby="service-heading">
      <div className="container">
        <h2 id="service-heading" className={styles.serviceHeading}>
          {t.heading}
        </h2>
        <p className={styles.serviceSub}>{t.sub}</p>

        <AnalyzerForm
          onFileChange={setFile}
          jobDescription={jobDescription}
          onJobDescriptionChange={setJobDescription}
          jobUrl={jobUrl}
          onJobUrlChange={setJobUrl}
          loading={loading}
          onSubmit={onSubmit}
        />

        {error ? (
          <div className={styles.error} role="alert">
            {error}
          </div>
        ) : null}

        {inlineResult ? <AnalysisResults result={inlineResult} /> : null}
      </div>
    </section>
  );
}
