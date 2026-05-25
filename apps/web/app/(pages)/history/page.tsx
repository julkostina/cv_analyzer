"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  clearHistory,
  loadHistory,
  removeHistoryEntry,
  type HistoryEntry,
} from "../../../lib/history-storage";
import { buildUnifiedNarrativeParagraphs, formatPercent } from "../../../lib/analysis-format";
import { uk } from "../../../lib/strings-uk";
import { PageHeader } from "../../../components/page-header/PageHeader";
import { PageLayout } from "../../../components/page-layout/PageLayout";
import surface from "../../../components/ui/surface.module.css";
import historyStyles from "./history.module.css";

function formatDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat("uk-UA", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function displayJobUrl(url: string): string {
  try {
    const u = new URL(url);
    const path = u.pathname.length > 40 ? `${u.pathname.slice(0, 37)}…` : u.pathname;
    return `${u.hostname}${path}`;
  } catch {
    return url.length > 56 ? `${url.slice(0, 53)}…` : url;
  }
}

function HistoryJobRow({ entry }: { entry: HistoryEntry }) {
  const h = uk.history;
  const url = entry.jobUrl?.trim();
  if (url) {
    return (
      <p className={historyStyles.jobRow}>
        <span className={historyStyles.jobLabel}>{h.jobLink}:</span>
        <a className={historyStyles.jobLink} href={url} target="_blank" rel="noopener noreferrer">
          {displayJobUrl(url)}
        </a>
      </p>
    );
  }
  const preview = entry.jobDescriptionPreview?.trim();
  if (preview) {
    return (
      <div className={historyStyles.jobRow}>
        <span className={historyStyles.jobLabel}>{h.jobTextOnly}</span>
        <p className={historyStyles.jobPreview}>{preview}</p>
      </div>
    );
  }
  return null;
}

function HistorySummaryBlock({ entry }: { entry: HistoryEntry }) {
  const h = uk.history;
  const analysis = entry.result.analysis;
  const summary = analysis && typeof analysis.summary === "string" ? analysis.summary : null;
  const paragraphs = buildUnifiedNarrativeParagraphs(
    summary,
    analysis?.strengths,
    analysis?.weaknesses,
    entry.result.recommendations,
  );
  if (paragraphs.length === 0) return null;

  return (
    <div className={historyStyles.summaryBlock}>
      <p className={historyStyles.summaryTitle}>{h.analysisSummary}</p>
      <div className={historyStyles.summaryScroll}>
        {paragraphs.map((para, i) => (
          <p key={i}>{para}</p>
        ))}
      </div>
    </div>
  );
}

export default function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const h = uk.history;

  const refresh = useCallback(() => {
    setEntries(loadHistory());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onRemove = useCallback(
    (id: string) => {
      removeHistoryEntry(id);
      refresh();
    },
    [refresh],
  );

  const onClear = useCallback(() => {
    clearHistory();
    refresh();
  }, [refresh]);

  return (
    <PageLayout>
      <PageHeader
        title={h.title}
        subtitle={
          <>
            {h.subtitleBeforeLink}
            <Link href="/">{h.backToAnalyzer}</Link>
          </>
        }
      />

      {entries.length === 0 ? (
        <section className={surface.card} aria-labelledby="empty-history">
          <h2 id="empty-history" className={surface.cardTitle}>
            {h.emptyTitle}
          </h2>
          <p className={surface.reasoning}>{h.emptyBody}</p>
        </section>
      ) : (
        <>
          <div className={historyStyles.toolbar}>
            <p className={historyStyles.count}>{h.count(entries.length)}</p>
            <button type="button" className={historyStyles.dangerBtn} onClick={onClear}>
              {h.clearAll}
            </button>
          </div>
          <ul className={historyStyles.list}>
            {entries.map((e) => (
              <li key={e.id} className={historyStyles.item}>
                <div className={historyStyles.itemHead}>
                  <span className={historyStyles.fileName}>{e.fileName}</span>
                  <time className={historyStyles.time} dateTime={e.savedAt}>
                    {formatDate(e.savedAt)}
                  </time>
                </div>
                <HistoryJobRow entry={e} />
                <div className={historyStyles.meta}>
                  {e.result.match_score != null ? (
                    <span>
                      {h.match}: {formatPercent(e.result.match_score)}
                    </span>
                  ) : (
                    <span>{h.matchDash}</span>
                  )}
                  {(e.result.recommendations?.length ?? 0) > 0 ? (
                    <span>{h.tips(e.result.recommendations!.length)}</span>
                  ) : null}
                </div>
                <HistorySummaryBlock entry={e} />
                <button type="button" className={historyStyles.removeBtn} onClick={() => onRemove(e.id)}>
                  {h.remove}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </PageLayout>
  );
}
