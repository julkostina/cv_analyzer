import { buildUnifiedNarrativeParagraphs, formatPercent, matchScoreUnits } from "../../lib/analysis-format";
import { gaugeAriaLabel, uk } from "../../lib/strings-uk";
import type { CVAnalysisResponse } from "../../lib/types";
import styles from "./HomeAnalyzer.module.css";

type AnalysisResultsProps = {
  result: CVAnalysisResponse | null;
  showPdfDownload?: boolean;
  pdfLoading?: boolean;
  onDownloadPdf?: () => void;
};

const u = uk.analysisResults;

function formatSemanticWeight(n: number | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return (Math.round(n * 1000) / 1000).toString();
}

/** Верхня півкола: плоска хорда внизу, дуга відкривається вгору (стиль спідометра). */
const GAUGE_CX = 50;
const GAUGE_CY = 48;
const GAUGE_R = 38;
const GAUGE_STROKE = 10;
const GAUGE_ARC_LENGTH = Math.PI * GAUGE_R;
/** sweep-flag 1 = верхня півдуга (хорда внизу, відкриття вгору). */
const GAUGE_PATH = `M ${GAUGE_CX - GAUGE_R} ${GAUGE_CY} A ${GAUGE_R} ${GAUGE_R} 0 0 1 ${GAUGE_CX + GAUGE_R} ${GAUGE_CY}`;

function JobMatchGaugeSvg({ progress }: { progress: number }) {
  const t = Math.max(0, Math.min(1, progress));
  const filled = GAUGE_ARC_LENGTH * t;
  const dashProgress = `${filled} ${GAUGE_ARC_LENGTH}`;

  return (
    <svg
      className={styles.matchGaugeSvg}
      viewBox="0 0 100 58"
      preserveAspectRatio="xMidYMid meet"
      overflow="visible"
      role="presentation"
      aria-hidden
    >
      <path
        className={styles.matchGaugeTrackRim}
        d={GAUGE_PATH}
        fill="none"
        strokeWidth={GAUGE_STROKE}
        strokeLinecap="round"
      />
      <path
        className={styles.matchGaugeArc}
        d={GAUGE_PATH}
        fill="none"
        strokeWidth={GAUGE_STROKE}
        strokeLinecap="round"
        strokeDasharray={dashProgress}
      />
    </svg>
  );
}

type SemanticBreakdown = NonNullable<CVAnalysisResponse["semantic_breakdown"]>;

function JobMatchSemanticRow({ breakdown }: { breakdown: SemanticBreakdown }) {
  return (
    <div className={styles.semanticRow}>
      <div className={styles.semanticMetric}>
        {u.semanticRow.skills}: {formatPercent(breakdown.skills_similarity ?? 0)}
      </div>
      <span className={styles.semanticSep} aria-hidden>
        ·
      </span>
      <div className={styles.semanticMetric}>
        {u.semanticRow.experience}: {formatPercent(breakdown.experience_similarity ?? 0)}
      </div>
      <span className={styles.semanticSep} aria-hidden>
        ·
      </span>
      <div className={styles.semanticMetric}>
        {u.semanticRow.overall}: {formatPercent(breakdown.overall_similarity ?? 0)}
      </div>
    </div>
  );
}

function JobMatchBlock({
  score,
  breakdown,
  weights,
}: {
  score: number;
  breakdown: CVAnalysisResponse["semantic_breakdown"];
  weights: CVAnalysisResponse["semantic_weights"];
}) {
  const units = matchScoreUnits(score);
  if (units == null) return null;

  const wSkills = weights?.skills ?? 0.5;
  const wExp = weights?.experience ?? 0.3;
  const wOverall = weights?.overall ?? 0.2;
  const weightsLine = breakdown
    ? u.weightsLine(formatSemanticWeight(wSkills), formatSemanticWeight(wExp), formatSemanticWeight(wOverall))
    : null;

  return (
    <div className={styles.matchBlock}>
      <figure className={styles.matchGaugeFigure} aria-label={gaugeAriaLabel(units)}>
        <JobMatchGaugeSvg progress={score} />
        <div className={styles.matchScoreStack}>
          <span className={styles.matchScoreUnits}>{units}</span>
          <span className={styles.matchScoreSub}>{u.outOf100}</span>
        </div>
      </figure>
      {breakdown ? <JobMatchSemanticRow breakdown={breakdown} /> : null}
      <MatchExplainabilityBlock weightsLine={weightsLine} />
    </div>
  );
}

function MatchExplainabilityBlock({ weightsLine }: { weightsLine: string | null }) {
  if (!weightsLine) return null;

  return (
    <details className={styles.explainability}>
      <summary>{u.explainabilityTitle}</summary>
      <p className={styles.explainabilityWeights}>{weightsLine}</p>
      <p className={styles.explainabilityIntro}>{u.explainabilityIntro}</p>
    </details>
  );
}

function AnalysisSummarySection({
  summary,
  strengths,
  weaknesses,
  recommendations,
}: {
  summary: string | null;
  strengths: unknown;
  weaknesses: unknown;
  recommendations: string[] | null | undefined;
}) {
  const paragraphs = buildUnifiedNarrativeParagraphs(summary, strengths, weaknesses, recommendations);
  if (paragraphs.length === 0) return null;

  return (
    <section className={styles.card}>
      <h2 className={styles.cardTitle}>{u.analysisSummary}</h2>
      <div className={styles.analysisProse}>
        {paragraphs.map((para, i) => (
          <p key={i}>{para}</p>
        ))}
      </div>
    </section>
  );
}

export function AnalysisResults({
  result,
  showPdfDownload,
  pdfLoading,
  onDownloadPdf,
}: AnalysisResultsProps) {
  if (!result) return null;

  if (!result.success) {
    return (
      <div className={styles.results}>
        {result.extracted_text ? (
          <section className={styles.card} aria-labelledby="partial-heading">
            <h2 id="partial-heading" className={styles.cardTitle}>
              {u.partialTitle}
            </h2>
            <p className={styles.partialIntro}>{u.partialIntro}</p>
            <details className={styles.details} open>
              <summary>{u.extractedSummary}</summary>
              <pre className={styles.pre}>{result.extracted_text}</pre>
            </details>
          </section>
        ) : (
          <section className={styles.card}>
            <p className={styles.partialIntro}>{u.partialEmpty}</p>
          </section>
        )}
      </div>
    );
  }

  const analysis = result.analysis;
  const summary = analysis && typeof analysis.summary === "string" ? analysis.summary : null;
  const strengths = analysis?.strengths;
  const weaknesses = analysis?.weaknesses;

  const a = uk.analyzerForm;

  return (
    <div className={styles.results}>
      {showPdfDownload && onDownloadPdf ? (
        <div className={styles.resultsActions}>
          <button
            type="button"
            className={`${styles.btnBase} ${styles.btnSecondary}`}
            disabled={pdfLoading}
            onClick={onDownloadPdf}
            title={a.downloadPdfHelp}
          >
            {pdfLoading ? a.pdfLoading : a.downloadPdf}
          </button>
        </div>
      ) : null}

      {result.match_score != null ? (
        <section className={`${styles.card} ${styles.cardJobMatch}`}>
          <h2 className={styles.cardTitle}>{u.jobMatch}</h2>
          <JobMatchBlock
            score={result.match_score}
            breakdown={result.semantic_breakdown}
            weights={result.semantic_weights}
          />
        </section>
      ) : null}

      <AnalysisSummarySection
        summary={summary}
        strengths={strengths}
        weaknesses={weaknesses}
        recommendations={result.recommendations}
      />

      {(result.matched_competencies?.length ?? 0) > 0 || (result.missing_competencies?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>{u.competencies}</h2>
          <div className={styles.competenciesRow}>
            {(result.matched_competencies?.length ?? 0) > 0 ? (
              <div className={styles.competencyCol}>
                <p className={styles.label}>{u.aligned}</p>
                <ul className={styles.list}>
                  {result.matched_competencies!.map((x, i) => (
                    <li key={`m-${i}`}>{x}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {(result.missing_competencies?.length ?? 0) > 0 ? (
              <div className={styles.competencyCol}>
                <p className={styles.label}>{u.needsAttention}</p>
                <ul className={styles.list}>
                  {result.missing_competencies!.map((x, i) => (
                    <li key={`g-${i}`}>{x}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {(result.skills?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>{u.skillsFromResume}</h2>
          <p className={styles.reasoning}>{result.skills!.join(", ")}</p>
        </section>
      ) : null}

      {result.extracted_text ? (
        <details className={styles.details}>
          <summary>{u.extractedSummary}</summary>
          <pre className={styles.pre}>{result.extracted_text}</pre>
        </details>
      ) : null}
    </div>
  );
}
