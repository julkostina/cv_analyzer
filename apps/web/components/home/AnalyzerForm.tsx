import { uk } from "../../lib/strings-uk";
import styles from "./HomeAnalyzer.module.css";

type AnalyzerFormProps = {
  onFileChange: (file: File | null) => void;
  jobDescription: string;
  onJobDescriptionChange: (value: string) => void;
  jobUrl: string;
  onJobUrlChange: (value: string) => void;
  loading: boolean;
  onSubmit: (e: React.FormEvent) => void;
};

export function AnalyzerForm({
  onFileChange,
  jobDescription,
  onJobDescriptionChange,
  jobUrl,
  onJobUrlChange,
  loading,
  onSubmit,
}: AnalyzerFormProps) {
  const a = uk.analyzerForm;
  return (
    <form className={styles.steps} onSubmit={onSubmit}>
      <section className={styles.card} aria-labelledby="resume-heading">
        <h2 id="resume-heading" className={styles.cardTitle}>
          {a.resumeSection}
        </h2>
        <div className={styles.resumeRow}>
          <div className={`${styles.field} ${styles.resumeFileField}`}>
            <label className={styles.label} htmlFor="cv-file">
              {a.fileLabel}
            </label>
            <input
              id="cv-file"
              className={styles.input}
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(ev) => {
                const f = ev.target.files?.[0];
                onFileChange(f ?? null);
              }}
            />
          </div>
          <div className={`${styles.field} ${styles.resumeUrlField}`}>
            <label className={styles.label} htmlFor="job-url">
              {a.jobUrlLabel}
            </label>
            <input
              id="job-url"
              className={styles.input}
              type="url"
              inputMode="url"
              value={jobUrl}
              onChange={(ev) => onJobUrlChange(ev.target.value)}
              placeholder={a.jobUrlPlaceholder}
              autoComplete="off"
            />
          </div>
        </div>
      </section>

      <section className={styles.card} aria-labelledby="job-heading">
        <h2 id="job-heading" className={styles.cardTitle}>
          {a.jobSection}
        </h2>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="job-text">
            {a.jobTextLabel}
          </label>
          <textarea
            id="job-text"
            className={styles.textarea}
            value={jobDescription}
            onChange={(ev) => onJobDescriptionChange(ev.target.value)}
            placeholder={a.jobTextPlaceholder}
            rows={6}
          />
        </div>
      </section>

      <section className={styles.card} aria-label={a.analyze}>
        <div className={styles.actions}>
          <button type="submit" className={`${styles.btnBase} ${styles.btnPrimary}`} disabled={loading}>
            {loading ? a.analyzing : a.analyze}
          </button>
        </div>
      </section>
    </form>
  );
}
