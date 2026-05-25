type ExperienceItem = {
  employer?: string | null;
  title?: string | null;
  duration?: string | null;
};

type CertificateItem = {
  name?: string | null;
  institution?: string | null;
  year?: string | null;
};

type EducationItem = {
  degree?: string | null;
  institution?: string | null;
  year?: string | null;
};

type ProjectItem = {
  name?: string | null;
  description?: string | null;
  link?: string | null;
};

type ExplainabilityAttribution = {
  feature: string;
  contribution: number;
};

type ExplainabilityMethodResult = {
  method: "shap" | "lime";
  baseline_score: number;
  predicted_score: number;
  top_positive: ExplainabilityAttribution[];
  top_negative: ExplainabilityAttribution[];
};

type MatchExplainability = {
  component_attributions?: Record<string, number> | null;
  shap?: ExplainabilityMethodResult | null;
  lime?: ExplainabilityMethodResult | null;
};

export type CVAnalysisResponse = {
  success: boolean;
  extracted_text?: string | null;
  analysis?: Record<string, unknown> | null;
  skills?: string[] | null;
  experience?: ExperienceItem[] | null;
  certificates?: CertificateItem[] | null;
  education?: EducationItem[] | null;
  projects?: ProjectItem[] | null;
  match_score?: number | null;
  match_score_reasoning?: string | null;
  recommendations?: string[] | null;
  matched_competencies?: string[] | null;
  missing_competencies?: string[] | null;
  semantic_breakdown?: Record<string, number> | null;
  semantic_weights?: Record<string, number> | null;
  semantic_score_narrative?: string | null;
  match_explainability?: MatchExplainability | null;
  error?: string | null;
};
