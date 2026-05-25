from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class CVAnalysisRequest(BaseModel):
    job_description: Optional[str] = None
    analysis_type: str = Field(default="full")
    extract_keywords: bool = True


# Explicit schemas for OpenAPI docs and validation
class ExperienceItem(BaseModel):
    """Single work experience entry."""
    employer: Optional[str] = Field(None, description="Company or organization name")
    title: Optional[str] = Field(None, description="Job title or role")
    duration: Optional[str] = Field(None, description="Employment period (e.g. 2022-2024)")


class CertificateItem(BaseModel):
    """Single certificate or course entry."""
    name: Optional[str] = Field(None, description="Certificate or course name")
    institution: Optional[str] = Field(None, description="Issuer (e.g. Meta, Epam, Coursera)")
    year: Optional[str] = Field(None, description="Completion date or year")


class EducationItem(BaseModel):
    """Single education entry."""
    degree: Optional[str] = Field(None, description="Degree or qualification")
    institution: Optional[str] = Field(None, description="School or university")
    year: Optional[str] = Field(None, description="Graduation year or period")


class ProjectItem(BaseModel):
    """Single project entry (pet / GitHub projects)."""
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Short description")
    link: Optional[str] = Field(None, description="URL (e.g. GitHub)")


class ExplainabilityAttribution(BaseModel):
    feature: str = Field(..., description="Human-readable feature label (skill or experience line).")
    contribution: float = Field(
        ...,
        description="Estimated contribution to match_score (positive raises, negative lowers).",
    )


class ExplainabilityMethodResult(BaseModel):
    method: Literal["shap", "lime"]
    baseline_score: float = Field(..., ge=0, le=1, description="Score with all features masked out.")
    predicted_score: float = Field(..., ge=0, le=1, description="Score with the full CV features active.")
    top_positive: List[ExplainabilityAttribution] = Field(
        default_factory=list,
        description="Features that most increase match_score locally.",
    )
    top_negative: List[ExplainabilityAttribution] = Field(
        default_factory=list,
        description="Features that most decrease match_score when removed or perturbed.",
    )


class MatchExplainability(BaseModel):
    component_attributions: Optional[Dict[str, float]] = Field(
        None,
        description="Exact linear decomposition of match_score into skills, experience, and overall blocks.",
    )
    shap: Optional[ExplainabilityMethodResult] = Field(
        None,
        description="Kernel SHAP local attributions over CV skill/experience features.",
    )
    lime: Optional[ExplainabilityMethodResult] = Field(
        None,
        description="LIME local linear approximation over the same binary features.",
    )


class CVAnalysisResponse(BaseModel):
    success: bool
    extracted_text: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    skills: Optional[List[str]] = None
    experience: Optional[List[ExperienceItem]] = None
    certificates: Optional[List[CertificateItem]] = None
    education: Optional[List[EducationItem]] = None
    projects: Optional[List[ProjectItem]] = None
    match_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="How well the CV matches the position requirements (0-1). Only present when job description/position URL was provided.",
    )
    match_score_reasoning: Optional[str] = Field(
        None,
        description="Human-readable score rationale, including semantic breakdown when Sentence Transformers are used.",
    )
    recommendations: Optional[List[str]] = Field(
        None,
        description="Never empty on success: actionable CV/candidacy improvements. Not job marketing or role-benefit copy.",
    )
    matched_competencies: Optional[List[str]] = Field(
        None,
        description="Job requirements clearly supported by the CV. Empty when no job was provided.",
    )
    missing_competencies: Optional[List[str]] = Field(
        None,
        description="Job requirements weak or missing in the CV. Empty when no job was provided.",
    )
    # Semantic matching breakdown (when use_semantic_matching is True and job description provided)
    semantic_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Per-component similarities: skills_similarity, experience_similarity, overall_similarity. Present when semantic matching is used.",
    )
    semantic_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Normalized weights used to combine the three similarities into match_score: keys skills, experience, overall (maps to overall_similarity / full-CV-vs-job).",
    )
    semantic_score_narrative: Optional[str] = Field(
        None,
        description="LLM-written interpretation of the semantic scores in context (when use_llm_semantic_narrative is enabled).",
    )
    match_explainability: Optional[MatchExplainability] = Field(
        None,
        description="SHAP/LIME attributions for match_score when semantic matching is used.",
    )
    error: Optional[str] = None
