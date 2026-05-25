"""SHAP and LIME local explanations for embedding-based match_score."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings
from app.models.cv_models import (
    ExperienceItem,
    ExplainabilityAttribution,
    ExplainabilityMethodResult,
    MatchExplainability,
)
from app.services.semantic_matcher import (
    SemanticMatchResult,
    embed_text,
    embed_texts_batch,
    normalized_semantic_weights,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ExplainFeatures:
    names: Tuple[str, ...]
    skills: Tuple[str, ...]
    experience_lines: Tuple[str, ...]


def _experience_lines(
    experience: Optional[Sequence[ExperienceItem]],
    fallback_text: str,
) -> List[str]:
    lines: List[str] = []
    if experience:
        for item in experience:
            parts = [p for p in (item.title, item.employer, item.duration) if p and str(p).strip()]
            if parts:
                lines.append(" — ".join(str(p).strip() for p in parts))
    if not lines and fallback_text.strip():
        for chunk in fallback_text.replace("•", "\n").split("\n"):
            c = chunk.strip()
            if c:
                lines.append(c[:400])
    return lines


def _build_features(
    skills: List[str],
    experience: Optional[Sequence[ExperienceItem]],
    cv_experience_text: str,
) -> _ExplainFeatures:
    skill_feats = [s.strip() for s in skills if s and s.strip()][: settings.explainer_max_skills]
    exp_lines = _experience_lines(experience, cv_experience_text)[: settings.explainer_max_experience]
    names: List[str] = [f"Навичка: {s}" for s in skill_feats]
    names.extend(f"Досвід: {line}" for line in exp_lines)
    return _ExplainFeatures(names=tuple(names), skills=tuple(skill_feats), experience_lines=tuple(exp_lines))


def _block(text: str, max_chars: int) -> str:
    if not text or not text.strip():
        return ""
    t = text.strip()
    return t[:max_chars] if len(t) > max_chars else t


@dataclass
class _SemanticScoreCache:
    """Pre-embed fixed CV/job blocks; only re-embed masked skills/experience per SHAP/LIME sample."""

    w_skills: float
    w_exp: float
    w_overall: float
    job_req_vec: Optional[List[float]]
    job_full_vec: Optional[List[float]]
    full_cv_vec: Optional[List[float]]

    def score(self, active_skills: Sequence[str], active_exp: Sequence[str]) -> float:
        cv_skills_block = " ".join(s for s in active_skills if s and str(s).strip())
        cv_exp_block = _block("\n".join(active_exp), 8000)

        skills_sim = 0.0
        if cv_skills_block and self.job_req_vec is not None:
            skills_cv_vec = embed_text(cv_skills_block)
            skills_sim = _cosine(skills_cv_vec, self.job_req_vec)
        elif self.job_req_vec is None:
            skills_sim = 1.0

        exp_sim = 0.0
        if cv_exp_block and self.job_req_vec is not None:
            exp_cv_vec = embed_text(cv_exp_block)
            exp_sim = _cosine(exp_cv_vec, self.job_req_vec)
        elif self.job_req_vec is None:
            exp_sim = 1.0

        overall_sim = 0.0
        if self.full_cv_vec is not None and self.job_full_vec is not None:
            overall_sim = _cosine(self.full_cv_vec, self.job_full_vec)
        elif self.job_full_vec is None:
            overall_sim = 1.0

        raw = self.w_skills * skills_sim + self.w_exp * exp_sim + self.w_overall * overall_sim
        return max(0.0, min(1.0, raw))


def _cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos = np.dot(va, vb) / (norm_a * norm_b)
    return float((cos + 1) / 2)


def _make_cached_predictor(
    features: _ExplainFeatures,
    cv_full_text: str,
    job_requirements_text: str,
    job_full_text: str,
):
    job_req_block = _block(job_requirements_text, 8000)
    job_full_block = _block(job_full_text, 12000)
    cv_full_block = _block(cv_full_text, 12000)
    ws, we, wo = normalized_semantic_weights()
    job_req_vec, job_full_vec, full_cv_vec = embed_texts_batch([job_req_block, job_full_block, cv_full_block])
    cache = _SemanticScoreCache(
        w_skills=ws,
        w_exp=we,
        w_overall=wo,
        job_req_vec=job_req_vec if job_req_block else None,
        job_full_vec=job_full_vec if job_full_block else None,
        full_cv_vec=full_cv_vec if cv_full_block else None,
    )

    n_skills = len(features.skills)

    def predict(masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, dtype=float))
        scores = np.zeros(masks.shape[0], dtype=float)
        for i, row in enumerate(masks):
            active_skills = [s for s, on in zip(features.skills, row[:n_skills]) if on >= 0.5]
            active_exp = [e for e, on in zip(features.experience_lines, row[n_skills:]) if on >= 0.5]
            scores[i] = cache.score(active_skills, active_exp)
        return scores

    return predict, cache


def component_attributions(sem: SemanticMatchResult) -> Dict[str, float]:
    ws, we, wo = normalized_semantic_weights()
    return {
        "skills": round(ws * sem.skills_similarity, 6),
        "experience": round(we * sem.experience_similarity, 6),
        "overall": round(wo * sem.overall_similarity, 6),
    }


def _split_attributions(
    values: np.ndarray,
    names: Sequence[str],
    *,
    top_k: int,
) -> Tuple[List[ExplainabilityAttribution], List[ExplainabilityAttribution]]:
    pairs = [(names[i], float(values[i])) for i in range(len(names))]
    positive = sorted((p for p in pairs if p[1] > 0), key=lambda x: x[1], reverse=True)[:top_k]
    negative = sorted((p for p in pairs if p[1] < 0), key=lambda x: x[1])[:top_k]
    return (
        [ExplainabilityAttribution(feature=f, contribution=c) for f, c in positive],
        [ExplainabilityAttribution(feature=f, contribution=c) for f, c in negative],
    )


def _run_shap(
    predict,
    features: _ExplainFeatures,
    baseline_score: float,
    predicted_score: float,
) -> Optional[ExplainabilityMethodResult]:
    try:
        import shap
    except ImportError:
        logger.warning("shap package not installed; skipping SHAP explainability")
        return None

    n = len(features.names)
    if n == 0:
        return None

    instance = np.ones(n, dtype=float)
    background = np.zeros((1, n), dtype=float)
    try:
        explainer = shap.KernelExplainer(predict, background, link="identity")
        raw = explainer.shap_values(instance.reshape(1, -1), nsamples=settings.explainer_shap_samples)
        values = np.asarray(raw, dtype=float).reshape(-1)[:n]
    except Exception:
        logger.warning("SHAP KernelExplainer failed", exc_info=True)
        return None

    top_pos, top_neg = _split_attributions(values, features.names, top_k=settings.explainer_top_features)
    return ExplainabilityMethodResult(
        method="shap",
        baseline_score=round(baseline_score, 6),
        predicted_score=round(predicted_score, 6),
        top_positive=top_pos,
        top_negative=top_neg,
    )


def _run_lime(
    predict,
    features: _ExplainFeatures,
    baseline_score: float,
    predicted_score: float,
) -> Optional[ExplainabilityMethodResult]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        logger.warning("lime package not installed; skipping LIME explainability")
        return None

    n = len(features.names)
    if n == 0:
        return None

    rng = np.random.default_rng(42)
    training = rng.integers(0, 2, size=(max(32, settings.explainer_lime_samples), n)).astype(float)
    instance = np.ones(n, dtype=float)

    try:
        explainer = LimeTabularExplainer(
            training_data=training,
            feature_names=list(features.names),
            mode="regression",
            discretize_continuous=False,
        )
        explanation = explainer.explain_instance(
            instance,
            predict,
            num_features=min(n, settings.explainer_top_features),
            num_samples=settings.explainer_lime_samples,
        )
        values = np.zeros(n, dtype=float)
        for idx, weight in explanation.as_map().get(1, []):
            values[idx] = float(weight)
    except Exception:
        logger.warning("LIME TabularExplainer failed", exc_info=True)
        return None

    top_pos, top_neg = _split_attributions(values, features.names, top_k=settings.explainer_top_features)
    return ExplainabilityMethodResult(
        method="lime",
        baseline_score=round(baseline_score, 6),
        predicted_score=round(predicted_score, 6),
        top_positive=top_pos,
        top_negative=top_neg,
    )


def explain_match_score(
    *,
    sem: SemanticMatchResult,
    skills: List[str],
    experience: Optional[Sequence[ExperienceItem]],
    cv_experience_text: str,
    cv_full_text: str,
    job_requirements_text: str,
    job_full_text: str,
) -> Optional[MatchExplainability]:
    if not settings.use_match_explainers:
        return None

    features = _build_features(skills, experience, cv_experience_text)
    comps = component_attributions(sem)

    if not features.names:
        return MatchExplainability(
            component_attributions=comps,
            shap=None,
            lime=None,
        )

    predict, cache = _make_cached_predictor(
        features, cv_full_text, job_requirements_text, job_full_text
    )
    n = len(features.names)
    baseline_score = float(predict(np.zeros(n))[0])
    predicted_score = float(predict(np.ones(n))[0])
    _ = cache  # cache warms job/full vectors

    shap_result: Optional[ExplainabilityMethodResult] = None
    lime_result: Optional[ExplainabilityMethodResult] = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        shap_future = pool.submit(_run_shap, predict, features, baseline_score, predicted_score)
        lime_future = pool.submit(_run_lime, predict, features, baseline_score, predicted_score)
        shap_result = shap_future.result()
        lime_result = lime_future.result()

    if shap_result is None and lime_result is None:
        return MatchExplainability(
            component_attributions=comps,
            shap=None,
            lime=None,
        )

    return MatchExplainability(
        component_attributions=comps,
        shap=shap_result,
        lime=lime_result,
    )
