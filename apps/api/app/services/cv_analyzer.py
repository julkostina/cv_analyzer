import asyncio
import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.models import (
    CVAnalysisRequest,
    CVAnalysisResponse,
    CertificateItem,
    EducationItem,
    ExperienceItem,
    ProjectItem,
)
from app.services.semantic_matcher import (
    SEMANTIC_METRIC_GUIDES,
    SemanticMatchResult,
    compute_semantic_match,
    normalized_semantic_weights,
)
from app.services.match_explainer import explain_match_score
from app.utils.text_preprocess import normalize_text_for_pipeline

logger = logging.getLogger(__name__)

_HUMAN_PROMPT = (
    "Поверни ПОВНИЙ JSON з ключами (англійською, як у схемі): skills, experience, certificates, education, projects, "
    "analysis, matched_competencies, missing_competencies, match_score (null), "
    "match_score_reasoning (null), recommendations. Для порожніх списків використовуй [].\n\nРезюме:\n{cv_text}\n{job_description_section}"
)

_CV_ANALYZER_SYSTEM = """Ти — ШІ-аналітик резюме.

Завдання: витягни структуровані дані з тексту резюме й щоразу поверни ПОВНИЙ JSON-об’єкт.

КРИТИЧНО: увесь текст для користувача (підсумки, рекомендації, рядки навичок, matched_competencies, missing_competencies, поля роботодавця/посади/назв тощо) — українською мовою. Назви технологій (React, TypeScript) залишай латиницею.
Тон summary: коуч → кандидат на «ви». Так: «У вас сильний досвід з React…», «Вам варто…». Ні: «Я маю…», «мені потрібно…», «вакансія вимагає від мене…».

Обов’язкові масиви: skills, experience, certificates, education, projects — якщо порожньо, використовуй [], не null.

analysis — об’єкт із ключами summary, strengths, weaknesses (українською).
- summary — чернетка підсумку для кандидата (її може доповнити сервер після підрахунку балу). Мінімум 8 речень на «ви»; без «я»/«мені». Серверна версія матиме ≥10 речень і пояснення балу.
  Якщо пишеш фінальний summary самостійно (без серверного доповнення): 2–4 абзаци, ≥10 речень, природна проза на «ви», як лист від кар’єрного коуча.
  ЗАБОРОНЕНО писати від імені кандидата: не «я», «мені», «моя», «мій», «мене», «я вже працював», «вакансія вимагає від мене». Описуй резюме в третій особі або на «ви», ніколи в першій особі однини.
  ЗАБОРОНЕНО: заголовки розділів, підзаголовки, мітки на кшталт «Сильні сторони:», «Очікування роботодавця:», «Прогалини:», «Висновок:», «Слабкі сторони:», нумеровані списки в тексті summary.
  ЗАБОРОНЕНО обмежитись лише похвалою — обов’язково вплети очікування ролі, прогалини та поради в той самий текст.
  Якщо є вакансія — у summary плавно (через «водночас», «з іншого боку», «проте», «також», «щоб підсилити відгук») переходь від:
  • того, що вже сильно в резюме (компанії, стек, проєкти);
  • того, чого шукає роботодавець (стаж, технології, командна робота тощо);
  • де є розбіжності або слабко підкреслено в CV (узгодь із matched_competencies / missing_competencies у JSON);
  • 2–4 конкретних кроків, що змінити — без окремого заголовка «Висновок».
  Якщо вакансії немає — 2–3 абзаци про сильні сторони, зони росту й поради, теж без міток розділів.
  Увесь зміст recommendations має бути вже в summary; не дублюй окремими списками.
- strengths і weaknesses — порожній рядок "" або [] (усі змістові фрази лише в summary).

recommendations — масив із 1–2 коротких речень: лише додаткові уточнення, яких немає в summary (якщо summary повний — один елемент «Див. підсумок вище.»). Не ключові слова. Той самий теплий тон на «ви».

Якщо є опис вакансії:
- matched_competencies — вимоги/навички з оголошення, які чітко підтверджує резюме.
- missing_competencies — вимоги, які у резюме слабкі або відсутні.
- match_score і match_score_reasoning завжди null (сервер заповнить їх семантичною оцінкою).

Якщо опису вакансії немає:
- matched_competencies: [], missing_competencies: [].
- match_score і match_score_reasoning: null.
- recommendations — загальні ринкові поради щодо покращення резюме (ті самі правила: лише дії для кандидата).

Класифікація:
- EXPERIENCE — роботи, стажування, фриланс з обов’язками. Не курси.
- CERTIFICATES — курси, тренінги (Meta, Coursera, Epam, AWS) без робочих обов’язків.
- EDUCATION — університети, ступені.
- PROJECTS — пет-проєкти, GitHub без формального найму.

Порядок тексту в PDF може бути зламаним — класифікуй за змістом.

Правила: Meta / Course / Program без робочих задач → CERTIFICATE. Слова «курс», «навчання», «програма» в навчальному контексті → CERTIFICATE. Описані робочі задачі → EXPERIENCE. Лише GitHub без компанії-наймача → PROJECT.
"""

def _ollama_json_prefix() -> str:
    summary_rule = (
        "analysis.summary — коротка чернетка 3–5 речень на «ви» (НЕ «я»/«мені»); сервер допише фінальний підсумок з балами. "
        if settings.use_llm_coaching_summary
        else "analysis.summary — 2–4 абзаци на «ви» (НЕ «я»/«мені»); без підписів розділів; коуч звертається до кандидата. "
    )
    return (
        "Поверни рівно ОДИН валідний JSON-об’єкт (без markdown, без тексту до чи після). "
        "Обов’язкові ключі англійською: skills, experience, certificates, education, projects, analysis, "
        "matched_competencies, missing_competencies, match_score, match_score_reasoning, recommendations. "
        "Заповни skills і experience з тексту резюме (не залишай усі масиви порожніми, якщо дані є в тексті). "
        f"{summary_rule}"
        "strengths/weaknesses — []. recommendations — 1–2 речення або «Див. підсумок вище.».\n\n"
    )


def _should_retry_ollama_for_summary(summary: Any, job_description_section: str) -> bool:
    """Skip extra Ollama passes when the server will rewrite summary after scoring."""
    if settings.use_llm_coaching_summary and "Опис вакансії (вимоги):" in job_description_section:
        return False
    return _summary_needs_expansion(summary, job_description_section)


class CVAnalysisOutput(BaseModel):
    skills: Optional[List[str]] = Field(None)
    experience: Optional[List[ExperienceItem]] = Field(None)
    certificates: Optional[List[CertificateItem]] = Field(None)
    education: Optional[List[EducationItem]] = Field(None)
    projects: Optional[List[ProjectItem]] = Field(None)
    analysis: Optional[Dict[str, Any]] = Field(None)
    match_score: Optional[float] = Field(None, ge=0, le=1)
    match_score_reasoning: Optional[str] = Field(None)
    recommendations: List[str] = Field(
        ...,
        min_length=1,
        description="Concrete improvements to the CV or candidacy; never job-pitch or 'what you will gain' role copy.",
    )
    matched_competencies: Optional[List[str]] = Field(None)
    missing_competencies: Optional[List[str]] = Field(None)


class MatchScoreFallbackOutput(BaseModel):
    """LLM-only estimate when embedding-based semantic scoring is unavailable."""

    match_score: float = Field(..., ge=0, le=1, description="Оцінка відповідності 0..1")
    match_score_reasoning: str = Field(
        ...,
        min_length=10,
        description="Обґрунтування українською; згадай, що це резервна оцінка без деталізації за векторами.",
    )


_COACHING_SECTION_LABEL_RE = re.compile(
    r"(?:^|[\n.]\s*)(?:Сильні сторони|Очікування роботодавця|Відповідність і прогалини|Прогалини|"
    r"Слабкі сторони|Висновок(?:\s+і\s+поради)?|Висновки)\s*:\s*",
    re.IGNORECASE,
)


def normalize_coaching_summary(text: str) -> str:
    """Turn labelled sections into flowing paragraphs (fallback if the model ignores prompt)."""
    t = (text or "").strip()
    if not t:
        return t
    if t.startswith("{") and "summary" in t[:80]:
        parsed = _parse_llm_json_blob(t)
        if isinstance(parsed, dict):
            inner = parsed.get("summary")
            if isinstance(inner, str) and inner.strip():
                t = inner.strip()
    if not t or not _COACHING_SECTION_LABEL_RE.search(t):
        return t
    t = _COACHING_SECTION_LABEL_RE.sub(". ", t)
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(r"\s+", " ", t).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sentences) <= 5:
        return t
    per = max(2, (len(sentences) + 2) // 3)
    paras = [" ".join(sentences[i : i + per]) for i in range(0, len(sentences), per)]
    return "\n\n".join(paras)


def _parse_llm_json_blob(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            text = " — ".join(
                str(item.get(k, "")).strip()
                for k in ("title", "name", "employer", "company", "institution", "degree")
                if item.get(k)
            )
            if text.strip():
                out.append(text.strip())
    return out


def _coerce_experience_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append({"title": item.strip(), "employer": None, "duration": None})
        elif isinstance(item, dict):
            out.append(
                {
                    "employer": item.get("employer") or item.get("company") or item.get("organization"),
                    "title": item.get("title") or item.get("role") or item.get("position"),
                    "duration": item.get("duration") or item.get("period") or item.get("dates"),
                }
            )
    return out


def _coerce_named_items(
    value: Any,
    *,
    name_keys: Tuple[str, ...],
    inst_keys: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append({"name": item.strip(), "institution": None, "year": None, "degree": item.strip()})
        elif isinstance(item, dict):
            name = next((item.get(k) for k in name_keys if item.get(k)), None)
            inst = next((item.get(k) for k in inst_keys if item.get(k)), None)
            row: Dict[str, Any] = {
                "name": name,
                "institution": inst,
                "year": item.get("year") or item.get("date") or item.get("period"),
            }
            if "degree" in item or "degree" in name_keys:
                row["degree"] = item.get("degree") or name
            if "description" in item:
                row["description"] = item.get("description")
            if "link" in item:
                row["link"] = item.get("link") or item.get("url")
            out.append(row)
    return out


def _normalize_parsed_cv_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map common alternate keys / shapes from local LLMs before Pydantic validation."""
    data = dict(raw)
    aliases = {
        "work_experience": "experience",
        "experiences": "experience",
        "certifications": "certificates",
        "educations": "education",
        "project_list": "projects",
        "matched": "matched_competencies",
        "missing": "missing_competencies",
    }
    for src, dst in aliases.items():
        if dst not in data or data.get(dst) in (None, [], {}):
            if src in data and data[src] not in (None, [], {}):
                data[dst] = data[src]

    analysis = data.get("analysis")
    if isinstance(analysis, str) and analysis.strip():
        data["analysis"] = {
            "summary": normalize_coaching_summary(analysis.strip()),
            "strengths": [],
            "weaknesses": [],
        }
    elif not isinstance(analysis, dict):
        data["analysis"] = {}
    elif isinstance(analysis.get("summary"), str):
        analysis = dict(analysis)
        analysis["summary"] = normalize_coaching_summary(analysis["summary"])
        data["analysis"] = analysis

    data["skills"] = _coerce_str_list(data.get("skills"))
    data["experience"] = _coerce_experience_list(data.get("experience"))
    data["certificates"] = _coerce_named_items(
        data.get("certificates"),
        name_keys=("name", "title", "certificate"),
        inst_keys=("institution", "issuer", "provider"),
    )
    edu_raw = data.get("education")
    edu_out: List[Dict[str, Any]] = []
    if isinstance(edu_raw, list):
        for item in edu_raw:
            if isinstance(item, str) and item.strip():
                edu_out.append({"degree": item.strip(), "institution": None, "year": None})
            elif isinstance(item, dict):
                edu_out.append(
                    {
                        "degree": item.get("degree") or item.get("name") or item.get("title"),
                        "institution": item.get("institution")
                        or item.get("school")
                        or item.get("university"),
                        "year": item.get("year") or item.get("date") or item.get("period"),
                    }
                )
    data["education"] = edu_out
    data["projects"] = _coerce_named_items(
        data.get("projects"),
        name_keys=("name", "title", "project"),
        inst_keys=("institution",),
    )
    data["matched_competencies"] = _coerce_str_list(data.get("matched_competencies"))
    data["missing_competencies"] = _coerce_str_list(data.get("missing_competencies"))
    return data


def _cv_output_from_parsed_dict(raw: Dict[str, Any]) -> Optional[CVAnalysisOutput]:
    data = _normalize_parsed_cv_dict(raw)
    recs = data.get("recommendations")
    if not recs or not isinstance(recs, list) or len(recs) == 0:
        data["recommendations"] = ["Перегляньте резюме та спробуйте аналіз ще раз."]
    elif isinstance(recs, list):
        data["recommendations"] = [str(r).strip() for r in recs if r and str(r).strip()]
        if not data["recommendations"]:
            data["recommendations"] = ["Перегляньте резюме та спробуйте аналіз ще раз."]
    try:
        return CVAnalysisOutput.model_validate(data)
    except Exception as e:
        logger.warning("CV JSON validation failed: %s", e)
        return None


def _normalize_llm_result(result: CVAnalysisOutput) -> tuple[CVAnalysisOutput, bool]:
    incomplete = False
    if result.skills is None:
        result = result.model_copy(update={"skills": []})
        incomplete = True
    if result.experience is None:
        result = result.model_copy(update={"experience": []})
        incomplete = True
    if result.certificates is None:
        result = result.model_copy(update={"certificates": []})
        incomplete = True
    if result.education is None:
        result = result.model_copy(update={"education": []})
        incomplete = True
    if result.projects is None:
        result = result.model_copy(update={"projects": []})
        incomplete = True
    if result.analysis is None:
        result = result.model_copy(update={"analysis": {}})
        incomplete = True
    if result.matched_competencies is None:
        result = result.model_copy(update={"matched_competencies": []})
        incomplete = True
    if result.missing_competencies is None:
        result = result.model_copy(update={"missing_competencies": []})
        incomplete = True
    return result, incomplete


def _is_extraction_empty(result: CVAnalysisOutput) -> bool:
    return (
        (not result.skills or len(result.skills) == 0)
        and (not result.experience or len(result.experience) == 0)
        and (not result.education or len(result.education) == 0)
        and (not result.certificates or len(result.certificates) == 0)
        and (not result.projects or len(result.projects) == 0)
        and (not result.analysis or result.analysis == {})
    )


def _ollama_empty_extraction_hint(model_name: str) -> str:
    """Hints that do not suggest the same model tag the user already runs."""
    m = model_name.lower()
    parts: List[str] = []
    small = any(tok in m for tok in ("3.2", ":3b", "1b", "1.5b", ":2b", "2.3b")) and "8b" not in m and "70b" not in m
    if small:
        parts.append("Малі моделі часто не справляються зі складними JSON-схемами.")
        parts.append("Спробуйте OLLAMA_MODEL=llama3:8b або llama3.1:8b (ollama pull …).")
    elif "llama3.1" in m and "8b" in m:
        parts.append(
            "Локальна модель не заповнила JSON. Спробуйте mistral або qwen2.5:7b (ollama pull), перезапустіть API; "
            "або ENVIRONMENT=production з GEMINI_API_KEY."
        )
    elif "llama3" in m and "8b" in m:
        parts.append("llama3:8b часто дає порожній JSON — краще OLLAMA_MODEL=llama3.1:8b.")
        parts.append(
            "Або mistral / qwen2.5; оновіть Ollama; або ENVIRONMENT=production з GEMINI_API_KEY."
        )
    else:
        parts.append("Спробуйте іншу модель (mistral, qwen2.5, llama3.1:8b) або Gemini у production.")
    parts.append(
        f"За потреби збільште OLLAMA_NUM_CTX (зараз {settings.ollama_num_ctx}). Тримайте MAX_CV_CHARS_FOR_LLM=0, щоб не обрізати резюме."
    )
    return " ".join(parts)


def _empty_extraction_error(
    backend: str,
    model_name: str,
    cv_chars_sent: int,
    cv_chars_total: int,
) -> str:
    if cv_chars_sent < cv_chars_total:
        cv_detail = f"До моделі надіслано {cv_chars_sent} символів резюме (обрізано з {cv_chars_total})."
    else:
        cv_detail = f"Довжина резюме: {cv_chars_total} символів."
    base = (
        "Аналіз не вдався: модель не повернула структуровані дані. "
        "Усі обов’язкові розділи (досвід, освіта, сертифікати, проєкти, навички) порожні. "
        f"Бекенд: {backend}, модель: {model_name}. {cv_detail} "
    )
    if backend == "Ollama":
        return base + _ollama_empty_extraction_hint(model_name)
    return base + "Спробуйте коротше резюме або перевірте налаштування моделі."


def _experience_text_from_result(result: CVAnalysisOutput) -> str:
    parts: List[str] = []
    for e in result.experience or []:
        bits = [x for x in (e.title, e.employer, e.duration) if x]
        if bits:
            parts.append(" — ".join(bits))
    return "\n".join(parts)


def _semantic_reasoning_text(sem: SemanticMatchResult) -> str:
    return (
        "Бал відповідності з косинусної схожості векторних подань резюме та вакансії "
        "(багатомовна модель Sentence Transformers).\n\n"
        f"• Схожість блоку навичок: {sem.skills_similarity:.1%}\n"
        f"• Схожість досвіду до вимог: {sem.experience_similarity:.1%}\n"
        f"• Схожість повного резюме до повного тексту вакансії: {sem.overall_similarity:.1%}\n\n"
        f"Зважений загальний бал: {sem.score:.1%}."
    )


_COACHING_SUMMARY_SYSTEM = """Ти — кар’єрний коуч. Напиши фінальний блок «Підсумок та поради» для кандидата українською.

Обов’язково:
- Мінімум 10 повних речень, 2–4 абзаци зв’язної прози (без маркерів «Сильні сторони:», списків).
- Звертайся на «ви» («у вас», «вам», «ви можете»). ЗАБОРОНЕНО «я», «мені», «моя робота», «від мене».
- Якщо є дані про бал: поясни простими словами, що саме підвищує загальний бал відповідності (навички/досвід/збіги з вакансією — з цифр і SHAP/LIME «підвищують»).
- Так само — що знижує бал (прогалини, слабкі блоки, «знижують» з SHAP/LIME, missing_competencies).
- Числа match_score і подібностей ОСТАТОЧНІ — не вигадуй інших відсотків.
- Вплети сильні сторони резюме, очікування роботодавця, поради; тон підтримливий.
- Поверни ЛИШЕ текст summary (без JSON, без markdown-заголовків)."""

_SEMANTIC_NARRATIVE_SYSTEM = """Ти — кар’єрний коуч, який допомагає кандидату зрозуміти, як його резюме лягає на текст вакансії.

Правила:
- Числові бали, які ти отримуєш, ОСТАТОЧНІ (вже пораховані з ембеддингів). Не змінюй їх, не перераховуй і не вигадуй інші числа.
- Поясни простою українською, що ці бали ймовірно означають саме для ЦІЄЇ людини, спираючись на наведені уривки.
- Будь стислим, підтримливим і конкретним до уривків (список навичок, фрагмент резюме, фрагмент вакансії). Якщо список навичок порожній або крихітний, скажи, що це може занизити бал навичок, не означаючи відсутності навичок узагалі.
- Звертайся на «ви». Лише українська мова. Без markdown-заголовків; короткі абзаци або марковані рядки — гаразд.
- Не більше 280 слів."""

_FALLBACK_MATCH_SCORE_SYSTEM = """Ти оцінюєш відповідність резюме вакансії, коли основний серверний конвейєр векторних подань (ембеддинги) тимчасово недоступний або дав збій.

Поверни:
- match_score — дійсне число від 0 до 1 (чим ближче до 1, тим краща відповідність за твоїм судженням).
- match_score_reasoning — 2–6 речень українською; обов’язково згадай, що це РЕЗЕРВНА оцінка без розбиття на підпоказники, бо вбудований семантичний підрахунок не спрацював.

Оцінюй лише за тим, що видно в наданих уривках резюме та вакансії; не вигадуй фактів, яких немає в тексті. Будь обережним і чесним щодо невизначеності."""


def _llm_message_text(msg: Any) -> str:
    raw = getattr(msg, "content", msg)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        pieces: List[str] = []
        for block in raw:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                pieces.append(str(block.get("text", block)))
            else:
                pieces.append(str(block))
        return "".join(pieces)
    return str(raw or "")


def _build_success_response(
    cv_text: str,
    result: CVAnalysisOutput,
    job_description: Optional[str],
) -> Tuple[CVAnalysisResponse, bool, Optional[SemanticMatchResult]]:
    recs = result.recommendations or ["Перегляньте резюме та спробуйте ще раз."]
    matched = list(result.matched_competencies or [])
    missing = list(result.missing_competencies or [])

    match_score: Optional[float] = None
    match_reason: Optional[str] = None
    semantic_breakdown: Optional[Dict[str, float]] = None
    semantic_weights: Optional[Dict[str, float]] = None
    semantic_metric_guides: Optional[Dict[str, str]] = None
    semantic_pipeline_failed = False
    match_explainability = None
    sem_result: Optional[SemanticMatchResult] = None

    job_stripped = (job_description or "").strip()
    if job_stripped and settings.use_semantic_matching:
        try:
            cv_exp = _experience_text_from_result(result)
            sem = compute_semantic_match(
                cv_skills=result.skills or [],
                cv_experience_text=cv_exp or cv_text[:8000],
                cv_full_text=cv_text,
                job_requirements_text=job_stripped,
                job_full_text=job_stripped,
            )
            sem_result = sem
            match_score = sem.score
            semantic_breakdown = {
                "skills_similarity": sem.skills_similarity,
                "experience_similarity": sem.experience_similarity,
                "overall_similarity": sem.overall_similarity,
            }
            ws, we, wo = normalized_semantic_weights()
            semantic_weights = {"skills": ws, "experience": we, "overall": wo}
            semantic_metric_guides = dict(SEMANTIC_METRIC_GUIDES)
            match_reason = _semantic_reasoning_text(sem)
            logger.info(
                "Semantic match score=%.3f (skills=%.3f exp=%.3f overall=%.3f)",
                sem.score,
                sem.skills_similarity,
                sem.experience_similarity,
                sem.overall_similarity,
            )
        except Exception:
            logger.exception("Semantic matching failed; will try LLM fallback for match_score")
            semantic_pipeline_failed = True
            match_score = result.match_score
            match_reason = result.match_score_reasoning

    resp = CVAnalysisResponse(
        success=True,
        extracted_text=cv_text,
        analysis=result.analysis,
        skills=result.skills,
        experience=result.experience,
        certificates=result.certificates,
        education=result.education,
        projects=result.projects,
        match_score=match_score,
        match_score_reasoning=match_reason,
        recommendations=recs,
        matched_competencies=matched if job_stripped else [],
        missing_competencies=missing if job_stripped else [],
        semantic_breakdown=semantic_breakdown,
        semantic_weights=semantic_weights,
        semantic_metric_guides=semantic_metric_guides,
        semantic_score_narrative=None,
        match_explainability=match_explainability,
        error=None,
    )
    return resp, semantic_pipeline_failed, sem_result


async def _attach_match_explainability(
    resp: CVAnalysisResponse,
    result: CVAnalysisOutput,
    cv_text: str,
    job_description: Optional[str],
    sem: Optional[SemanticMatchResult],
) -> CVAnalysisResponse:
    if not settings.use_match_explainers or sem is None:
        return resp
    job_stripped = (job_description or "").strip()
    if not job_stripped:
        return resp
    cv_exp = _experience_text_from_result(result)
    try:
        me = await asyncio.to_thread(
            explain_match_score,
            sem=sem,
            skills=result.skills or [],
            experience=result.experience,
            cv_experience_text=cv_exp or cv_text[:8000],
            cv_full_text=cv_text,
            job_requirements_text=job_stripped,
            job_full_text=job_stripped,
        )
        if me is not None:
            return resp.model_copy(update={"match_explainability": me})
    except Exception:
        logger.warning("Match explainability failed; continuing without SHAP/LIME", exc_info=True)
    return resp


_CANDIDATE_FIRST_PERSON_RE = re.compile(
    r"\b(?:"
    r"я\s+маю|я\s+вже|я\s+працював|я\s+працювала|мені\s+потрібно|мені\s+необхідно|"
    r"моя\s+робота|мій\s+досвід|моє\s+резюме|мене\s+|від\s+мене|"
    r"я\s+розробляв|я\s+розробляла"
    r")\b",
    re.IGNORECASE,
)


def _summary_uses_candidate_first_person(summary: str) -> bool:
    return bool(_CANDIDATE_FIRST_PERSON_RE.search(summary))


def _sentence_count(text: str) -> int:
    return len([p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()])


def _format_explainability_for_prompt(resp: CVAnalysisResponse) -> str:
    me = resp.match_explainability
    if not me:
        return "(локальні пояснення SHAP/LIME недоступні)\n"
    lines: List[str] = []
    comps = me.component_attributions
    if comps:
        lines.append("Внесок блоків у загальний бал (бали):")
        for key, label in (
            ("skills", "Навички"),
            ("experience", "Досвід"),
            ("overall", "Загальна схожість"),
        ):
            v = comps.get(key)
            if v is not None:
                lines.append(f"  {label}: {v * 100:+.2f} балів")
    for method, title in ((me.shap, "SHAP — підвищують бал"), (me.lime, "LIME — підвищують бал")):
        if not method:
            continue
        if method.top_positive:
            lines.append(f"{title}:")
            for item in method.top_positive[:6]:
                lines.append(f"  + {item.feature}: {item.contribution * 100:+.2f} балів")
        if method.top_negative:
            lines.append("Знижують бал (SHAP/LIME):")
            for item in method.top_negative[:6]:
                lines.append(f"  − {item.feature}: {item.contribution * 100:+.2f} балів")
    return "\n".join(lines) + ("\n" if lines else "")


def _summary_needs_expansion(summary: Any, job_description_section: str) -> bool:
    """When a job is provided, require a substantive coaching summary on «ви», not «я»."""
    if not isinstance(summary, str):
        return True
    text = summary.strip()
    if _summary_uses_candidate_first_person(text):
        return True
    if "Опис вакансії (вимоги):" not in job_description_section:
        return len(text) < 300
    if _sentence_count(text) < 8:
        return True
    if len(text) < 500:
        return True
    lower = text.lower()
    has_gap_hint = any(
        tok in lower
        for tok in (
            "проте",
            "однак",
            "втім",
            "бракує",
            "прогалин",
            "не вистачає",
            "варто",
            "рекоменд",
            "порад",
            "висновок",
            "підсум",
            "вам ",
            "у вас",
            "ви ",
        )
    )
    return not has_gap_hint


def _job_section(job_description: Optional[str]) -> str:
    if job_description and job_description.strip():
        return (
            f"\n\nОпис вакансії (вимоги):\n{job_description.strip()}\n\n"
            "Порівняй резюме з вимогами. Заповни matched_competencies та missing_competencies українською. "
            "Залиш match_score і match_score_reasoning null. "
            "analysis.summary — 2–4 абзаци на «ви» (не від імені кандидата «я»); без заголовків розділів."
        )
    return (
        "\n\nОпис вакансії не надано. matched_competencies і missing_competencies мають бути []. "
        "match_score і match_score_reasoning — null. "
        "Дай загальні ринкові рекомендації українською щодо покращення резюме: лише дії для кандидата, без реклами вакансій."
    )


class CVAnalyzer:
    def __init__(self) -> None:
        self._ollama_llm = None
        self._ollama_llm_prose = None
        self._gemini_llm = None

    def _coaching_llm(self, raw_llm: Any) -> Any:
        """Prose rewrite without JSON mode in development (faster, cleaner Ukrainian)."""
        if settings.environment == "development":
            return self._ensure_ollama_llm_prose()
        return raw_llm

    async def _enrich_coaching_summary(
        self,
        resp: CVAnalysisResponse,
        raw_llm: Any,
        job_description: Optional[str],
        cv_text_for_prompt: str,
    ) -> CVAnalysisResponse:
        if not settings.use_llm_coaching_summary:
            return resp
        jd = (job_description or "").strip()
        draft = ""
        if isinstance(resp.analysis, dict):
            raw_sum = resp.analysis.get("summary")
            if isinstance(raw_sum, str):
                draft = raw_sum.strip()

        has_scores = resp.match_score is not None and resp.semantic_breakdown is not None and bool(jd)
        sb = resp.semantic_breakdown or {}
        sw = resp.semantic_weights or {}
        ws, we, wo = float(sw.get("skills", 0.5)), float(sw.get("experience", 0.3)), float(sw.get("overall", 0.2))
        matched = ", ".join(resp.matched_competencies or []) or "—"
        missing = ", ".join(resp.missing_competencies or []) or "—"
        skills_line = ", ".join(s for s in (resp.skills or [])[:80] if s and str(s).strip()) or "—"

        score_block = ""
        if has_scores:
            units = int(round(float(resp.match_score) * 100))
            score_block = (
                f"Загальний бал відповідності: {units} з 100 (дробово {float(resp.match_score):.4f}).\n"
                f"Ваги зведення: навички {ws:.2f}, досвід {we:.2f}, загальна схожість {wo:.2f}.\n"
                f"Подібність навичок: {float(sb.get('skills_similarity', 0)):.1%}; "
                f"досвіду: {float(sb.get('experience_similarity', 0)):.1%}; "
                f"повного тексту: {float(sb.get('overall_similarity', 0)):.1%}.\n"
                f"Збігається з вакансією: {matched}\n"
                f"Слабко/відсутнє: {missing}\n"
                f"{_format_explainability_for_prompt(resp)}"
            )

        human = (
            "Чернетка підсумку від першого проходу (можеш переписати повністю):\n"
            f"{draft or '(немає)'}\n\n"
            f"Навички з резюме: {skills_line}\n\n"
        )
        if jd:
            human += f"Уривок вакансії:\n{jd[:2500]}\n\n"
        if has_scores:
            human += f"Дані для пояснення балу (що підвищує / знижує):\n{score_block}\n"
        else:
            human += "Бал відповідності не пораховано (немає вакансії або збій семантики). Пиши загальний підсумок резюме без вигаданих відсотків.\n"
        human += f"Уривок резюме:\n{(cv_text_for_prompt or '')[:2500]}\n"

        try:
            llm = self._coaching_llm(raw_llm)
            out = await llm.ainvoke(
                [SystemMessage(content=_COACHING_SUMMARY_SYSTEM), HumanMessage(content=human)]
            )
            text = normalize_coaching_summary(_llm_message_text(out).strip())
            if not text or _summary_uses_candidate_first_person(text):
                logger.warning("Coaching summary enrichment empty or first-person; keeping draft")
                return resp
            if _sentence_count(text) < 8:
                retry_human = (
                    "Попередній текст занадто короткий. Мінімум 10 речень на «ви», з поясненням що підвищує і знижує бал.\n\n"
                    + human
                )
                out2 = await llm.ainvoke(
                    [SystemMessage(content=_COACHING_SUMMARY_SYSTEM), HumanMessage(content=retry_human)]
                )
                text2 = normalize_coaching_summary(_llm_message_text(out2).strip())
                if text2 and _sentence_count(text2) >= 8 and not _summary_uses_candidate_first_person(text2):
                    text = text2
            if len(text) > 8000:
                text = text[:8000].rsplit(" ", 1)[0] + "…"
            analysis = dict(resp.analysis) if isinstance(resp.analysis, dict) else {}
            analysis["summary"] = text
            analysis["strengths"] = []
            analysis["weaknesses"] = []
            logger.info("Coaching summary enriched (%s sentences)", _sentence_count(text))
            return resp.model_copy(update={"analysis": analysis})
        except Exception:
            logger.warning("Coaching summary enrichment failed; keeping draft summary", exc_info=True)
            return resp

    async def _enrich_semantic_score_narrative(
        self,
        resp: CVAnalysisResponse,
        raw_llm: Any,
        job_description: Optional[str],
        cv_text_for_prompt: str,
    ) -> CVAnalysisResponse:
        if not settings.use_llm_semantic_narrative:
            return resp
        if not resp.semantic_breakdown or resp.match_score is None:
            return resp
        jd = (job_description or "").strip()
        if not jd:
            return resp
        sb = resp.semantic_breakdown
        sw = resp.semantic_weights or {}
        ws, we, wo = float(sw.get("skills", 0.5)), float(sw.get("experience", 0.3)), float(sw.get("overall", 0.2))
        skills_line = ", ".join(s for s in (resp.skills or [])[:120] if s and str(s).strip()) or "(не витягнуто)"
        cv_snip = (cv_text_for_prompt or "").strip()[:2800]
        job_snip = jd[:2800]
        human = (
            "Нижче подібності кожна в діапазоні від 0 до 1 (більше — ближче за змістом у просторі ембеддингів).\n\n"
            f"Загальний match_score (зважена суміш): {float(resp.match_score):.4f}\n"
            f"Ваги: навички {ws:.4f}, досвід {we:.4f}, загальна схожість {wo:.4f}\n\n"
            f"skills_similarity (навички vs вакансія): {float(sb.get('skills_similarity', 0)):.4f}\n"
            f"experience_similarity (досвід vs вакансія): {float(sb.get('experience_similarity', 0)):.4f}\n"
            f"overall_similarity (повне резюме vs повний текст вакансії): {float(sb.get('overall_similarity', 0)):.4f}\n\n"
            f"Витягнуті навички (через кому): {skills_line}\n\n"
            f"Уривок вакансії:\n{job_snip}\n\n"
            f"Уривок резюме:\n{cv_snip}\n\n"
            "Напиши пояснення для кандидата згідно з інструкціями."
        )
        try:
            out = await raw_llm.ainvoke(
                [SystemMessage(content=_SEMANTIC_NARRATIVE_SYSTEM), HumanMessage(content=human)]
            )
            text = _llm_message_text(out).strip()
            if not text:
                return resp
            if len(text) > 6000:
                text = text[:6000].rsplit(" ", 1)[0] + "…"
            return resp.model_copy(update={"semantic_score_narrative": text})
        except Exception:
            logger.warning("LLM semantic score narrative failed; response left without narrative", exc_info=True)
            return resp

    async def _llm_fallback_match_score(
        self,
        resp: CVAnalysisResponse,
        raw_llm: Any,
        cv_text_for_prompt: str,
        job_description: Optional[str],
    ) -> CVAnalysisResponse:
        jd = (job_description or "").strip()
        if not jd or resp.match_score is not None:
            return resp
        cv_snip = (cv_text_for_prompt or "").strip()[:6000]
        job_snip = jd[:6000]
        human = (
            "Оціни відповідність за цими уривками.\n\n"
            f"Резюме:\n{cv_snip}\n\n"
            f"Вакансія:\n{job_snip}"
        )
        structured = raw_llm.with_structured_output(MatchScoreFallbackOutput)
        try:
            out = await structured.ainvoke(
                [
                    SystemMessage(content=_FALLBACK_MATCH_SCORE_SYSTEM),
                    HumanMessage(content=human),
                ]
            )
            reason = (out.match_score_reasoning or "").strip()
            if not reason:
                return resp
            return resp.model_copy(
                update={
                    "match_score": float(out.match_score),
                    "match_score_reasoning": reason,
                }
            )
        except Exception:
            logger.warning("LLM fallback match score failed; leaving match_score unset", exc_info=True)
            return resp

    async def analyze_cv(
        self,
        cv_text: str,
        request: Optional[CVAnalysisRequest] = None,
    ) -> CVAnalysisResponse:
        cv_text = normalize_text_for_pipeline(cv_text or "")
        job = request.job_description if request else None
        if job:
            job = normalize_text_for_pipeline(job)

        if settings.environment == "development":
            return await self._analyze_with_ollama(cv_text, job)
        return await self._analyze_with_gemini(cv_text, job)

    def _ensure_ollama_llm(self) -> Any:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("langchain-ollama package is required for development")

        if not self._ollama_llm:
            self._ollama_llm = ChatOllama(
                model=settings.ollama_model,
                temperature=0.1,
                num_ctx=settings.ollama_num_ctx,
                format="json",
                num_predict=settings.ollama_extract_num_predict,
            )
            logger.info(
                "Using Ollama model: %s (num_ctx=%s, format=json, num_predict=%s)",
                settings.ollama_model,
                settings.ollama_num_ctx,
                settings.ollama_extract_num_predict,
            )
        return self._ollama_llm

    def _ensure_ollama_llm_prose(self) -> Any:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("langchain-ollama package is required for development")

        if not self._ollama_llm_prose:
            self._ollama_llm_prose = ChatOllama(
                model=settings.ollama_model,
                temperature=0.2,
                num_ctx=settings.ollama_num_ctx,
                num_predict=settings.ollama_coaching_num_predict,
            )
            logger.info(
                "Ollama prose client: %s (num_predict=%s)",
                settings.ollama_model,
                settings.ollama_coaching_num_predict,
            )
        return self._ollama_llm_prose

    async def _ollama_invoke_json(
        self,
        cv_text_for_prompt: str,
        job_description_section: str,
        *,
        attempt: int,
    ) -> Optional[CVAnalysisOutput]:
        llm = self._ensure_ollama_llm()
        sys_content = f"{_ollama_json_prefix()}{_CV_ANALYZER_SYSTEM}"
        human_content = _HUMAN_PROMPT.format(
            cv_text=cv_text_for_prompt,
            job_description_section=job_description_section,
        )
        if attempt > 1:
            human_content = (
                "Попередня відповідь: порожня, коротка, з мітками розділів або написана від «я» (кандидат). "
                "analysis.summary — 2–4 абзаци, коуч звертається до кандидата на «ви» (у вас, вам варто); "
                "ЗАБОРОНЕНО «я», «мені», «моя робота», «від мене».\n\n"
                + human_content
            )
        messages = [
            SystemMessage(content=sys_content),
            HumanMessage(content=human_content),
        ]
        try:
            resp = await llm.ainvoke(messages)
        except Exception:
            logger.exception("Ollama JSON invoke failed (attempt %s)", attempt)
            return None
        raw_text = _llm_message_text(resp)
        blob = _parse_llm_json_blob(raw_text)
        if not blob:
            logger.warning(
                "Ollama JSON: no parseable object (attempt %s, preview: %.220s)",
                attempt,
                raw_text.replace("\n", " "),
            )
            return None
        parsed = _cv_output_from_parsed_dict(blob)
        if parsed is None:
            logger.warning("Ollama JSON: validation failed (attempt %s)", attempt)
        return parsed

    async def _ollama_extract_json(
        self,
        cv_text_for_prompt: str,
        job_description_section: str,
    ) -> Optional[CVAnalysisOutput]:
        max_attempts = max(1, settings.ollama_json_max_attempts)
        last: Optional[CVAnalysisOutput] = None
        for attempt in range(1, max_attempts + 1):
            result = await self._ollama_invoke_json(
                cv_text_for_prompt,
                job_description_section,
                attempt=attempt,
            )
            if result is None:
                continue
            result, incomplete = _normalize_llm_result(result)
            if incomplete:
                logger.warning("Ollama JSON incomplete (attempt %s); filled defaults.", attempt)
            last = result
            if not _is_extraction_empty(result):
                summ = (result.analysis or {}).get("summary") if isinstance(result.analysis, dict) else None
                if _should_retry_ollama_for_summary(summ, job_description_section) and attempt < max_attempts:
                    logger.warning(
                        "Ollama JSON: summary too brief or missing conclusions (attempt %s), retrying",
                        attempt,
                    )
                    continue
                if attempt > 1:
                    logger.info("Ollama JSON succeeded on attempt %s", attempt)
                return result
        return last

    async def _run_structured_chain(
        self,
        cv_text: str,
        cv_text_for_prompt: str,
        job_description: Optional[str],
        structured_llm: Any,
        raw_llm: Any,
        backend: str,
        model_name: str,
    ) -> CVAnalysisResponse:
        job_description_section = _job_section(job_description)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", _CV_ANALYZER_SYSTEM),
                ("human", _HUMAN_PROMPT),
            ]
        )
        try:
            chain = prompt_template | structured_llm
            result: CVAnalysisOutput = await chain.ainvoke(
                {
                    "cv_text": cv_text_for_prompt,
                    "job_description_section": job_description_section,
                }
            )
            result, incomplete = _normalize_llm_result(result)
            if incomplete:
                logger.warning("LLM returned incomplete response; filled defaults.")
            if _is_extraction_empty(result):
                return CVAnalysisResponse(
                    success=False,
                    extracted_text=cv_text,
                    error=_empty_extraction_error(
                        backend,
                        model_name,
                        len(cv_text_for_prompt),
                        len(cv_text),
                    ),
                )
            built, sem_failed, sem = _build_success_response(cv_text, result, job_description)
            built = await _attach_match_explainability(
                built, result, cv_text, job_description, sem
            )
            if sem_failed:
                built = await self._llm_fallback_match_score(
                    built, raw_llm, cv_text_for_prompt, job_description
                )
            built = await self._enrich_coaching_summary(
                built, raw_llm, job_description, cv_text_for_prompt
            )
            if settings.use_llm_semantic_narrative:
                return await self._enrich_semantic_score_narrative(
                    built, raw_llm, job_description, cv_text_for_prompt
                )
            return built
        except Exception as e:
            error_msg = f"Помилка аналізу: {e!s}\n{traceback.format_exc()}"
            return CVAnalysisResponse(success=False, extracted_text=cv_text, error=error_msg)

    def _cv_text_for_prompt(self, cv_text: str) -> str:
        max_chars = settings.max_cv_chars_for_llm
        if max_chars > 0 and len(cv_text) > max_chars:
            logger.warning("CV truncated from %d to %d chars for LLM context", len(cv_text), max_chars)
            return (
                cv_text[:max_chars]
                + "\n\n[Текст резюме обрізано для контексту моделі. Витягни дані з наведеного вище.]"
            )
        return cv_text

    async def _analyze_with_ollama(
        self,
        cv_text: str,
        job_description: Optional[str] = None,
    ) -> CVAnalysisResponse:
        cv_text_for_prompt = self._cv_text_for_prompt(cv_text)
        job_description_section = _job_section(job_description)
        llm = self._ensure_ollama_llm()

        result = await self._ollama_extract_json(cv_text_for_prompt, job_description_section)
        if result is None or _is_extraction_empty(result):
            logger.info("Ollama JSON empty after retries; trying LangChain structured output")
            structured_llm = llm.with_structured_output(CVAnalysisOutput)
            return await self._run_structured_chain(
                cv_text,
                cv_text_for_prompt,
                job_description,
                structured_llm,
                llm,
                "Ollama",
                settings.ollama_model,
            )

        built, sem_failed, sem = _build_success_response(cv_text, result, job_description)
        built = await _attach_match_explainability(built, result, cv_text, job_description, sem)
        if sem_failed:
            built = await self._llm_fallback_match_score(
                built, llm, cv_text_for_prompt, job_description
            )
        built = await self._enrich_coaching_summary(built, llm, job_description, cv_text_for_prompt)
        if settings.use_llm_semantic_narrative:
            return await self._enrich_semantic_score_narrative(
                built, llm, job_description, cv_text_for_prompt
            )
        return built

    async def _analyze_with_gemini(
        self,
        cv_text: str,
        job_description: Optional[str] = None,
    ) -> CVAnalysisResponse:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai package is required for production")

        cv_text_for_prompt = self._cv_text_for_prompt(cv_text)

        if not self._gemini_llm:
            self._gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
            )

        structured_llm = self._gemini_llm.with_structured_output(CVAnalysisOutput)
        return await self._run_structured_chain(
            cv_text,
            cv_text_for_prompt,
            job_description,
            structured_llm,
            self._gemini_llm,
            "Gemini",
            "gemini-2.5-flash",
        )
