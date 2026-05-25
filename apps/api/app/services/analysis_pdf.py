"""Build a PDF report from a CV analysis API response."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from app.config import settings
from app.models import CVAnalysisResponse

logger = logging.getLogger(__name__)

_DEJAVU_CDN_TTF = (
    "https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans.ttf"
)


def _font_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    d = root / "cv-analyzer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_dejavu_font_path() -> Path:
    explicit = (settings.pdf_font_path or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise RuntimeError(f"PDF_FONT_PATH is not a valid file: {p}")

    cached = _font_cache_dir() / "DejaVuSans.ttf"
    if cached.is_file() and cached.stat().st_size > 10_000:
        return cached

    logger.info("Downloading DejaVu Sans for PDF (one-time cache): %s", cached)
    try:
        urllib.request.urlretrieve(_DEJAVU_CDN_TTF, cached)  # noqa: S310
    except Exception as e:
        raise RuntimeError(
            "Could not download DejaVu font for PDF. Check network or set PDF_FONT_PATH to a local .ttf file."
        ) from e
    return cached


def render_analysis_pdf(resp: CVAnalysisResponse) -> bytes:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError as e:
        raise RuntimeError("PDF export requires the fpdf2 package: pip install fpdf2") from e

    font_path = _resolve_dejavu_font_path()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("DejaVu", "", str(font_path))
    pdf.set_font("DejaVu", "", 11)
    line_w = pdf.epw

    def _line(text: str) -> None:
        for paragraph in (text or "").split("\n"):
            p = (paragraph or " ").strip() or " "
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(line_w, 6, p, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.set_font("DejaVu", "", 14)
    _line("Звіт з аналізу резюме")
    pdf.set_font("DejaVu", "", 11)

    if not resp.success:
        _line("Помилка обробки")
        _line(resp.error or "Невідома помилка")
        return _pdf_bytes(pdf)

    if resp.match_score is not None:
        pct = round(float(resp.match_score) * 100, 1)
        _line(f"Бал відповідності вакансії: {pct}%")

    if resp.semantic_breakdown:
        sb = resp.semantic_breakdown
        _line("Семантичне розбиття (схожість 0–1):")
        _line(
            f"  Навички: {sb.get('skills_similarity', 0):.3f}\n"
            f"  Досвід: {sb.get('experience_similarity', 0):.3f}\n"
            f"  Загальна схожість: {sb.get('overall_similarity', 0):.3f}"
        )
        if resp.semantic_weights:
            sw = resp.semantic_weights
            _line(
                "  Ваги (навички, досвід, загальна схожість): "
                f"{sw.get('skills', 0):.3f}, {sw.get('experience', 0):.3f}, {sw.get('overall', 0):.3f}"
            )

    if resp.semantic_score_narrative:
        _line("Інтерпретація (з моделі):")
        _line(resp.semantic_score_narrative)

    if resp.match_score_reasoning:
        _line("Обґрунтування балу:")
        _line(resp.match_score_reasoning)

    if resp.match_explainability:
        me = resp.match_explainability
        if me.component_attributions:
            _line("Внесок блоків у загальний бал (SHAP, лінійна декомпозиція):")
            labels = {"skills": "Навички", "experience": "Досвід", "overall": "Загальна схожість"}
            for key, label in labels.items():
                val = me.component_attributions.get(key)
                if val is not None:
                    _line(f"  {label}: {val * 100:+.2f} балів")
        for method_result, title in ((me.shap, "SHAP"), (me.lime, "LIME")):
            if not method_result:
                continue
            _line(f"{title} (локальні атрибуції):")
            if method_result.top_positive:
                _line("  Підвищують бал:")
                for item in method_result.top_positive:
                    _line(f"    • {item.feature}: {item.contribution * 100:+.2f} балів")
            if method_result.top_negative:
                _line("  Знижують бал:")
                for item in method_result.top_negative:
                    _line(f"    • {item.feature}: {item.contribution * 100:+.2f} балів")

    if resp.matched_competencies:
        _line("Відповідні компетенції:")
        _line("\n".join(f"• {x}" for x in resp.matched_competencies))

    if resp.missing_competencies:
        _line("Слабкі або відсутні компетенції:")
        _line("\n".join(f"• {x}" for x in resp.missing_competencies))

    if resp.analysis:
        summ = resp.analysis.get("summary")
        summ_text = str(summ).strip() if summ else ""
        if summ_text:
            _line("Підсумок та поради:")
            _line(summ_text)
        else:
            strengths = resp.analysis.get("strengths")
            weaknesses = resp.analysis.get("weaknesses")
            if strengths:
                _line("Сильні сторони:")
                if isinstance(strengths, list):
                    _line("\n".join(f"• {s}" for s in strengths))
                else:
                    _line(str(strengths))
            if weaknesses:
                _line("Слабкі сторони:")
                if isinstance(weaknesses, list):
                    _line("\n".join(f"• {w}" for w in weaknesses))
                else:
                    _line(str(weaknesses))

    if resp.recommendations and not (resp.analysis and str(resp.analysis.get("summary") or "").strip()):
        _line("Рекомендовані покращення:")
        _line("\n".join(f"• {r}" for r in resp.recommendations))

    if resp.skills:
        _line("Витягнуті навички:")
        _line(", ".join(resp.skills))

    return _pdf_bytes(pdf)


def _pdf_bytes(pdf) -> bytes:
    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return bytes(raw)
