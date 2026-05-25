import logging
import os
import traceback
from pathlib import Path
from typing import Annotated, Optional

import aiofiles
import httpx
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.models import CVAnalysisRequest, CVAnalysisResponse
from app.services.analysis_pdf import render_analysis_pdf
from app.services.cv_analyzer import CVAnalyzer
from app.services.cv_parser import CVParser
from app.utils.file_validator import validate_file
from app.utils.job_url_extract import extract_job_text_from_html
from app.utils.text_preprocess import normalize_text_for_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

JOB_URL_CONTENT_LIMIT = 50_000


async def _fetch_job_description_from_url(url: str) -> str:
    """Fetch URL and return job posting as readable plain text (JSON-LD JobPosting preferred)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CVAnalyzer/1.0; +https://localhost)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    if len(html) > JOB_URL_CONTENT_LIMIT:
        html = html[:JOB_URL_CONTENT_LIMIT]

    extracted = extract_job_text_from_html(html)
    if not extracted or len(extracted.strip()) < 80:
        raise ValueError(
            "Сторінка вакансії не містить розпізнаного тексту оголошення. "
            "Вставте опис вакансії вручну в поле «Вимоги / опис вакансії»."
        )
    if len(extracted) > JOB_URL_CONTENT_LIMIT:
        extracted = extracted[:JOB_URL_CONTENT_LIMIT] + "\n[... обрізано]"
    return normalize_text_for_pipeline(extracted)


@router.post("/analyze", response_model=None)
async def analyze_cv(
    file: UploadFile = File(..., description="CV file (PDF or DOCX)"),
    job_description: Annotated[
        Optional[str], Form(description="Position requirements as plain text (optional)")
    ] = None,
    job_description_url: Annotated[
        Optional[str],
        Form(description="URL of job posting to fetch requirements from (optional). Used with or instead of job_description."),
    ] = None,
    return_pdf: Annotated[
        bool,
        Form(description="If true and analysis succeeds, response is application/pdf."),
    ] = False,
):
    file_path = None
    try:
        job_text = (job_description or "").strip()
        if job_description_url and job_description_url.strip():
            try:
                url_content = await _fetch_job_description_from_url(job_description_url.strip())
                job_text = f"{job_text}\n\n{url_content}".strip() if job_text else url_content
            except Exception as e:
                logger.warning("Failed to fetch job_description_url %s: %s", job_description_url, e)
                raise HTTPException(
                    status_code=400,
                    detail=f"Не вдалося завантажити опис вакансії за посиланням: {e!s}",
                ) from e

        job_text = normalize_text_for_pipeline(job_text) if job_text else ""

        content = await file.read()
        file_type, safe_filename = validate_file(
            content,
            file.filename,
            file.content_type,
            settings.max_upload_size_bytes,
        )
        file_path = UPLOAD_DIR / safe_filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        parser = CVParser()
        try:
            cv_text = await parser.parse_file(str(file_path), file_type)
        except Exception as e:
            logger.warning("CV parse failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Не вдалося витягти текст із цього файлу. "
                    "Переконайтеся, що PDF не захищений паролем і не пошкоджений; для DOCX спробуйте «Зберегти як» нову копію. "
                    "Якщо проблема лишається, експортуйте резюме в інший формат (наприклад, PDF із видимим текстом)."
                ),
            ) from e
        cv_text = normalize_text_for_pipeline(cv_text or "")

        if not cv_text:
            raise HTTPException(
                status_code=400,
                detail="Не вдалося розібрати резюме: текст порожній або недоступний.",
            )

        analyzer = CVAnalyzer()
        request = (
            CVAnalysisRequest(job_description=job_text or None)
            if job_text
            else None
        )
        result = await analyzer.analyze_cv(cv_text, request)

        if return_pdf:
            if not result.success:
                raise HTTPException(
                    status_code=400,
                    detail=result.error or "Analysis failed; PDF was not generated.",
                )
            pdf_bytes = render_analysis_pdf(result)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": 'attachment; filename="cv-analysis-report.pdf"',
                },
            )

        return result

    except ValueError as e:
        logger.warning("CV analyze validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CV analyze failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=traceback.format_exc() if settings.environment == "development" else str(e),
        )
    finally:
        if file_path is not None and file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                pass


@router.post("/report/pdf")
async def report_pdf_from_analysis(result: CVAnalysisResponse = Body(...)):
    """Build PDF from an existing analysis response (no re-upload or re-analysis)."""
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=result.error or "Analysis failed; PDF was not generated.",
        )
    try:
        pdf_bytes = render_analysis_pdf(result)
    except Exception as e:
        logger.exception("PDF render failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cv-analysis-report.pdf"'},
    )