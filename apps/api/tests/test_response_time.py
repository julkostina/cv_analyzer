"""Response-time benchmarks for NFR-01 (run locally; skipped in default CI).

Usage:
  cd apps/api
  RUN_LIVE_BENCHMARKS=1 pytest tests/test_response_time.py -v -s

Optional:
  BENCHMARK_RUNS=3          — repeat live pipeline and show median
  NFR_MAX_SECONDS=90        — fail if live pipeline exceeds this limit
"""
from __future__ import annotations

import os
import statistics
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.models import CVAnalysisRequest, CVAnalysisResponse
from tests.benchmark_samples import SAMPLE_CV_TEXT, SAMPLE_JOB_TEXT
from tests.test_analyze import MINIMAL_PDF


def _elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def _print_benchmark(title: str, seconds: float, *, runs: int = 1) -> None:
    print(f"\n=== {title} ===")
    if runs > 1:
        print(f"Runs: {runs}")
    print(f"Elapsed: {seconds:.3f} s")


@pytest.mark.asyncio
async def test_analyze_endpoint_overhead_with_mocks(client):
    """Router + upload path without LLM/embeddings (baseline overhead)."""
    mock_response = CVAnalysisResponse(
        success=True,
        extracted_text=SAMPLE_CV_TEXT[:200],
        analysis={"summary": "Benchmark mock summary."},
        skills=["React"],
        experience=[],
        education=[],
        match_score=0.8,
        recommendations=[],
        error=None,
    )

    start = time.perf_counter()
    with (
        patch("app.routers.cv_router.CVParser") as MockParser,
        patch("app.routers.cv_router.CVAnalyzer") as MockAnalyzer,
    ):
        MockParser.return_value.parse_file = AsyncMock(return_value=SAMPLE_CV_TEXT)
        MockAnalyzer.return_value.analyze_cv = AsyncMock(return_value=mock_response)

        response = await client.post(
            "/api/v1/analyze",
            files={"file": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
            data={"job_description": SAMPLE_JOB_TEXT},
        )

    elapsed = _elapsed_seconds(start)
    _print_benchmark("Analyze endpoint (mocked pipeline)", elapsed)

    assert response.status_code == 200
    assert elapsed < 5.0, f"Mocked endpoint took unexpectedly long: {elapsed:.3f}s"


def test_semantic_match_response_time():
    """Embedding + cosine step only (no LLM). Skips if model unavailable."""
    pytest.importorskip("sentence_transformers")

    from app.services.semantic_matcher import compute_semantic_match

    skills = ["React", "TypeScript", "JavaScript", "HTML", "CSS", "Git"]

    try:
        compute_semantic_match(
            cv_skills=skills,
            cv_experience_text=SAMPLE_CV_TEXT[:8000],
            cv_full_text=SAMPLE_CV_TEXT,
            job_requirements_text=SAMPLE_JOB_TEXT,
            job_full_text=SAMPLE_JOB_TEXT,
        )
    except Exception as exc:
        pytest.skip(f"Sentence Transformers model unavailable: {exc}")

    start = time.perf_counter()
    result = compute_semantic_match(
        cv_skills=skills,
        cv_experience_text=SAMPLE_CV_TEXT[:8000],
        cv_full_text=SAMPLE_CV_TEXT,
        job_requirements_text=SAMPLE_JOB_TEXT,
        job_full_text=SAMPLE_JOB_TEXT,
    )
    elapsed = _elapsed_seconds(start)

    _print_benchmark("Semantic match (Sentence Transformers, warm)", elapsed)
    print(f"match_score: {result.score:.3f}")

    assert result.score is not None
    assert elapsed < 30.0, f"Semantic step alone exceeded 30s: {elapsed:.3f}s"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_analyze_pipeline_response_time():
    """Full CVAnalyzer pipeline (LLM + semantic + optional explainers). Requires Ollama/Gemini."""
    if not os.getenv("RUN_LIVE_BENCHMARKS"):
        pytest.skip("Set RUN_LIVE_BENCHMARKS=1 to run live NFR-01 timing")

    from app.config import settings
    from app.services.cv_analyzer import CVAnalyzer

    runs = max(1, int(os.getenv("BENCHMARK_RUNS", "1")))
    analyzer = CVAnalyzer()
    request = CVAnalysisRequest(job_description=SAMPLE_JOB_TEXT)

    timings: list[float] = []
    last_response: CVAnalysisResponse | None = None

    for i in range(runs):
        start = time.perf_counter()
        last_response = await analyzer.analyze_cv(SAMPLE_CV_TEXT, request)
        timings.append(_elapsed_seconds(start))
        print(f"  run {i + 1}/{runs}: {timings[-1]:.3f} s")

    elapsed = statistics.median(timings) if runs > 1 else timings[0]

    _print_benchmark("Live analyze pipeline (CVAnalyzer.analyze_cv)", elapsed, runs=runs)
    print(f"environment: {settings.environment}")
    print(f"ollama_model: {settings.ollama_model}")
    print(f"use_match_explainers: {settings.use_match_explainers}")
    print(f"use_llm_coaching_summary: {settings.use_llm_coaching_summary}")
    print(f"success: {last_response.success if last_response else False}")
    if last_response and last_response.error:
        print(f"error: {last_response.error[:300]}")

    assert last_response is not None
    assert last_response.success, last_response.error or "analysis failed"

    max_seconds = os.getenv("NFR_MAX_SECONDS")
    if max_seconds:
        limit = float(max_seconds)
        assert elapsed <= limit, (
            f"Pipeline median {elapsed:.3f}s exceeds NFR_MAX_SECONDS={limit}"
        )
    else:
        print(
            "\nTip: set NFR_MAX_SECONDS=<your NFR limit> to enforce the threshold "
            "after you update NFR-01 in the thesis."
        )


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_analyze_endpoint_response_time(client):
    """End-to-end POST /analyze with real analyzer; parser returns fixture CV text."""
    if not os.getenv("RUN_LIVE_BENCHMARKS"):
        pytest.skip("Set RUN_LIVE_BENCHMARKS=1 to run live NFR-01 timing")

    from app.config import settings

    start = time.perf_counter()
    with patch("app.routers.cv_router.CVParser") as MockParser:
        MockParser.return_value.parse_file = AsyncMock(return_value=SAMPLE_CV_TEXT)

        response = await client.post(
            "/api/v1/analyze",
            files={"file": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
            data={"job_description": SAMPLE_JOB_TEXT},
        )

    elapsed = _elapsed_seconds(start)

    _print_benchmark("Live POST /api/v1/analyze (real analyzer)", elapsed)
    print(f"environment: {settings.environment}")
    print(f"status: {response.status_code}")

    assert response.status_code == 200, response.text[:500]
    data = response.json()
    assert data.get("success") is True, data.get("error")

    max_seconds = os.getenv("NFR_MAX_SECONDS")
    if max_seconds:
        assert elapsed <= float(max_seconds), (
            f"Endpoint took {elapsed:.3f}s, limit is {max_seconds}s"
        )
