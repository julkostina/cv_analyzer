"""Embeddings + cosine similarity between CV and job text blocks."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# Shown in API/UI when semantic matching is used (same keys as semantic_weights response).
SEMANTIC_METRIC_GUIDES: Dict[str, str] = {
    "skills": (
        "Низький бал навичок часто означає, що список навичок порожній або дуже короткий, "
        "або що він слабо нагадує текст вакансії в просторі векторних подань. "
        "Це не означає, що у вас немає навичок."
    ),
    "experience": (
        "Порівнюється текстовий блок із ваших посад (назва, роботодавець, дати) "
        "або запасний фрагмент резюме з усім текстом вакансії. "
        "Це не арифметика років досвіду й не структурний чекліст."
    ),
    "overall": (
        "Це схожість повного тексту резюме та повного опису вакансії (косинус у просторі векторів). "
        "Це не лише загальний бал відповідності."
    ),
    "match_score": (
        "Загальний бал — зважена комбінація трьох показників вище з вагами з цієї відповіді "
        "(типово: навички 0,5, досвід 0,3, загальна схожість 0,2). "
        "Адміністратор може змінити ваги в конфігурації сервера."
    ),
}


def normalized_semantic_weights() -> Tuple[float, float, float]:
    w_skills = float(settings.semantic_weights_skills)
    w_exp = float(settings.semantic_weights_experience)
    w_overall = float(settings.semantic_weights_overall)
    total = w_skills + w_exp + w_overall
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w_skills / total, w_exp / total, w_overall / total)


@dataclass
class SemanticMatchResult:
    score: float
    skills_similarity: float
    experience_similarity: float
    overall_similarity: float


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    import numpy as np
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos = np.dot(va, vb) / (norm_a * norm_b)
    return float((cos + 1) / 2)


class _Embedder:
    def __init__(self, embed_fn, dim: int):
        self._embed = embed_fn
        self._dim = dim

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


def _get_openai_embedder():
    from langchain_openai import OpenAIEmbeddings
    key = settings.openai_api_key or ""
    if not key:
        return None
    model = getattr(settings, "openai_embedding_model", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, openai_api_key=key)


def _get_embedder():
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        emb = _get_openai_embedder()
        if emb is not None:
            return emb
        logger.warning("embedding_provider=openai but OPENAI_API_KEY not set; falling back to sentence_transformers")
    elif provider != "sentence_transformers":
        return _get_openai_embedder() or _get_sentence_transformers_embedder()

    try:
        return _get_sentence_transformers_embedder()
    except ImportError as e:
        emb = _get_openai_embedder()
        if emb is not None:
            logger.info("sentence_transformers not installed (%s); using OpenAI embeddings", e)
            return emb
        raise


def _get_sentence_transformers_embedder():
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    st = SentenceTransformer(model_name, device="cpu")
    dim = st.get_sentence_embedding_dimension()

    def embed_fn(text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * dim
        return st.encode(text.strip(), normalize_embeddings=True).tolist()

    def embed_batch(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result: List[List[float]] = [[0.0] * dim for _ in texts]
        non_empty_idx: List[int] = []
        non_empty: List[str] = []
        for i, t in enumerate(texts):
            s = (t or "").strip()
            if s:
                non_empty_idx.append(i)
                non_empty.append(s)
        if not non_empty:
            return result
        encoded = st.encode(non_empty, normalize_embeddings=True, batch_size=min(32, len(non_empty)))
        for idx, vec in zip(non_empty_idx, encoded):
            result[idx] = vec.tolist()
        return result

    embedder = _Embedder(embed_fn, dim)
    embedder._embed_batch = embed_batch  # type: ignore[attr-defined]
    return embedder


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = _get_embedder()
    return _embedder


def embed_text(text: str) -> List[float]:
    emb = get_embedder()
    if not (text or text.strip()):
        dim = getattr(emb, "dimension", None) or len(emb.embed_query(" "))
        return [0.0] * dim
    return emb.embed_query(text.strip())


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    emb = get_embedder()
    batch_fn = getattr(emb, "_embed_batch", None)
    if callable(batch_fn):
        return batch_fn(texts)
    return [embed_text(t) for t in texts]


def compute_semantic_match(
    cv_skills: List[str],
    cv_experience_text: str,
    cv_full_text: str,
    job_requirements_text: str,
    job_full_text: str,
) -> SemanticMatchResult:
    w_skills, w_exp, w_overall = normalized_semantic_weights()

    def _block(text: str, max_chars: int = 8000) -> str:
        if not text or not text.strip():
            return ""
        t = text.strip()
        return t[:max_chars] if len(t) > max_chars else t

    cv_skills_block = " ".join(s for s in cv_skills if s and s.strip()) if cv_skills else ""
    cv_exp_block = _block(cv_experience_text)
    cv_full_block = _block(cv_full_text, max_chars=12000)
    job_req_block = _block(job_requirements_text)
    job_full_block = _block(job_full_text, max_chars=12000)

    blocks = [cv_skills_block, cv_exp_block, cv_full_block, job_req_block, job_full_block]
    vecs = embed_texts_batch(blocks)
    skills_cv_vec = vecs[0] if cv_skills_block else None
    exp_cv_vec = vecs[1] if cv_exp_block else None
    full_cv_vec = vecs[2] if cv_full_block else None
    job_req_vec = vecs[3] if job_req_block else None
    job_full_vec = vecs[4] if job_full_block else None

    skills_sim = 0.0
    if skills_cv_vec and job_req_vec:
        skills_sim = _cosine_similarity(skills_cv_vec, job_req_vec)
    elif not job_req_vec:
        skills_sim = 1.0

    exp_sim = 0.0
    if exp_cv_vec and job_req_vec:
        exp_sim = _cosine_similarity(exp_cv_vec, job_req_vec)
    elif not job_req_vec:
        exp_sim = 1.0

    overall_sim = 0.0
    if full_cv_vec and job_full_vec:
        overall_sim = _cosine_similarity(full_cv_vec, job_full_vec)
    elif not job_full_block:
        overall_sim = 1.0

    score = w_skills * skills_sim + w_exp * exp_sim + w_overall * overall_sim
    score = max(0.0, min(1.0, score))

    return SemanticMatchResult(
        score=score,
        skills_similarity=skills_sim,
        experience_similarity=exp_sim,
        overall_similarity=overall_sim,
    )
