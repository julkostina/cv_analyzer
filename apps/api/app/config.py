from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
_API_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _API_DIR / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), case_sensitive=False, env_file_encoding="utf-8")
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    openai_api_key: str = ""
    gemini_api_key: str = ""
    environment: str = "development"
    # Comma-separated frontend origin(s) allowed to call this API in production (e.g. https://your-app.vercel.app).
    allowed_origins: str = ""

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    # Ollama model for development (ollama pull …). 8B+ models follow JSON schema more reliably than 3B.
    ollama_model: str = "llama3.1:8b"
    # Context window (tokens). Match model limit (8192 for llama3.1:8b); larger values are clamped by Ollama.
    ollama_num_ctx: int = 8192
    # Retries when Ollama returns empty or unparseable JSON (development). Use 1–2 for faster runs.
    ollama_json_max_attempts: int = 2
    # Cap tokens for JSON extraction (structured fields + short summary draft).
    ollama_extract_num_predict: int = 2400
    # Cap tokens for coaching-summary rewrite (final «Підсумок та поради»).
    ollama_coaching_num_predict: int = 2048

    # Max CV characters sent to the LLM (rest is truncated). 0 = no truncation (full CV sent; needs larger context).
    max_cv_chars_for_llm: int = 12000

    max_upload_size_mb: int = 10

    use_semantic_matching: bool = True
    # Kernel SHAP + LIME over skill/experience features for match_score (see match_explainer.py).
    use_match_explainers: bool = True
    explainer_max_skills: int = 8
    explainer_max_experience: int = 5
    explainer_top_features: int = 5
    explainer_shap_samples: int = 12
    explainer_lime_samples: int = 12
    # Extra LLM call for semantic_score_narrative (removed from UI; keep off for speed).
    use_llm_semantic_narrative: bool = False
    # Rewrite analysis.summary with score drivers (min 10 sentences, «ви», what raises/lowers match).
    use_llm_coaching_summary: bool = True
    # Weights for embedding cosine components (normalized to sum 1.0 before scoring). See semantic_matcher.py.
    semantic_weights_skills: float = 0.5
    semantic_weights_experience: float = 0.3
    semantic_weights_overall: float = 0.2
    embedding_provider: str = "sentence_transformers"
    pdf_font_path: str = ""

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

settings = Settings()
