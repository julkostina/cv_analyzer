# CV Analyzer API

FastAPI бекенд для аналізу резюме з використанням LangChain.

## Налаштування

1. Створіть віртуальне середовище:
   
   python -m venv .venv
   source .venv/bin/activate  # На Windows: .venv\Scripts\activate
   2. Встановіть залежності:
   
   pip install -e .
   3. Скопіюйте змінні оточення:
   
   cp .env.example .env
   # Відредагуйте .env з вашими API ключами
   4. Запустіть сервер:
   
   uvicorn app.main:app --reload

## Швидкість аналізу (Ollama)

Типовий час займають **2 проходи LLM** (витяг JSON + підсумок з балами) і **SHAP/LIME** над ембеддингами.

У `apps/api/.env` для швидших локальних запусків:

- `OLLAMA_JSON_MAX_ATTEMPTS=1` або `2`
- `MAX_CV_CHARS_FOR_LLM=12000` (не `0`, якщо резюме дуже довге)
- `USE_LLM_SEMANTIC_NARRATIVE=false` (за замовчуванням)
- `USE_MATCH_EXPLAINERS=false` — найбільший виграш; підсумок лишиться без SHAP/LIME у тексті
- `EXPLAINER_SHAP_SAMPLES=8` та `EXPLAINER_LIME_SAMPLES=8` — компроміс швидкість/деталі

Після зміни `.env` перезапустіть API.

## Вимірювання часу відповіді (NFR-01)

Локальні бенчмарки (Ollama/Gemini має бути доступний):

```bash
cd apps/api
RUN_LIVE_BENCHMARKS=1 pytest tests/test_response_time.py -v -s
```

Опційно: `BENCHMARK_RUNS=3` (медіана), `NFR_MAX_SECONDS=90` (перевірка порогу після оновлення NFR).

Без `RUN_LIVE_BENCHMARKS` у звичайному `pytest` live-тести пропускаються; завжди виконується лише швидкий тест накладних витрат ендпоінта та (за наявності моделі) семантичного кроку.
