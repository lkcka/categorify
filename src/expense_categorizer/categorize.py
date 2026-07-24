"""
CLI: python -m expense_categorizer.categorize

Прогоняет пайплайн по CSV-выписке без меток и пишет результат в новый CSV:
исходные колонки + category / confidence / source.

Типичный вход — data/sample_statement.csv (см. generator.build_sample_statement).
Обязательная колонка: description. Опционально: mcc (и любые другие поля —
они сохраняются как есть).

--llm-backend управляет LLM-слоем так же, как в evaluate:
  off    — слой выключен
  mock   — честная заглушка (ПО УМОЛЧАНИЮ)
  ollama — локальный Ollama
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from expense_categorizer.pipeline import CategorizationPipeline, DEFAULT_ML_MODEL_PATH
from expense_categorizer.llm.mock_client import MockLLMClient
from expense_categorizer.llm.ollama_client import OllamaLLMClient


def build_llm_client(backend: str, model: str, host: str):
    if backend == "off":
        return None
    if backend == "mock":
        return MockLLMClient()
    if backend == "ollama":
        return OllamaLLMClient(model=model, host=host)
    raise ValueError(f"Неизвестный --llm-backend: {backend}")


def categorize_dataframe(df: pd.DataFrame, pipeline: CategorizationPipeline,
                         llm_batch_size: int) -> tuple[pd.DataFrame, float]:
    if "description" not in df.columns:
        raise ValueError(
            "Во входном CSV нужна колонка 'description'. "
            f"Найдены колонки: {list(df.columns)}"
        )

    rows = df.to_dict("records")
    t0 = time.perf_counter()
    results = pipeline.categorize_batch(rows, llm_batch_size=llm_batch_size)
    elapsed = time.perf_counter() - t0

    out = df.copy()
    out["category"] = [r.category for r in results]
    out["confidence"] = [r.confidence for r in results]
    out["source"] = [r.source for r in results]
    return out, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Категоризация банковской выписки (CSV → CSV)"
    )
    parser.add_argument("--input", "-i", default="data/sample_statement.csv",
                        help="Входной CSV с колонкой description")
    parser.add_argument("--output", "-o", default="data/sample_statement_categorized.csv",
                        help="Куда записать результат")
    parser.add_argument("--no-ml", action="store_true",
                        help="Отключить ML-слой (только правила + LLM/fallback)")
    parser.add_argument("--llm-backend", choices=["off", "mock", "ollama"], default="mock",
                        help="off=выключен, mock=честная заглушка (по умолчанию), "
                             "ollama=реальный локальный вызов")
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--llm-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--ml-trust-threshold", type=float, default=None,
                        help="Порог доверия ML-ответу БЕЗ передачи в LLM")
    parser.add_argument("--ml-model", default=DEFAULT_ML_MODEL_PATH,
                        help="Путь к обученной ML-модели (.joblib)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Файл не найден: {input_path}\n"
            f"Сначала сгенерируйте данные: python -m expense_categorizer.generator"
        )

    df = pd.read_csv(input_path)
    llm_client = build_llm_client(args.llm_backend, args.llm_model, args.ollama_host)
    pipeline = CategorizationPipeline(
        ml_model_path=None if args.no_ml else args.ml_model,
        llm_client=llm_client,
        llm_confidence_threshold=args.llm_confidence_threshold,
        ml_trust_threshold_override=args.ml_trust_threshold,
    )

    result_df, elapsed = categorize_dataframe(df, pipeline, args.llm_batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    n = len(result_df)
    print(f"Категоризировано строк: {n}")
    if elapsed > 0:
        print(f"Время: {elapsed:.2f} с ({n / elapsed:.1f} строк/сек)")
    print(f"Кэш: hits={pipeline.cache_hits}, misses={pipeline.cache_misses}")
    print(f"LLM-батчей: {pipeline.llm_batches_called}")
    print("\n--- Разбивка по источнику ---")
    print(result_df["source"].value_counts().to_string())
    print("\n--- Разбивка по категориям ---")
    print(result_df["category"].value_counts().to_string())
    if args.llm_backend == "mock":
        print("\n[i] LLM в MOCK-режиме: неуверенные строки честно попадут в 'Прочее'.")
    print(f"\nРезультат -> {out_path}")


if __name__ == "__main__":
    main()
