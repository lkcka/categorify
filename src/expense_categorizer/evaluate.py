"""
CLI: python -m expense_categorizer.evaluate

Прогоняет ТЕКУЩИЙ пайплайн по golden-набору и печатает честный отчёт:
accuracy, разбивку по источнику решения, время обработки и (если LLM
активен) затраты токенов + гипотетическую $ стоимость на платном API.

--llm-backend управляет ТОЛЬКО способом эмуляции/вызова LLM-слоя:
  off    — слой выключен полностью (эквивалент шага 5, для регресс-теста)
  mock   — честная заглушка "модель недоступна" (ПО УМОЛЧАНИЮ)
  ollama — реальный локальный вызов через Ollama (требует установки)
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from expense_categorizer.pipeline import CategorizationPipeline, DEFAULT_ML_MODEL_PATH
from expense_categorizer.metrics import evaluate
from expense_categorizer.llm.mock_client import MockLLMClient
from expense_categorizer.llm.ollama_client import OllamaLLMClient
from expense_categorizer.llm.cost import CostEstimate, project_cost


def build_llm_client(backend: str, model: str, host: str):
    if backend == "off":
        return None
    if backend == "mock":
        return MockLLMClient()
    if backend == "ollama":
        return OllamaLLMClient(model=model, host=host)
    raise ValueError(f"Неизвестный --llm-backend: {backend}")


def run_pipeline(df: pd.DataFrame, pipeline: CategorizationPipeline, llm_batch_size: int):
    rows = df.to_dict("records")
    t0 = time.perf_counter()
    results = pipeline.categorize_batch(rows, llm_batch_size=llm_batch_size)
    elapsed = time.perf_counter() - t0
    preds = pd.DataFrame([
        {"pred_category": r.category, "confidence": r.confidence, "source": r.source}
        for r in results
    ])
    return preds, elapsed


def main():
    parser = argparse.ArgumentParser(description="Оценка качества категоризации на golden-наборе")
    parser.add_argument("--golden", default="data/golden.csv")
    parser.add_argument("--errors-out", default="reports/errors.csv")
    parser.add_argument("--confusion-out", default="reports/confusion_matrix.csv")
    parser.add_argument("--no-ml", action="store_true",
                         help="Отключить ML-слой (сравнение с baseline)")
    parser.add_argument("--llm-backend", choices=["off", "mock", "ollama"], default="mock",
                         help="off=выключен, mock=честная заглушка (по умолчанию), "
                              "ollama=реальный локальный вызов")
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--llm-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--ml-trust-threshold", type=float, default=None,
                         help="Порог доверия ML-ответу БЕЗ передачи в LLM")
    args = parser.parse_args()

    golden_df = pd.read_csv(args.golden)
    llm_client = build_llm_client(args.llm_backend, args.llm_model, args.ollama_host)
    pipeline = CategorizationPipeline(
        ml_model_path=None if args.no_ml else DEFAULT_ML_MODEL_PATH,
        llm_client=llm_client,
        llm_confidence_threshold=args.llm_confidence_threshold,
    )

    predictions, elapsed = run_pipeline(golden_df, pipeline, args.llm_batch_size)
    report = evaluate(golden_df, predictions)

    print(f"\n=== Accuracy: {report.accuracy:.4f} ({report.n_correct}/{report.n_total}) ===\n")
    print("--- Classification report (по категориям) ---")
    print(report.classification_report_text)
    print("--- Разбивка по источнику решения ---")
    print(report.source_breakdown)

    n = len(golden_df)
    print("\n--- Производительность ---")
    if elapsed > 0:
        print(f"Всего строк: {n}, время обработки: {elapsed:.2f} с ({n / elapsed:.1f} строк/сек)")
    else:
        print(f"Всего строк: {n}")
    print(f"Кэш пайплайна: hits={pipeline.cache_hits}, misses={pipeline.cache_misses}")
    print(f"LLM-батчей вызвано: {pipeline.llm_batches_called}")

    if pipeline.total_prompt_tokens or pipeline.total_completion_tokens:
        print("\n--- Токены и гипотетическая стоимость LLM-слоя ---")
        print(f"prompt_tokens={pipeline.total_prompt_tokens}, "
              f"completion_tokens={pipeline.total_completion_tokens}")
        est_now = CostEstimate.from_tokens(pipeline.total_prompt_tokens, pipeline.total_completion_tokens)
        print(f"Гипотетическая стоимость на платном API (уровня GPT-4o-mini) "
              f"для этих {n} строк: ${est_now.cost_usd:.4f}")
        for target in (100, 1000):
            proj = project_cost(pipeline.total_prompt_tokens, pipeline.total_completion_tokens, n, target)
            print(f"  Экстраполяция на {target} транзакций (при похожей доле, доходящей до LLM): "
                  f"${proj.cost_usd:.4f}")
    elif args.llm_backend == "mock":
        print("\n[i] LLM-слой в MOCK-режиме: реальных токенов не потрачено. "
              "Неуверенные транзакции честно помечены как 'Прочее' — это ожидаемое "
              "поведение degraded mode, а не потеря функциональности.")

    out_dir = Path(args.errors_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report.errors_df.to_csv(args.errors_out, index=False)
    report.confusion_df.to_csv(args.confusion_out)
    print(f"\nОшибки -> {args.errors_out} ({len(report.errors_df)} строк)")
    print(f"Матрица ошибок -> {args.confusion_out}")

    if report.accuracy > 0.95:
        print("\nПорог accuracy > 95% достигнут.")
    else:
        print(f"\nПорог > 95% НЕ достигнут (сейчас {report.accuracy:.2%}).")


if __name__ == "__main__":
    main()