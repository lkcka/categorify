"""
Сводный пайплайн категоризации — единственное место, где фиксируется
порядок слоёв и логика принятия решения "куда отправить транзакцию".

Порядок:
  1. Правила (детерминированные, быстрые, 100% объяснимые) — rules.py
  2. ML-классификатор (TF-IDF + LogisticRegression) — только если правила
     не дали ответа и уверенность >= порога с шага обучения — ml_model.py
  3. LLM-фоллбэк — только для строк, где НЕ уверен ML. Батчируется,
     кэшируется, по умолчанию работает в честном MOCK-режиме (llm/).
  4. Честный fallback "Прочее" — если ни один слой не уверен.

Дизайн-решение про "MOCK по умолчанию": на уровне ЭТОГО класса
llm_client=None означает "слой физически выключен" (это нужно для
регресс-тестов и юнит-тестов без побочных эффектов и печати
предупреждений). А вот выбор "MOCK — дефолт из коробки" сделан на уровне
точек входа (CLI evaluate.py / categorize.py, позже — REST API), которые
сами создают MockLLMClient(), если пользователь ничего не настроил. Это
стандартное разделение "чистый механизм / дефолт продукта".

Батчинг: categorize_batch() — основной метод для реальной обработки
выписки (используется evaluate.py, categorize.py и будущим REST-
эндпоинтом). Одиночный categorize() — тонкая обёртка над батчем из
одного элемента, подходит для API "категоризировать одну транзакцию",
но неэффективна как основной путь при больших объёмах — так и задумано.

Кэш по (нормализованное описание, MCC) экономит и время, и, что более
важно для LLM, реальные токены/деньги — один и тот же мерчант в выписке
встречается многократно, а вызывается LLM/ML/правила только один раз.
"""
from dataclasses import dataclass
from pathlib import Path

from expense_categorizer.categories import FALLBACK_CATEGORY
from expense_categorizer.rules import RuleEngine
from expense_categorizer.ml_model import MLCategorizer
from expense_categorizer.normalize import normalize
from expense_categorizer.llm.base import LLMClient, LLMItem

DEFAULT_ML_MODEL_PATH = "models/ml_classifier.joblib"


@dataclass(frozen=True)
class CategorizationResult:
    category: str
    confidence: float
    source: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CategorizationPipeline:
    def __init__(self, ml_model_path: str | Path | None = DEFAULT_ML_MODEL_PATH,
                 llm_client: LLMClient | None = None,
                 llm_confidence_threshold: float = 0.55,
                 ml_trust_threshold_override: float | None = None,
                 cache_size: int = 20_000):
        self.rule_engine = RuleEngine()

        self.ml_categorizer: MLCategorizer | None = None
        if ml_model_path is not None and Path(ml_model_path).exists():
            self.ml_categorizer = MLCategorizer.from_file(ml_model_path)
        elif ml_model_path is not None:
            print(f"[pipeline] ML-модель не найдена по пути '{ml_model_path}' — "
                  f"работаю в деградированном режиме rules-only "
                  f"(запустите: python -m expense_categorizer.train_ml).")

        self.llm_client = llm_client
        self.llm_confidence_threshold = llm_confidence_threshold
        # Порог, начиная с которого ML-ответ принимается БЕЗ передачи в LLM
        self.ml_trust_threshold_override = ml_trust_threshold_override

        self._cache: dict[tuple[str, str], CategorizationResult] = {}
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_batches_called = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def categorize(self, description: str, mcc=None, direction: str | None = None) -> CategorizationResult:
        """Одиночный вызов — обёртка над батчем из 1 элемента. Подходит для
        API 'категоризировать одну транзакцию', но НЕ является эффективным
        основным путём: LLM-слой при таком использовании не может
        батчироваться между разными вызовами categorize()."""
        return self.categorize_batch([{"description": description, "mcc": mcc}])[0]

    def categorize_batch(self, rows: list[dict], llm_batch_size: int = 10) -> list[CategorizationResult]:
        n = len(rows)
        results: list[CategorizationResult | None] = [None] * n

        # Ключ -> список индексов исходных строк с этим же ключом (дедупликация
        # перед LLM: одинаковый мерчант встречается в выписке многократно).
        idx_by_cache_key: dict[tuple[str, str], list[int]] = {}
        llm_queue_keys: list[tuple[str, str]] = []
        llm_queue_seen: set[tuple[str, str]] = set()

        for idx, row in enumerate(rows):
            description = row.get("description") or ""
            mcc = row.get("mcc")
            cache_key = (normalize(description), self.rule_engine.normalize_mcc(mcc) or "")

            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                results[idx] = cached
                continue
            self.cache_misses += 1

            rule_result = self.rule_engine.classify(description, mcc)
            if rule_result is not None:
                res = CategorizationResult(rule_result.category, rule_result.confidence, rule_result.source)
                results[idx] = res
                self._store_cache(cache_key, res)
                continue

            if self.ml_categorizer is not None:
                category, confidence = self.ml_categorizer.predict(description)
                trust_threshold = (self.ml_trust_threshold_override
                                    if self.ml_trust_threshold_override is not None
                                    else self.ml_categorizer.bundle.threshold)
                if confidence >= trust_threshold:
                    res = CategorizationResult(category, confidence, "ml")
                    results[idx] = res
                    self._store_cache(cache_key, res)
                    continue

                if self.llm_client is None:
                    # LLM-слой выключен -> честный fallback, как на шаге 5
                    res = CategorizationResult(FALLBACK_CATEGORY, confidence, "ml_low_confidence")
                    results[idx] = res
                    self._store_cache(cache_key, res)
                    continue

                # ML не уверен, LLM подключен -> кандидат в батч
                idx_by_cache_key.setdefault(cache_key, []).append(idx)
                if cache_key not in llm_queue_seen:
                    llm_queue_seen.add(cache_key)
                    llm_queue_keys.append(cache_key)
                continue

            # ML вообще не подключен (rules-only деградированный режим)
            res = CategorizationResult(FALLBACK_CATEGORY, 0.0, "fallback_no_rule")
            results[idx] = res
            self._store_cache(cache_key, res)

        if llm_queue_keys:
            self._resolve_llm_queue(rows, llm_queue_keys, idx_by_cache_key, results, llm_batch_size)

        # Защитный рубеж: если LLM-клиент вернул не все id (ошибка парсинга и
        # т.п.), не должно остаться None — честный fallback, а не падение.
        for idx, res in enumerate(results):
            if res is None:
                results[idx] = CategorizationResult(FALLBACK_CATEGORY, 0.0, "llm_missing_response")

        return results

    def _resolve_llm_queue(self, rows, llm_queue_keys, idx_by_cache_key, results, llm_batch_size):
        id_to_key: dict[str, tuple[str, str]] = {}
        unique_items: list[LLMItem] = []
        for pos, cache_key in enumerate(llm_queue_keys):
            first_idx = idx_by_cache_key[cache_key][0]
            row = rows[first_idx]
            safe_id = str(pos)
            id_to_key[safe_id] = cache_key
            # normalize_mcc корректно обрабатывает NaN/float/строку -> "5411"
            # или None. Без этого pandas NaN (bool(nan) == True в Python!)
            # утекал в промпт как буквальный текст "MCC=nan", вводя модель
            # в заблуждение.
            unique_items.append(LLMItem(item_id=safe_id, description=row.get("description") or "",
                                         mcc=self.rule_engine.normalize_mcc(row.get("mcc"))))

        for batch_start in range(0, len(unique_items), llm_batch_size):
            batch = unique_items[batch_start: batch_start + llm_batch_size]
            responses = self.llm_client.categorize_batch(batch)
            self.llm_batches_called += 1

            for resp in responses:
                cache_key = id_to_key.get(resp.item_id)
                if cache_key is None:
                    continue
                self.total_prompt_tokens += resp.prompt_tokens
                self.total_completion_tokens += resp.completion_tokens

                if resp.confidence >= self.llm_confidence_threshold:
                    res = CategorizationResult(resp.category, resp.confidence, "llm",
                                                resp.prompt_tokens, resp.completion_tokens)
                else:
                    res = CategorizationResult(FALLBACK_CATEGORY, resp.confidence, "llm_low_confidence",
                                                resp.prompt_tokens, resp.completion_tokens)

                self._store_cache(cache_key, res)
                for idx in idx_by_cache_key[cache_key]:
                    results[idx] = res

    def _store_cache(self, key: tuple[str, str], result: CategorizationResult) -> None:
        if len(self._cache) >= self._cache_size:
            self._cache.clear()  # простая политика вытеснения для учебного прототипа
        self._cache[key] = result