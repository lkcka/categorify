"""
Реальный LLM-клиент через локальный Ollama (http://localhost:11434).

ОПЦИОНАЛЬНЫЙ компонент: требует установленного Ollama и скачанной модели
(`ollama pull qwen2.5:3b`). Без него пайплайн работает в MOCK-режиме
(mock_client.py) и не падает — это специально разделено, чтобы третье
лицо могло воспроизвести проект по README без установки LLM вообще.

Формат ответа: просим строгий JSON-массив (format="json" в Ollama API),
парсим с защитой "одна сломанная транзакция не должна убить весь батч".

Подсчёт токенов: Ollama возвращает точные prompt_eval_count/eval_count
в самом ответе — используем их напрямую, без сторонних токенизаторов
(что было бы неточно для конкретной модели).
"""
import json
import logging

import httpx

from expense_categorizer.categories import CATEGORIES, FALLBACK_CATEGORY
from expense_categorizer.llm.base import LLMClient, LLMItem, LLMResponse
from expense_categorizer.llm.prompts import build_batch_prompt

logger = logging.getLogger(__name__)


def _flatten_llm_entries(parsed) -> list[dict]:
    """
    Нормализует РЕАЛЬНО НАБЛЮДАВШИЕСЯ формы ответа маленькой локальной
    модели к плоскому списку {'id', 'category', 'confidence'}.

    Ollama format="json" гарантирует валидный JSON, но НЕ гарантирует
    конкретную схему (плоский массив). На практике qwen2.5:3b иногда
    вместо запрошенного [{...}, {...}] возвращает объект вида
    {"0": [{...}], "1": [{...}]} — тот же контент в другой обёртке.
    Здесь мы разворачиваем известные варианты, не теряя данные, вместо
    того чтобы молча схлопывать их в пустой список (что раньше приводило
    к неотличимости от полного отказа LLM).
    """
    if isinstance(parsed, list):
        return [e for e in parsed if isinstance(e, dict)]

    if isinstance(parsed, dict):
        for wrapper_key in ("results", "items", "transactions", "response", "data"):
            if wrapper_key in parsed and isinstance(parsed[wrapper_key], list):
                return _flatten_llm_entries(parsed[wrapper_key])

        # Наблюдавшийся реальный случай: {"<id>": {...}} или {"<id>": [{...}]}
        flattened, looks_like_id_map = [], True
        for key, value in parsed.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("id", key)
                flattened.append(entry)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                entry = dict(value[0])
                entry.setdefault("id", key)
                flattened.append(entry)
            else:
                looks_like_id_map = False
        if looks_like_id_map and flattened:
            return flattened

    return []

class OllamaLLMClient(LLMClient):
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                 timeout: float = 180.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def categorize_batch(self, items: list[LLMItem]) -> list[LLMResponse]:
        if not items:
            return []

        prompt = build_batch_prompt(items)
        try:
            resp = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            prompt_tokens = int(data.get("prompt_eval_count", 0))
            completion_tokens = int(data.get("eval_count", 0))
            raw_parsed = json.loads(data["response"])
            entries = _flatten_llm_entries(raw_parsed)
            if len(entries) < len(items):
                logger.warning(
                    "[llm] Ollama вернула %d элементов вместо ожидаемых %d "
                    "(батч частично не распознан, недостающие уйдут в честный "
                    "fallback). Сырой ответ (обрезан): %.300s",
                    len(entries), len(items), data["response"],
                )
        except Exception as exc:
            logger.warning(f"[llm] Ошибка вызова Ollama ({exc!r}) — батч из {len(items)} "
                   f"строк уходит в честный fallback 'Прочее'.")
            return [
                LLMResponse(item_id=i.item_id, category=FALLBACK_CATEGORY, confidence=0.0,
                            prompt_tokens=0, completion_tokens=0,
                            raw_source=f"llm_ollama_error:{self.model}")
                for i in items
            ]

        by_id: dict[str, tuple[str, float]] = {}
        for entry in entries:
            item_id = str(entry.get("id", ""))
            category = entry.get("category", FALLBACK_CATEGORY)
            if category not in CATEGORIES:
                category = FALLBACK_CATEGORY
            try:
                confidence = float(entry.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            by_id[item_id] = (category, confidence)

        # Токены делим поровну между элементами батча — приблизительный учёт
        # для оценки $ на транзакцию; точнее без отдельного токенизатора
        # конкретной модели недостижимо, это явное упрощение.
        n = len(items)
        per_item_prompt = prompt_tokens // n if n else 0
        per_item_completion = completion_tokens // n if n else 0

        results = []
        for i in items:
            category, confidence = by_id.get(i.item_id, (FALLBACK_CATEGORY, 0.0))
            results.append(LLMResponse(
                item_id=i.item_id, category=category, confidence=confidence,
                prompt_tokens=per_item_prompt, completion_tokens=per_item_completion,
                raw_source=f"llm_ollama:{self.model}",
            ))
        return results