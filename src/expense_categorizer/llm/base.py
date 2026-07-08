"""
Абстракция LLM-клиента для фоллбэк-слоя категоризации.

Любая реализация (mock, Ollama, платный API) обязана уметь принимать
БАТЧ транзакций за один вызов — это прямое требование по стоимости и
производительности (см. README, раздел "Стоимость"). Одиночные вызовы
"1 транзакция = 1 запрос" не поддерживаются намеренно: при реальном
объёме выписки (сотни-тысячи строк) это дорого и медленно.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMItem:
    """Единица батча — то, что реально уходит в промпт.
    Намеренно НЕ содержит ground truth (в проде его и нет)."""
    item_id: str
    description: str
    mcc: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    item_id: str
    category: str
    confidence: float     # 0..1, самооценка модели
    prompt_tokens: int
    completion_tokens: int
    raw_source: str        # "llm_mock" | "llm_ollama:<model>" | "llm_ollama_error:<model>"


class LLMClient(ABC):
    @abstractmethod
    def categorize_batch(self, items: list[LLMItem]) -> list[LLMResponse]:
        ...