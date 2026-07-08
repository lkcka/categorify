"""
Честный "degraded mode" по умолчанию.

Это НЕ заглушка "для галочки" — это прямая реализация требования ТЗ
"поведение без доступа к модели": если реальная LLM недоступна (нет
Ollama, нет API-ключа), сервис не должен падать и не должен выдавать
случайные/сфабрикованные категории. Он честно возвращает "Прочее" с
нулевой уверенностью — то есть ведёт себя так же, как если бы LLM-слоя
не было вовсе (см. pipeline.py: confidence=0.0 всегда ниже порога
принятия ответа, поэтому 'угадать' категорию мок физически не может).
"""
from expense_categorizer.categories import FALLBACK_CATEGORY
from expense_categorizer.llm.base import LLMClient, LLMItem, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(self):
        self._warned = False

    def categorize_batch(self, items: list[LLMItem]) -> list[LLMResponse]:
        if not self._warned:
            print("[llm] MOCK-режим: реальная LLM не подключена. Неуверенные "
                  "транзакции честно помечаются как 'Прочее' (не гадаем). "
                  "Для реального LLM-фоллбэка используйте --llm-backend ollama "
                  "(см. README).")
            self._warned = True
        return [
            LLMResponse(item_id=i.item_id, category=FALLBACK_CATEGORY, confidence=0.0,
                        prompt_tokens=0, completion_tokens=0, raw_source="llm_mock")
            for i in items
        ]