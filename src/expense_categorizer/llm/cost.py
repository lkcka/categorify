"""
Оценка гипотетической $ стоимости LLM-фоллбэка на платном API.

Проект использует локальный Ollama ($0), но заказчику важно понимать
порядок затрат при переходе на облачный API (например, если локальных
ресурсов для inference не хватит в проде). Цены ниже — ориентировочные
(уровень GPT-4o-mini), их нужно свежесверять на
сайте провайдера перед реальным использованием — здесь это ТОЛЬКО оценка
порядка величины, а не финансовое обязательство.
"""
from dataclasses import dataclass

PRICE_PER_1K_PROMPT_USD = 0.00015      # $0.15 / 1M input tokens
PRICE_PER_1K_COMPLETION_USD = 0.0006   # $0.60 / 1M output tokens


@dataclass
class CostEstimate:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @classmethod
    def from_tokens(cls, prompt_tokens: int, completion_tokens: int) -> "CostEstimate":
        cost = (prompt_tokens / 1000 * PRICE_PER_1K_PROMPT_USD +
                completion_tokens / 1000 * PRICE_PER_1K_COMPLETION_USD)
        return cls(prompt_tokens, completion_tokens, cost)


def project_cost(prompt_tokens: int, completion_tokens: int,
                  n_transactions_seen: int, n_transactions_target: int) -> CostEstimate:
    """Линейная экстраполяция: 'если на N транзакций ушло X токенов, сколько
    будет стоить обработать target транзакций'. Предполагает похожую долю
    строк, доходящих до LLM, и похожий кэш-хитрейт — упрощение, явно
    обозначенное как оценка, а не гарантия."""
    if n_transactions_seen == 0:
        return CostEstimate(0, 0, 0.0)
    scale = n_transactions_target / n_transactions_seen
    return CostEstimate.from_tokens(round(prompt_tokens * scale), round(completion_tokens * scale))