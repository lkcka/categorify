"""
Диагностический скрипт: смотрим на СЫРОЙ ответ Ollama для реального
батча "неуверенных" транзакций, чтобы понять, почему парсинг JSON
даёт пустой результат (подозрение: модель заворачивает массив в
объект вместо плоского top-level array).

Запуск: python scripts/debug_llm.py
"""
import json
import httpx

from expense_categorizer.llm.base import LLMItem
from expense_categorizer.llm.prompts import build_batch_prompt

items = [
    LLMItem(item_id="0", description="POKUPKA AROMAMANIYA VORONEZH", mcc=""),
    LLMItem(item_id="1", description="POS ZOLOTOE YABLOKO", mcc=""),
    LLMItem(item_id="2", description="SLATA SANKT-PETERBURG RUS", mcc=""),
    LLMItem(item_id="3", description="BATON SAMARA RUS", mcc=""),
    LLMItem(item_id="4", description="VERNYY*348690", mcc=""),
    LLMItem(item_id="5", description="TAKSOVICHKOF*27965", mcc=""),
    LLMItem(item_id="6", description="KVEST-HAUS", mcc=""),
    LLMItem(item_id="7", description="OPLAA KOFEPORT", mcc=""),
    LLMItem(item_id="8", description="OKKO", mcc=""),
    LLMItem(item_id="9", description="VELOBAYK HCELYABINSK RUS", mcc=""),
]

prompt = build_batch_prompt(items)
print("=== PROMPT ===")
print(prompt)
print("\n=== ВЫЗЫВАЮ OLLAMA ===")

resp = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    },
    timeout=180.0,
)
resp.raise_for_status()
data = resp.json()

print("\n=== СЫРОЙ 'response' (то, что мы пытаемся распарсить как JSON) ===")
print(repr(data["response"]))

print("\n=== prompt_eval_count / eval_count ===")
print(data.get("prompt_eval_count"), data.get("eval_count"))

print("\n=== Попытка json.loads ===")
try:
    parsed = json.loads(data["response"])
    print(f"Тип: {type(parsed)}")
    print(parsed)
except Exception as e:
    print(f"Ошибка парсинга: {e!r}")

from expense_categorizer.llm.ollama_client import _flatten_llm_entries
print("\n=== После _flatten_llm_entries ===")
print(_flatten_llm_entries(parsed))