# Categorify

Учебный прототип автоматической категоризации транзакций банковской выписки (Россия).

Пайплайн из трёх слоёв:

1. **Правила** — мерчанты, MCC, ключевые слова доходов/переводов, ИП-подсказки  
2. **ML** — TF-IDF + LogisticRegression (только при достаточной уверенности)  
3. **LLM-фоллбэк** — Ollama или честный mock; иначе категория **«Прочее»**

Интерфейс — CLI (и Docker). Данные синтетические.

## Требования

- Python 3.10+
- (опционально) [Ollama](https://ollama.com/) для реального LLM-слоя
- (опционально) Docker

## Установка

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -e .
```

## Быстрый старт

```bash
# 1. Синтетические данные
python -m expense_categorizer.generator

# 2. Golden-набор для оценки
python -m expense_categorizer.golden

# 3. Обучение ML-модели
python -m expense_categorizer.train_ml

# 4. Категоризация демо-выписки
python -m expense_categorizer.categorize --llm-backend off

# 5. Оценка качества на golden
python -m expense_categorizer.evaluate --llm-backend off
```

Результат категоризации: `data/sample_statement_categorized.csv`  
(исходные колонки + `category`, `confidence`, `source`).

## CLI

| Команда | Назначение |
|---------|------------|
| `python -m expense_categorizer.generator` | `data/ml_train.csv`, `data/sample_statement.csv` |
| `python -m expense_categorizer.golden` | `data/golden.csv` (holdout-мерчанты) |
| `python -m expense_categorizer.train_ml` | `models/ml_classifier.joblib` |
| `python -m expense_categorizer.categorize` | CSV-выписка → CSV с категориями |
| `python -m expense_categorizer.evaluate` | метрики на golden + `reports/` |
| `python -m expense_categorizer.data_stats` | распределение категорий в CSV |

После `pip install -e .` доступны и entry points вида `expense-categorizer-categorize`.

### Категоризация своей выписки

Входной CSV должен содержать колонку `description`. Опционально — `mcc` и любые другие поля (они сохраняются).

```bash
python -m expense_categorizer.categorize \
  -i data/sample_statement.csv \
  -o data/out.csv \
  --llm-backend off
```

Полезные флаги (те же, что у `evaluate`):

- `--llm-backend off|mock|ollama` — слой LLM (`mock` по умолчанию)
- `--no-ml` — только правила + LLM/fallback
- `--ml-model` — путь к `.joblib`
- `--llm-model`, `--ollama-host` — для Ollama

### Оценка качества

```bash
python -m expense_categorizer.evaluate --llm-backend off
```

Пишет `reports/errors.csv` и `reports/confusion_matrix.csv`.

## Docker

```bash
docker build -t expense-categorizer .

# Оценка без LLM (по умолчанию)
docker run --rm -v "%cd%/reports:/app/reports" expense-categorizer

# Категоризация демо-выписки
docker run --rm \
  -v "%cd%/data:/app/data" \
  expense-categorizer categorize --llm-backend off
```

С Ollama через Compose:

```bash
docker compose --profile ollama up -d ollama
docker compose --profile ollama run --rm app \
  evaluate --llm-backend ollama --ollama-host http://ollama:11434
```

## Структура

```
src/expense_categorizer/   # ядро и CLI
  pipeline.py              # rules → ML → LLM → fallback
  categorize.py            # CLI: CSV → CSV
  evaluate.py              # CLI: метрики на golden
  rules.py, ml_model.py, llm/
data/                      # CSV (генерируются)
models/                    # обученная ML-модель
reports/                   # отчёты evaluate
```

## Категории

Продукты, Кафе и рестораны, Транспорт, Такси, Связь и интернет, Подписки и сервисы, Здоровье, Красота и уход, Одежда и обувь, Дом и ремонт, Развлечения, Образование, Путешествия, Переводы, Прочие доходы, Прочее.

## Замечания

- Без обученной модели пайплайн работает в режиме rules-only (с предупреждением).
- `mock` LLM не угадывает категории: неуверенные строки честно уходят в «Прочее».
- Цель accuracy > 95% на holdout **без** реального LLM недостижима по задумке — хвост как раз для LLM-слоя.
