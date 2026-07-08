"""
Генератор синтетических транзакций.

Два публичных билдера:
  build_ml_train()        -> обучающая выборка для ML-фоллбэка (шаг 5).
  build_sample_statement()-> демо-CSV без меток для API/CLI.

Оба используют ТОЛЬКО merchant_mode="train" — holdout-бренды сюда
принципиально не попадают (см. golden.py и catalog.py).
"""
import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from expense_categorizer.categories import CATEGORIES
from expense_categorizer.catalog import (
    MERCHANTS, CATEGORY_META, CATEGORY_WEIGHTS, IP_HINTS,
    AMBIGUOUS_TEMPLATES, TRANSFER_TEMPLATES, INCOME_TEMPLATES,
    CITIES, COMPANY_NAMES,
)
from expense_categorizer.textgen import (
    render_retail_description, render_service_description, render_amount,
    random_person_name, maybe_add_noise,
)

ANCHOR_DATE = date(2024, 6, 30)  # фиксировано для воспроизводимости


def _random_date(rng: random.Random) -> date:
    return ANCHOR_DATE - timedelta(days=rng.randint(0, 90))


def _sample_retail_or_service(category, rng, noise_rate=0.0,
                               merchant_mode="train", force_ip=False):
    meta = CATEGORY_META[category]
    hints = IP_HINTS.get(category)
    use_ip = force_ip or (hints is not None and merchant_mode != "holdout"
                           and rng.random() < 0.12)

    if use_ip and hints:
        name = random_person_name(rng)
        description = f"ИП {name} {rng.choice(hints)}"
        merchant_display, is_holdout = name, False
    else:
        bucket = "holdout" if merchant_mode == "holdout" else "train"
        merchant_display = rng.choice(MERCHANTS[category][bucket])
        is_holdout = bucket == "holdout"
        city = rng.choice(CITIES)
        description = (render_service_description(merchant_display, rng)
                        if meta["pool"] == "service"
                        else render_retail_description(merchant_display, city, rng))

    description = maybe_add_noise(description, rng, noise_rate)
    amount = render_amount(*meta["amount_range"], rng)
    mcc = rng.choice(meta["mcc"]) if meta["mcc"] and rng.random() < 0.6 else ""
    return description, amount, mcc, "debit", merchant_display, is_holdout


def _sample_transfer(rng):
    tmpl = rng.choice(TRANSFER_TEMPLATES)
    name = random_person_name(rng)
    phone = f"+7 9{rng.randint(10,99)} {rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(10,99)}"
    description = tmpl.format(name=name, phone=phone)
    amount = render_amount(*CATEGORY_META["Переводы"]["amount_range"], rng)
    return description, amount, "", rng.choice(["debit", "credit"]), name, False


def _sample_income(rng):
    tmpl = rng.choice(INCOME_TEMPLATES)
    company = rng.choice(COMPANY_NAMES)
    name = random_person_name(rng)
    all_train_merchants = [m for cat in MERCHANTS.values() for m in cat["train"]]
    merchant = rng.choice(all_train_merchants)
    description = tmpl.format(company=company, name=name, merchant=merchant)

    # Служебное поле _merchant должно отражать РЕАЛЬНО подставленную в текст
    # сущность, иначе разбор ошибок на шаге 6 будет вестись по неверным данным.
    if "{merchant}" in tmpl:
        debug_merchant = merchant
    elif "{company}" in tmpl:
        debug_merchant = company
    elif "{name}" in tmpl:
        debug_merchant = name
    else:
        debug_merchant = None

    amount = render_amount(*CATEGORY_META["Прочие доходы"]["amount_range"], rng)

    # Доход по определению не может быть направлением "debit" — это была
    # семантическая ошибка данных, влияющая на реалистичность выписки.
    return description, amount, "", "credit", debug_merchant, False


def _sample_ambiguous(rng):
    tmpl = rng.choice(AMBIGUOUS_TEMPLATES)
    num = rng.randint(1000, 999999)
    description = tmpl.format(num=num)
    amount = render_amount(*CATEGORY_META["Прочее"]["amount_range"], rng)
    return description, amount, "", "debit", None, False


def sample_row(category, rng, row_id, noise_rate=0.0,
                merchant_mode="train", force_ip=False):
    if category == "Переводы":
        description, amount, mcc, direction, merchant, is_holdout = _sample_transfer(rng)
    elif category == "Прочие доходы":
        description, amount, mcc, direction, merchant, is_holdout = _sample_income(rng)
    elif category == "Прочее":
        description, amount, mcc, direction, merchant, is_holdout = _sample_ambiguous(rng)
    else:
        description, amount, mcc, direction, merchant, is_holdout = _sample_retail_or_service(
            category, rng, noise_rate, merchant_mode, force_ip
        )
    return {
        "transaction_id": row_id,
        "date": _random_date(rng).isoformat(),
        "description": description,
        "amount": round(amount, 2),
        "direction": direction,
        "mcc": mcc,
        "category": category,
        "_merchant": merchant,       # служебное поле, для анализа ошибок
        "_is_holdout": is_holdout,   # служебное поле, для анализа ошибок
    }


def _weighted_categories(n, rng):
    cats, weights = list(CATEGORY_WEIGHTS.keys()), list(CATEGORY_WEIGHTS.values())
    return rng.choices(cats, weights=weights, k=n)


def build_dataset(n_rows, seed, merchant_mode="train", noise_rate=0.0):
    rng = random.Random(seed)
    cats = _weighted_categories(n_rows, rng)
    rows = [sample_row(c, rng, i, noise_rate=noise_rate, merchant_mode=merchant_mode)
            for i, c in enumerate(cats)]
    return pd.DataFrame(rows)


def build_ml_train(n_rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Обучающая выборка для ML-фоллбэка. Только train-мерчанты, умеренный
    шум — модель должна научиться обобщать на опечатки/сокращения, но не
    видит holdout-бренды из golden-сета."""
    return build_dataset(n_rows, seed, merchant_mode="train", noise_rate=0.25)


def build_sample_statement(n_rows: int = 50, seed: int = 7) -> pd.DataFrame:
    """Небольшой безлейбловый CSV — то, что реально загрузит пользователь
    в API/CLI. Категория здесь — то, что должен вывести сервис, а не вход."""
    df = build_dataset(n_rows, seed, merchant_mode="train", noise_rate=0.15)
    return df.drop(columns=["category", "_merchant", "_is_holdout"])


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетической выписки")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--ml-train-rows", type=int, default=2500)
    parser.add_argument("--sample-rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ml_train = build_ml_train(args.ml_train_rows, args.seed)
    ml_train.to_csv(out / "ml_train.csv", index=False)
    print(f"ml_train.csv: {len(ml_train)} строк -> {out/'ml_train.csv'}")

    sample = build_sample_statement(args.sample_rows, args.seed + 1)
    sample.to_csv(out / "sample_statement.csv", index=False)
    print(f"sample_statement.csv: {len(sample)} строк -> {out/'sample_statement.csv'}")


if __name__ == "__main__":
    main()