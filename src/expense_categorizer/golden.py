"""
Golden (held-out) набор для оценки качества.

Состав (доли от n_rows):
  35% известный мерчант, чистое описание   -> проверка правил
  25% известный мерчант, шумное описание   -> проверка устойчивости к опечаткам
  15% НИКОГДА не виденный мерчант (holdout)-> честный тест генерализации
  10% ИП/самозанятые (часть с подсказкой, часть без)
  10% переводы и доходы                    -> отдельная лексика, должно быть легко
   5% явная неоднозначность (ground truth = "Прочее")

Правило разметки неоднозначных случаев зафиксировано явно (не задним
числом): если в описании нет НИ ОДНОГО сигнала о категории — правильный
ответ "Прочее". Согласовано на этапе проектирования (см. диалог/README).
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from expense_categorizer.categories import CATEGORIES
from expense_categorizer.catalog import CATEGORY_WEIGHTS, IP_HINTS
from expense_categorizer.generator import sample_row, _sample_ambiguous, _random_date

REGULAR_CATEGORIES = [c for c in CATEGORIES
                       if c not in ("Переводы", "Прочие доходы", "Прочее")]
IP_CAPABLE_CATEGORIES = list(IP_HINTS.keys())


def _pick_regular_category(rng):
    weights = [CATEGORY_WEIGHTS[c] for c in REGULAR_CATEGORIES]
    return rng.choices(REGULAR_CATEGORIES, weights=weights, k=1)[0]


def _ambiguous_ip_row(rng, rid):
    from expense_categorizer.textgen import random_person_name
    name = random_person_name(rng)
    description = f"ИП {name}"
    amount = round(rng.uniform(300, 8000), 2)
    return {
        "transaction_id": rid, "date": _random_date(rng).isoformat(),
        "description": description, "amount": amount, "direction": "debit",
        "mcc": "", "category": "Прочее", "_merchant": name, "_is_holdout": False,
    }


def build_golden(n_rows: int = 450, seed: int = 123) -> pd.DataFrame:
    rng = random.Random(seed)
    composition = {
        "known_clean": 0.35, "known_noisy": 0.25, "holdout_unseen": 0.15,
        "ip": 0.10, "transfer_income": 0.10, "ambiguous": 0.05,
    }
    counts = {k: round(v * n_rows) for k, v in composition.items()}
    counts["known_clean"] += n_rows - sum(counts.values())  # компенсация округления

    rows, rid = [], 0

    for _ in range(counts["known_clean"]):
        c = _pick_regular_category(rng)
        rows.append(sample_row(c, rng, rid, noise_rate=0.0, merchant_mode="train")); rid += 1

    for _ in range(counts["known_noisy"]):
        c = _pick_regular_category(rng)
        rows.append(sample_row(c, rng, rid, noise_rate=0.7, merchant_mode="train")); rid += 1

    for _ in range(counts["holdout_unseen"]):
        c = _pick_regular_category(rng)
        rows.append(sample_row(c, rng, rid, noise_rate=0.3, merchant_mode="holdout")); rid += 1

    for _ in range(counts["ip"]):
        c = rng.choice(IP_CAPABLE_CATEGORIES)
        if rng.random() < 0.4:
            rows.append(_ambiguous_ip_row(rng, rid))       # без подсказки -> "Прочее"
        else:
            rows.append(sample_row(c, rng, rid, noise_rate=0.1,
                                    merchant_mode="train", force_ip=True))
        rid += 1

    for _ in range(counts["transfer_income"]):
        c = rng.choice(["Переводы", "Прочие доходы"])
        rows.append(sample_row(c, rng, rid)); rid += 1

    for _ in range(counts["ambiguous"]):
        description, amount, mcc, direction, merchant, _ = _sample_ambiguous(rng)
        rows.append({
            "transaction_id": rid, "date": _random_date(rng).isoformat(),
            "description": description, "amount": round(amount, 2), "direction": direction,
            "mcc": mcc, "category": "Прочее", "_merchant": merchant, "_is_holdout": False,
        })
        rid += 1

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Генерация golden-набора для оценки")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--rows", type=int, default=450)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_golden(args.rows, args.seed)
    df.to_csv(out / "golden.csv", index=False)
    print(f"golden.csv: {len(df)} строк -> {out/'golden.csv'}")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()