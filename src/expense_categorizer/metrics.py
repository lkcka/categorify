"""Расчёт метрик качества категоризации по golden-набору."""
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from expense_categorizer.categories import CATEGORIES


@dataclass
class EvalReport:
    accuracy: float
    n_total: int
    n_correct: int
    classification_report_text: str
    confusion_df: pd.DataFrame
    source_breakdown: pd.DataFrame
    errors_df: pd.DataFrame


def evaluate(golden_df: pd.DataFrame, predictions: pd.DataFrame) -> EvalReport:
    """
    golden_df: колонка 'category' — истина.
    predictions: колонки 'pred_category', 'confidence', 'source',
                 индекс должен совпадать по позиции с golden_df.
    """
    merged = golden_df.reset_index(drop=True).join(predictions.reset_index(drop=True))
    merged["correct"] = merged["category"] == merged["pred_category"]

    n_total = len(merged)
    n_correct = int(merged["correct"].sum())
    accuracy = n_correct / n_total if n_total else 0.0

    report_text = classification_report(
        merged["category"], merged["pred_category"],
        labels=CATEGORIES, zero_division=0,
    )

    cm = confusion_matrix(merged["category"], merged["pred_category"], labels=CATEGORIES)
    confusion_df = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)

    source_breakdown = (
        merged.groupby("source")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("n", ascending=False)
    )

    cols = [c for c in ["transaction_id", "description", "mcc", "category",
                         "pred_category", "confidence", "source"] if c in merged.columns]
    errors_df = merged.loc[~merged["correct"], cols]

    return EvalReport(accuracy, n_total, n_correct, report_text, confusion_df, source_breakdown, errors_df)