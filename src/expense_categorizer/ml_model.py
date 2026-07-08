"""
Слой 2 пайплайна: ML-классификатор (TF-IDF + LogisticRegression) для
случаев, не покрытых детерминированными правилами (шаг 4), — опечатки в
известных мерчантах и мерчанты, которых нет в справочнике, но которые
лексически похожи на известные категории.

Почему LogisticRegression, а не LinearSVC/нейросеть:
 - predict_proba "из коробки" даёт уверенность, нужную для порога
   "ML не уверен -> Прочее" (LinearSVC потребовал бы доп. калибровку
   через CalibratedClassifierCV — лишняя сложность для такой задачи).
 - Обучается за секунды на CPU, никаких GPU-зависимостей.
 - Прозрачно объяснимо при разборе ошибок.

Почему TF-IDF word (1-2 grams) + char (3-5 grams) вместе, а не эмбеддинги:
 - word n-grams ловят лексические маркеры категории ("АПТЕКА", "ТАКСИ").
 - char n-grams устойчивы к опечаткам и транслитерационному шуму, а также
   обобщаются на неизвестные (holdout) бренды со схожей структурой имени.
 - Обе векторизации объединены в один sklearn Pipeline (FeatureUnion) —
   на инференсе не нужна ручная склейка фич.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from expense_categorizer.normalize import normalize

CANDIDATE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
CANDIDATE_C = [0.5, 1.0, 2.0, 5.0]
CANDIDATE_CHAR_NGRAMS = [(3, 5), (4, 6)]
CANDIDATE_MIN_DF = [1, 2]
N_CV_REPEATS = 5  # усреднение сглаживает шум одного случайного group-разбиения

def _build_sklearn_pipeline(C: float = 5.0, char_ngram: tuple[int, int] = (3, 5),
                             min_df: int = 1) -> Pipeline:
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, min_df=min_df)
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class MLBundle:
    pipeline: Pipeline
    threshold: float
    classes_: list[str]
    val_accuracy_at_threshold: float
    val_coverage_at_threshold: float
    C: float
    char_ngram: tuple[int, int]
    min_df: int
    val_accuracy_std: float

def _effective_accuracy(y_true, proba, classes, threshold, fallback_label):
    """Метрика для подбора порога: имитирует реальное поведение пайплайна
    (ML-слой отдаёт категорию только при достаточной уверенности, иначе
    -> честный fallback). Максимизация этой метрики — это максимизация
    итоговой accuracy пайплайна, а не абстрактная точность классификатора."""
    best_idx = proba.argmax(axis=1)
    best_conf = proba.max(axis=1)
    preds = np.where(best_conf >= threshold, classes[best_idx], fallback_label)
    accuracy = (preds == y_true).mean()
    coverage = (best_conf >= threshold).mean()
    return accuracy, coverage


def train_ml_bundle(df: pd.DataFrame, fallback_label: str = "Прочее",
                     val_size: float = 0.15, seed: int = 42) -> MLBundle:
    texts = df["description"].map(normalize)
    labels = df["category"].values

    if "_merchant" not in df.columns:
        raise ValueError("Колонка '_merchant' обязательна для честного grouped-сплита")
    row_ids = pd.Series("row_" + df.index.astype(str), index=df.index)
    groups = df["_merchant"].fillna(row_ids).values

    # Одно случайное group-разбиение — шумная оценка (пример: на маленьких
    # группах по 10 брендов/категория один "неудачный" hold-out бренд
    # драматически меняет метрику). Усредняем по N_CV_REPEATS разным
    # разбиениям, чтобы выбор гиперпараметров опирался на сигнал, а не шум.
    print(f"  Подбор гиперпараметров (C, char n-gram, min_df) и порога")
    print(f"  на {N_CV_REPEATS}x grouped-валидации (НЕ golden.csv):")

    splits = []
    for rep in range(N_CV_REPEATS):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + rep)
        train_idx, val_idx = next(splitter.split(texts, labels, groups))
        splits.append((train_idx, val_idx))

    best = None  # (C, char_ngram, min_df, threshold, mean_acc, std_acc, mean_cov)
    for C in CANDIDATE_C:
        for char_ngram in CANDIDATE_CHAR_NGRAMS:
            for min_df in CANDIDATE_MIN_DF:
                # накопим per-threshold метрики по всем повторам
                acc_by_threshold = {t: [] for t in CANDIDATE_THRESHOLDS}
                cov_by_threshold = {t: [] for t in CANDIDATE_THRESHOLDS}
                for train_idx, val_idx in splits:
                    x_train, x_val = texts.iloc[train_idx], texts.iloc[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]
                    pipeline_candidate = _build_sklearn_pipeline(C=C, char_ngram=char_ngram, min_df=min_df)
                    pipeline_candidate.fit(x_train, y_train)
                    classes = pipeline_candidate.named_steps["clf"].classes_
                    val_proba = pipeline_candidate.predict_proba(x_val)
                    for t in CANDIDATE_THRESHOLDS:
                        acc, cov = _effective_accuracy(y_val, val_proba, classes, t, fallback_label)
                        acc_by_threshold[t].append(acc)
                        cov_by_threshold[t].append(cov)
                for t in CANDIDATE_THRESHOLDS:
                    mean_acc = float(np.mean(acc_by_threshold[t]))
                    std_acc = float(np.std(acc_by_threshold[t]))
                    mean_cov = float(np.mean(cov_by_threshold[t]))
                    if best is None or mean_acc > best[4] or (mean_acc == best[4] and mean_cov > best[6]):
                        best = (C, char_ngram, min_df, t, mean_acc, std_acc, mean_cov)

    best_C, best_char_ngram, best_min_df, threshold, val_acc, val_acc_std, val_cov = best
    print(f"    Лучшая комбинация: C={best_C}, char_ngram={best_char_ngram}, min_df={best_min_df}, "
          f"threshold={threshold:.2f}")
    print(f"    effective_val_accuracy={val_acc:.4f} ± {val_acc_std:.4f} (по {N_CV_REPEATS} разбиениям), "
          f"coverage={val_cov:.2f}")

    # Дообучаем финальную модель на train+val: порог уже честно выбран на
    # отложенной части, golden.csv в этом процессе не участвовал вообще.
    pipeline_final = _build_sklearn_pipeline(C=best_C, char_ngram=best_char_ngram, min_df=best_min_df)
    pipeline_final.fit(texts, labels)


    return MLBundle(
        pipeline=pipeline_final,
        threshold=threshold,
        classes_=list(pipeline_final.named_steps["clf"].classes_),
        val_accuracy_at_threshold=val_acc,
        val_coverage_at_threshold=val_cov,
        C=best_C,
        char_ngram=best_char_ngram,
        min_df=best_min_df,
        val_accuracy_std=val_acc_std,
    )


def save_bundle(bundle: MLBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": bundle.pipeline,
        "threshold": bundle.threshold,
        "classes_": bundle.classes_,
        "C": bundle.C,
        "char_ngram": bundle.char_ngram,
        "min_df": bundle.min_df,
    }, path)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "threshold": bundle.threshold,
        "C": bundle.C,
        "char_ngram": list(bundle.char_ngram),
        "val_accuracy_at_threshold": bundle.val_accuracy_at_threshold,
        "val_coverage_at_threshold": bundle.val_coverage_at_threshold,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bundle(path: str | Path) -> MLBundle:
    data = joblib.load(path)
    return MLBundle(
        pipeline=data["pipeline"],
        threshold=data["threshold"],
        classes_=data["classes_"],
        val_accuracy_at_threshold=float("nan"),
        val_coverage_at_threshold=float("nan"),
        C=data.get("C", float("nan")),
        char_ngram=data.get("char_ngram", (None, None)),
        min_df=data.get("min_df", 1),
        val_accuracy_std=float("nan"),
    )


class MLCategorizer:
    """Обёртка над MLBundle для использования в пайплайне."""

    def __init__(self, bundle: MLBundle):
        self.bundle = bundle

    @classmethod
    def from_file(cls, path: str | Path) -> "MLCategorizer":
        return cls(load_bundle(path))

    def predict(self, description: str) -> tuple[str, float]:
        norm = normalize(description)
        proba = self.bundle.pipeline.predict_proba([norm])[0]
        idx = int(np.argmax(proba))
        classes = self.bundle.pipeline.named_steps["clf"].classes_
        return str(classes[idx]), float(proba[idx])