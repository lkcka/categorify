"""
CLI: python -m expense_categorizer.train_ml

Обучает ML-фоллбэк на data/ml_train.csv, честно подбирает порог
уверенности на ОТЛОЖЕННОЙ части train-выборки (НЕ на golden.csv — golden
используется только один раз, для итоговой независимой оценки, во
избежание переобучения конфигурации модели под тестовый набор).
"""
import argparse

import pandas as pd

from expense_categorizer.ml_model import train_ml_bundle, save_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/ml_train.csv")
    parser.add_argument("--out", default="models/ml_classifier.joblib")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    print(f"Обучающих строк: {len(df)}")
    bundle = train_ml_bundle(df, val_size=args.val_size, seed=args.seed)

    print(f"\nВыбран порог уверенности: {bundle.threshold:.2f}")
    print(f"Гиперпараметры: C={bundle.C}, char_ngram={bundle.char_ngram}, min_df={bundle.min_df}")
    print(f"Эффективная accuracy на валидации при этом пороге: {bundle.val_accuracy_at_threshold:.4f}" +
            f"± {bundle.val_accuracy_std:.4f}")
    print(f"Доля решений, принятых ML (не ушедших в fallback): {bundle.val_coverage_at_threshold:.2f}")
    
    save_bundle(bundle, args.out)
    print(f"\nМодель сохранена -> {args.out}")


if __name__ == "__main__":
    main()