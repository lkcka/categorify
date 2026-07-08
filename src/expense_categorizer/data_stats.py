"""Быстрая проверка сгенерированного CSV: распределение категорий, holdout-доля."""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    print(f"Всего строк: {len(df)}")
    if "category" in df.columns:
        print("\nРаспределение категорий:")
        print(df["category"].value_counts())
    if "_is_holdout" in df.columns:
        print(f"\nСтрок с holdout-мерчантом (никогда не виденным): {int(df['_is_holdout'].sum())}")
    if "direction" in df.columns:
        print("\nНаправление:")
        print(df["direction"].value_counts())
    print("\nПримеры строк:")
    print(df.sample(min(8, len(df)), random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()