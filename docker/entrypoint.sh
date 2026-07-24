#!/bin/sh
set -eu

cmd="${1:-evaluate}"
shift || true

case "$cmd" in
  evaluate|eval)
    exec python -m expense_categorizer.evaluate "$@"
    ;;
  categorize|cat)
    exec python -m expense_categorizer.categorize "$@"
    ;;
  train)
    exec python -m expense_categorizer.train_ml "$@"
    ;;
  generate)
    exec python -m expense_categorizer.generator "$@"
    ;;
  golden)
    exec python -m expense_categorizer.golden "$@"
    ;;
  stats)
    exec python -m expense_categorizer.data_stats "$@"
    ;;
  debug-llm)
    exec python scripts/debug_llm.py "$@"
    ;;
  shell)
    exec /bin/sh "$@"
    ;;
  *)
    exec python -m "expense_categorizer.${cmd}" "$@"
    ;;
esac
