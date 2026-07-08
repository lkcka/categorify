"""
Слой 1 пайплайна: детерминированные правила.

Дизайн-принцип: правила на 100% предсказуемы и объяснимы, поэтому здесь
осознанно НЕТ нечёткого/fuzzy-сопоставления. Опечатки и неизвестные
написания — зона ответственности ML-слоя (шаг 5): char n-gram TF-IDF
устойчив к ним "из коробки", а вот объяснить, почему правило сработало
"на расстоянии Левенштейна 2", — сложно и хрупко. Разделение
ответственности между слоями держит каждый слой простым.

Порядок проверки (важен, конфликты разрешаются порядком):
 1. IP-подсказки (явные фразы услуг: "мастер маникюра" и т.п.)
 2. Явно неопределимые по построению паттерны (маркетплейс без товара,
    голый "ИП Фамилия И.О." без описания услуги, обезличенное списание)
    -> "Прочее". Это осознанное бизнес-решение, а не отказ распознать.
 3. Ключевые слова доходов (кэшбэк, зарплата, возврат, начисление %...)
    — проверяются РАНЬШЕ словаря мерчантов, иначе "Возврат покупки Ашан"
    попадёт в "Продукты" по имени мерчанта внутри строки дохода.
 4. Ключевые слова переводов (перевод, sbp, п2п...)
 5. Точное совпадение с известным мерчантом из справочника (train-часть)
 6. Совпадение по MCC-коду (менее специфично -> ниже словаря по приоритету)
 7. Нет совпадения -> None (передаётся дальше по пайплайну)
"""
import re
from dataclasses import dataclass

from expense_categorizer.catalog import MERCHANTS, CATEGORY_META, IP_HINTS
from expense_categorizer.normalize import normalize, merchant_variants, contains_phrase

INCOME_KEYWORDS = [
    "ЗАРАБОТНОЙ ПЛАТЫ", "ЗАРПЛАТ", "КЭШБЭК", "CASHBACK", "ПРЕМИИ", "ПРЕМИЯ",
    "НАЧИСЛЕНИЕ ПРОЦЕНТОВ", "ВОЗВРАТ", "ПОПОЛНЕНИЕ СЧЕТА ОТ",
]
TRANSFER_KEYWORDS = ["ПЕРЕВОД", "PEREVOD", "СБП", "SBP", "П2П"]
AMBIGUOUS_KEYWORDS = [
    "OZON", "WILDBERRIES", "AVITO", "МАРКЕТПЛЕЙС", "MARKETPLACE",
    "СПИСАНИЕ ПО ОПЕРАЦИИ", "ПРОЧИЕ УСЛУГИ", "YANDEX MARKET",
]


@dataclass(frozen=True)
class RuleResult:
    category: str
    confidence: float
    source: str


def _build_merchant_patterns() -> list[tuple[re.Pattern, str]]:
    """Компилируем regex один раз при инициализации (не на каждой строке).
    Строится ТОЛЬКО из train-мерчантов каталога — holdout-бренды сюда
    никогда не попадают, иначе метрика на golden-сете была бы фиктивной."""
    patterns = []
    for category, buckets in MERCHANTS.items():
        for name in buckets["train"]:
            for variant in merchant_variants(name):
                pattern = re.compile(r"(?<!\S)" + re.escape(variant) + r"(?!\S)")
                patterns.append((pattern, category))
    # длинные (более специфичные) варианты матчим первыми
    patterns.sort(key=lambda p: -len(p[0].pattern))
    return patterns


def _build_mcc_index() -> dict[str, str]:
    index = {}
    for category, meta in CATEGORY_META.items():
        for mcc in meta["mcc"]:
            index[str(mcc)] = category
    return index


def _build_ip_hint_index() -> list[tuple[str, str]]:
    pairs = [(normalize(phrase), category)
             for category, phrases in IP_HINTS.items() for phrase in phrases]
    return sorted(pairs, key=lambda p: -len(p[0]))


class RuleEngine:
    def __init__(self):
        self._merchant_patterns = _build_merchant_patterns()
        self._mcc_index = _build_mcc_index()
        self._ip_hints = _build_ip_hint_index()

    def classify(self, description: str, mcc=None) -> RuleResult | None:
        norm = normalize(description)
        if not norm:
            return None

        for phrase, category in self._ip_hints:
            if contains_phrase(norm, phrase):
                return RuleResult(category, 0.97, "rule_ip_hint")

        if any(contains_phrase(norm, kw) for kw in INCOME_KEYWORDS):
            return RuleResult("Прочие доходы", 0.9, "rule_income_kw")

        if any(contains_phrase(norm, kw) for kw in TRANSFER_KEYWORDS):
            return RuleResult("Переводы", 0.9, "rule_transfer_kw")

        for pattern, category in self._merchant_patterns:
            if pattern.search(norm):
                return RuleResult(category, 1.0, "rule_merchant")


        mcc_str = self.normalize_mcc(mcc)
        if mcc_str and mcc_str in self._mcc_index:
            # MCC — структурные данные от платёжной сети, надёжнее текстовой
            # эвристики "явная неопределённость" ниже. У по-настоящему
            # неоднозначных строк (ambiguous-шаблоны генератора) MCC всегда
            # пуст по построению (см. catalog.py CATEGORY_META["Прочее"]),
            # поэтому поднятие этой проверки не может случайно "разрешить"
            # то, что объективно неразрешимо.
            return RuleResult(self._mcc_index[mcc_str], 0.8, "rule_mcc")

        # Явно неопределимые паттерны — последний рубеж перед ML/fallback.
        if self._looks_like_bare_ip(norm) or any(
            contains_phrase(norm, kw) or kw in norm for kw in AMBIGUOUS_KEYWORDS
        ):
            return RuleResult("Прочее", 0.9, "rule_explicit_unknown")

        return None

    @staticmethod
    def _looks_like_bare_ip(norm: str) -> bool:
        """'Голый' ИП без описания услуги: 'ИП', фамилия (буквенный токен),
        и ноль-два однобуквенных инициала — и НИЧЕГО больше. Если после
        инициалов идёт что-то ещё (описание услуги), это уже не голый
        случай — но такие строки перехватываются выше, в rule_ip_hint,
        если фраза-подсказка распознана.
        Пример 'ИП ПЕТРОВ И А' (после normalize точки в инициалах стали
        пробелами -> 4 токена) -> True.
        """
        tokens = norm.split()
        if not tokens or tokens[0] != "ИП":
            return False
        tokens = tokens[1:]
        if not tokens:
            return True  # просто "ИП" без имени вообще
        surname, initials = tokens[0], tokens[1:]
        if not surname.isalpha():
            return False
        return all(len(t) == 1 for t in initials)

    @staticmethod
    def normalize_mcc(mcc) -> str | None:
        if mcc is None:
            return None
        try:
            if isinstance(mcc, float) and mcc != mcc:  # NaN
                return None
            return str(int(float(mcc)))
        except (ValueError, TypeError):
            return None