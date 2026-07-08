"""
Единая нормализация текста описаний транзакций.

Критично: и правила (точное сопоставление словаря), и ML-слой (шаг 5)
должны "видеть" текст одинаково нормализованным — иначе несоответствие
токенизации между слоями будет давать необъяснимые расхождения.
"""
import re

from expense_categorizer.textgen import transliterate

_APOSTROPHES_RE = re.compile(r"['’ʼ]")
_NON_ALNUM_RE = re.compile(r"[^0-9A-ZА-ЯЁ\s]")
_DIGIT_RUN_RE = re.compile(r"\b\d+\b")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Верхний регистр, апострофы удаляются (не заменяются пробелом,
    т.к. 'О'Кей' должно остаться одним словом 'ОКЕЙ'), прочая пунктуация
    заменяется пробелом, голые числа (номера терминалов/счетов — шум для
    сопоставления) убираются, пробелы схлопываются."""
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = _APOSTROPHES_RE.sub("", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _DIGIT_RUN_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def merchant_variants(name: str) -> list[str]:
    """Из канонического имени мерчанта (кириллица, как в каталоге) строит
    варианты для сопоставления: нормализованная кириллица + транслит,
    т.к. в реальном merchant descriptor бренд почти всегда латиницей."""
    variants = {normalize(name), normalize(transliterate(name))}
    return [v for v in variants if v]


def contains_phrase(norm_text: str, norm_phrase: str) -> bool:
    """Проверка вхождения фразы как отдельного 'токена/подстроки по
    границам пробелов', а не произвольной подстроки — иначе короткие
    названия (МТС, ОК) будут ложно матчиться внутри других слов."""
    if not norm_phrase:
        return False
    pattern = r"(?<!\S)" + re.escape(norm_phrase) + r"(?!\S)"
    return re.search(pattern, norm_text) is not None