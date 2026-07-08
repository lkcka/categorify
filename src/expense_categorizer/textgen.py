"""
Рендеринг "сырых" строк описания транзакции + инъекция шума.

Почему POS-транзакции транслитерируются в латиницу и капсуятся: платёжные
сети (Visa/Mastercard) передают merchant descriptor латиницей — это не
"шум ради шума", а имитация реального формата, с которым столкнётся
пайплайн на настоящих выписках.
"""
import random

_TRANSLIT_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def transliterate(text: str) -> str:
    return "".join(_TRANSLIT_MAP.get(ch, ch) for ch in text.lower()).upper()


RETAIL_TEMPLATES = [
    "{name}",
    "OPLATA {name}",
    "POKUPKA {name} {city}",
    "{name} {city} RUS",
    "POS {name}",
    "{name}*{num}",
    "TERMINAL {num} {name} {city}",
    "OPLATA TOVAROV {name}",
]

SERVICE_TEMPLATES = [
    "Оплата услуг {name}",
    "{name} абонентская плата",
    "Списание ЖКУ {name}",
    "Оплата {name} л/с {num}",
    "{name} счет №{num}",
]

_FIRST_NAMES = ["Иван", "Петр", "Сергей", "Андрей", "Мария", "Елена", "Ольга",
                "Наталья", "Дмитрий", "Алексей", "Анна", "Татьяна", "Виктор", "Юрий"]
_LAST_NAMES = ["Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Попов",
               "Соколов", "Морозов", "Волков", "Новиков", "Федоров", "Егоров"]


def random_person_name(rng: random.Random) -> str:
    """Формат 'Фамилия И.И.' — типичное представление в описаниях ИП/переводов.
    Женское окончание добавляется эвристически (упрощение, не лингвистически
    строго, но достаточно для синтетики)."""
    last = rng.choice(_LAST_NAMES)
    if rng.random() < 0.4:
        last += "а"
    first_initial = rng.choice(_FIRST_NAMES)[0]
    patronymic_initial = rng.choice(_FIRST_NAMES)[0]
    return f"{last} {first_initial}.{patronymic_initial}."


def render_retail_description(name: str, city: str, rng: random.Random) -> str:
    tmpl = rng.choice(RETAIL_TEMPLATES)
    return tmpl.format(name=transliterate(name), city=transliterate(city),
                        num=rng.randint(1000, 999999))


def render_service_description(name: str, rng: random.Random) -> str:
    tmpl = rng.choice(SERVICE_TEMPLATES)
    return tmpl.format(name=name, num=rng.randint(1000, 999999))


def render_amount(lo: float, hi: float, rng: random.Random) -> float:
    """Суммы 'как настоящие' цены (округлые окончания), а не чистый uniform."""
    raw = rng.uniform(lo, hi)
    base = int(raw // 100) * 100
    ending = rng.choice([0, 90, 99, 50, 49, 9])
    value = max(lo, min(base + ending, hi))
    return round(float(value), 2)


_ABBREV = {"ОПЛАТА": "ОПЛ", "ПОКУПКА": "ПОК", "ПЕРЕВОД": "ПРВД",
           "УСЛУГ": "УСЛ", "МАГАЗИН": "МАГ"}


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[: i + 1] + word[i] + word[i + 1:]


def maybe_add_noise(text: str, rng: random.Random, noise_rate: float) -> str:
    """С вероятностью noise_rate применяет ОДНУ операцию шума (сокращение
    известного слова либо однобуквенную опечатку). Шум умеренный намеренно:
    реальные выписки бывают неряшливыми, но редко нечитаемыми."""
    if rng.random() >= noise_rate:
        return text
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    upper_word = words[idx].upper()
    if upper_word in _ABBREV and rng.random() < 0.5:
        words[idx] = _ABBREV[upper_word]
    else:
        words[idx] = _typo(words[idx], rng)
    return " ".join(words)