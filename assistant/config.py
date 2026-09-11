"""Конфигурация ассистента. Все параметры переопределяются переменными окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


@dataclass
class Settings:
    # --- пороги уверенности (шкала релевантности 0..1) ---------------------
    # выше ANSWER — отвечаем; между REFUSE и ANSWER — предлагаем варианты;
    # ниже REFUSE — честно сообщаем, что данных нет.
    threshold_answer: float = field(default_factory=lambda: _f("THRESHOLD_ANSWER", 0.24))
    threshold_refuse: float = field(default_factory=lambda: _f("THRESHOLD_REFUSE", 0.15))
    # минимальный отрыв топ-1 от топ-2 разных интентов, иначе — уточнение
    min_margin: float = field(default_factory=lambda: _f("MIN_MARGIN", 0.015))

    # --- детектор «вопрос не из этой предметной области» -------------------
    # Срабатывает, когда совпадение пришлось на тело ответа, а не на сам
    # вопрос документа, и при этом заметная часть лексики запроса базе
    # неизвестна: «как записаться к врачу через госуслуги» формально
    # пересекается с документом про оформление льготы, но ответом не является.
    ood_score_max: float = field(default_factory=lambda: _f("OOD_SCORE_MAX", 0.45))
    ood_domain_min: float = field(default_factory=lambda: _f("OOD_DOMAIN_MIN", 0.60))
    ontopic_score_max: float = field(default_factory=lambda: _f("ONTOPIC_SCORE_MAX", 0.60))
    clarify_enabled: bool = field(default_factory=lambda: _b("CLARIFY_ENABLED", True))

    # --- LLM (необязательно) ----------------------------------------------
    llm_enabled: bool = field(default_factory=lambda: _b("LLM_ENABLED", False))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout: float = field(default_factory=lambda: _f("LLM_TIMEOUT", 20.0))

    # --- Telegram ----------------------------------------------------------
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    telegram_api: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_API", "https://api.telegram.org")
    )
    webhook_secret: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", ""))

    # --- контакты оператора -------------------------------------------------
    operator_phone: str = "705-507"
    operator_name: str = "АО «ОЕИРЦ»"
    operator_site: str = "https://oeirc.ru"

    # --- зона ответственности ------------------------------------------------
    region: str = "Тульская область"
    other_regions: List[str] = field(
        default_factory=lambda: [
            "костром", "москв", "петербург", "питер", "калуг", "рязан",
            "орел", "орёл", "липецк", "воронеж", "твер", "ярославл",
            "владимир", "брянск", "курск", "смоленск", "нижн новгород",
            "нижний новгород", "казан", "екатеринбург", "самар", "сочи",
            "краснодар", "новосибирск", "перм", "уфа", "уфе", "челябинск",
            "ростов", "волгоград", "иванов", "белгород", "тамбов",
        ]
    )


settings = Settings()
