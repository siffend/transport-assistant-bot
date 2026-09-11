"""Необязательный слой генерации ответа LLM поверх найденного контекста.

Работает с любым OpenAI-совместимым endpoint (GigaChat через прокси,
YandexGPT-совместимые шлюзы, vLLM, Ollama, OpenRouter): задаётся через
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.

Ключевая деталь — проверка заземления (grounding check): если в ответе
модели появилось число или сумма, которых нет в переданном контексте,
ответ отбрасывается и используется экстрактивный текст из базы знаний.
Так прототип не выдумывает тарифы и сроки, даже если модель «уверена».
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import List, Optional, Sequence, Tuple

from .config import settings

SYSTEM_PROMPT = """Ты — ассистент первой линии поддержки пассажиров общественного транспорта Тульской области.

Правила, нарушать которые нельзя:
1. Отвечай ТОЛЬКО на основании переданных фрагментов базы знаний.
2. Не добавляй числа, тарифы, сроки, телефоны и адреса, которых нет во фрагментах.
3. Если во фрагментах нет ответа — напиши ровно: НЕТ_ДАННЫХ
4. Отвечай по-русски, кратко (до 120 слов), деловым тоном, без приветствий.
5. Не придумывай ссылки и названия нормативных актов — их подставит система.
"""

_NUM_RE = re.compile(r"\d[\d\s.,\-]*")


def _numbers(text: str) -> List[str]:
    out = []
    for raw in _NUM_RE.findall(text or ""):
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 2:  # однозначные числа («1.», «2.») игнорируем
            out.append(digits)
    return out


def grounding_check(answer: str, context: str) -> Tuple[bool, List[str]]:
    """Проверяет, что все значимые числа ответа встречаются в контексте."""
    ctx_digits = set(_numbers(context))
    ctx_join = re.sub(r"\D", "", context)
    bad = []
    for num in _numbers(answer):
        if num in ctx_digits or num in ctx_join:
            continue
        bad.append(num)
    return (not bad), bad


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate(question: str, contexts: Sequence[str]) -> Optional[str]:
    """Возвращает сгенерированный ответ либо None (тогда работает fallback)."""
    if not settings.llm_enabled or not settings.llm_base_url:
        return None

    context_block = "\n\n".join(
        f"[Фрагмент {i + 1}]\n{c}" for i, c in enumerate(contexts)
    )
    payload = {
        "model": settings.llm_model,
        "temperature": 0.1,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Фрагменты базы знаний:\n{context_block}\n\nВопрос пассажира: {question}",
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    try:
        data = _post_json(url, payload, headers, settings.llm_timeout)
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:  # сеть, таймаут, формат — тихо уходим в fallback
        return None

    if not text or "НЕТ_ДАННЫХ" in text:
        return None

    ok, bad = grounding_check(text, context_block)
    if not ok:
        # модель привнесла числа, которых нет в источнике — не доверяем
        return None
    return text
