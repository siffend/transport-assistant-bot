"""Telegram-бот первой линии поддержки пассажиров.

Тонкий клиент Telegram Bot API на стандартной библиотеке: ни python-telegram-bot,
ни requests не требуются. Поддерживаются оба режима доставки обновлений:

  long polling — для разработки и демонстрации:      python run_bot.py
  webhook      — для продакшена (Flask-приложение):  см. assistant/web.py

Логика ответа целиком в assistant/pipeline.py — транспорт легко заменить.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .config import settings
from .pipeline import Assistant, Reply

log = logging.getLogger("tg")

MAX_TEXT = 4000  # запас к лимиту Telegram в 4096 символов


class TelegramAPI:
    def __init__(self, token: str, api_root: Optional[str] = None):
        if not token:
            raise ValueError("Не задан TELEGRAM_TOKEN")
        root = (api_root or settings.telegram_api).rstrip("/")
        self.base = f"{root}/bot{token}"

    def _call(self, method: str, payload: dict, timeout: float = 65.0) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/{method}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            log.error("Telegram HTTP %s: %s", exc.code, body)
            return {"ok": False, "error": body}

    # --- методы API --------------------------------------------------------
    def get_updates(self, offset: Optional[int], timeout: int = 30) -> List[dict]:
        payload = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        data = self._call("getUpdates", payload, timeout=timeout + 20)
        return data.get("result", []) if data.get("ok") else []

    def send_message(self, chat_id: int, text: str, buttons: Optional[List[dict]] = None) -> dict:
        payload = {
            "chat_id": chat_id,
            "text": text[:MAX_TEXT],
            "disable_web_page_preview": True,
        }
        if buttons:
            payload["reply_markup"] = {"inline_keyboard": buttons}
        return self._call("sendMessage", payload, timeout=20)

    def answer_callback(self, callback_id: str) -> dict:
        return self._call("answerCallbackQuery", {"callback_query_id": callback_id}, timeout=10)

    def set_webhook(self, url: str, secret: str = "") -> dict:
        payload = {"url": url, "allowed_updates": ["message", "callback_query"]}
        if secret:
            payload["secret_token"] = secret
        return self._call("setWebhook", payload, timeout=20)

    def delete_webhook(self) -> dict:
        return self._call("deleteWebhook", {"drop_pending_updates": False}, timeout=20)


def build_keyboard(reply: Reply) -> Optional[List[List[dict]]]:
    """Кнопки уточнений и эскалации в формате inline_keyboard."""
    if not reply.buttons:
        return None
    rows = []
    for btn in reply.buttons[:6]:
        label = btn.label if len(btn.label) <= 58 else btn.label[:55] + "…"
        rows.append([{"text": label, "callback_data": btn.payload[:64]}])
    return rows


class BotRuntime:
    """Состояние диалогов + маршрутизация обновлений в ядро ассистента."""

    def __init__(self, assistant: Optional[Assistant] = None, api: Optional[TelegramAPI] = None):
        self.assistant = assistant or Assistant()
        self.api = api or TelegramAPI(settings.telegram_token)
        self.sessions: Dict[int, dict] = {}

    def state_for(self, chat_id: int) -> dict:
        return self.sessions.setdefault(chat_id, {})

    # --- обработка одного обновления ---------------------------------------
    def handle_update(self, update: dict) -> None:
        if "callback_query" in update:
            self._handle_callback(update["callback_query"])
        elif "message" in update:
            self._handle_message(update["message"])

    def _handle_message(self, message: dict) -> None:
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text") or message.get("caption")
        if chat_id is None:
            return
        if not text:
            self.api.send_message(
                chat_id,
                "Пока понимаю только текстовые сообщения. Опишите вопрос словами.",
            )
            return
        reply = self.assistant.ask(text, self.state_for(chat_id))
        self._send(chat_id, reply)
        log.info("chat=%s kind=%s intent=%s conf=%.2f doc=%s",
                 chat_id, reply.kind, reply.intent, reply.confidence, reply.doc_id)

    def _handle_callback(self, callback: dict) -> None:
        chat_id = callback.get("message", {}).get("chat", {}).get("id")
        payload = callback.get("data", "")
        self.api.answer_callback(callback.get("id", ""))
        if chat_id is None:
            return
        if payload == "operator":
            reply = self.assistant.operator_reply()
        else:
            reply = self.assistant.ask(payload, self.state_for(chat_id))
        self._send(chat_id, reply)

    def _send(self, chat_id: int, reply: Reply) -> None:
        self.api.send_message(chat_id, reply.full_text, build_keyboard(reply))

    # --- long polling -------------------------------------------------------
    def run_polling(self, poll_timeout: int = 30) -> None:
        log.info("Бот запущен в режиме long polling")
        self.api.delete_webhook()
        offset: Optional[int] = None
        while True:
            try:
                updates = self.api.get_updates(offset, timeout=poll_timeout)
            except Exception as exc:  # сеть моргнула — продолжаем
                log.warning("Ошибка получения обновлений: %s", exc)
                time.sleep(3)
                continue
            for upd in updates:
                offset = upd["update_id"] + 1
                try:
                    self.handle_update(upd)
                except Exception:
                    log.exception("Ошибка обработки обновления %s", upd.get("update_id"))
