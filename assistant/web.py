from __future__ import annotations

import logging
import os
from typing import Dict

from flask import Flask, jsonify, request, send_from_directory

from .config import settings
from .pipeline import Assistant, Reply

log = logging.getLogger("web")

WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"
)


def reply_to_dict(reply: Reply) -> dict:
    return {
        "kind": reply.kind,
        "text": reply.text,
        "full_text": reply.full_text,
        "intent": reply.intent,
        "confidence": round(reply.confidence, 3),
        "doc_id": reply.doc_id,
        "source": {
            "npa": reply.source.npa if reply.source else "",
            "urls": reply.source.urls if reply.source else [],
        },
        "buttons": [
            {"label": b.label, "payload": b.payload}
            for b in reply.buttons
        ],
        "debug": reply.debug,
    }


def create_app(assistant: Assistant | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)

    app.assistant = assistant or Assistant()
    app.sessions: Dict[str, Dict[str, object]] = {} # type: ignore

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "main.html")

    @app.get("/why-us")
    def why_us():
        return send_from_directory(WEB_DIR, "why_us.html")

    @app.get("/css/<path:filename>")
    def css(filename):
        return send_from_directory(
            os.path.join(WEB_DIR, "css"),
            filename
        )

    @app.get("/font/<path:filename>")
    def font(filename):
        return send_from_directory(
            os.path.join(WEB_DIR, "font"),
            filename
        )

    @app.get("/img/<path:filename>")
    def img(filename):
        return send_from_directory(
            os.path.join(WEB_DIR, "img"),
            filename
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "documents": len(app.assistant.documents),
                "llm_enabled": settings.llm_enabled,
                "thresholds": {
                    "answer": settings.threshold_answer,
                    "refuse": settings.threshold_refuse,
                },
            }
        )

    @app.post("/api/ask")
    def api_ask():
        data = request.get_json(silent=True) or {}

        text = (data.get("text") or "").strip()
        session_id = (
            data.get("session")
            or request.remote_addr
            or "default"
        )

        state = app.sessions.setdefault(session_id, {})

        if text == "operator":
            reply = app.assistant.operator_reply()
        else:
            reply = app.assistant.ask(text, state)

        return jsonify(reply_to_dict(reply))

    @app.post("/telegram/webhook")
    def telegram_webhook():
        if settings.webhook_secret:
            got = request.headers.get(
                "X-Telegram-Bot-Api-Secret-Token",
                ""
            )

            if got != settings.webhook_secret:
                return jsonify({"ok": False}), 403

        from .telegram_bot import BotRuntime

        runtime = getattr(app, "_tg_runtime", None)

        if runtime is None:
            runtime = BotRuntime(assistant=app.assistant)
            app._tg_runtime = runtime

        runtime.handle_update(
            request.get_json(silent=True) or {}
        )

        return jsonify({"ok": True})

    return app