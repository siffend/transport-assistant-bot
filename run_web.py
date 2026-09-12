#!/usr/bin/env python3
"""Запуск демо-чата в браузере и вебхука Telegram.

    python run_web.py            # http://127.0.0.1:8000
    PORT=9000 python run_web.py
"""

from __future__ import annotations

import logging
import os

from run_bot import load_dotenv


def chat() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    from assistant import config

    config.settings = config.Settings()
    from assistant.web import create_app

    app = create_app()
    port = int(os.getenv("PORT", "8000"))
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=False)


if __name__ == "__main__":
    chat()
