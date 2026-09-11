#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import sys


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'")
            )


def main() -> None:
    # СНАЧАЛА загружаем .env
    load_dotenv()

    # И ТОЛЬКО ПОТОМ импортируем бота
    from assistant import config

    config.settings = config.Settings()

    if not config.settings.telegram_token:
        sys.exit(
            "Не задан TELEGRAM_TOKEN."
        )

    from assistant.telegram_bot import BotRuntime

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    BotRuntime().run_polling()


if __name__ == "__main__":
    main()