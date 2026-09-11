"""Генерация базовой части базы знаний из исходного Excel.

Читает data/source/khakaton.xlsx (21 тестовый вопрос) и сохраняет
data/kb/base_from_xlsx.json — документы дословно, без редактуры.
Редакторские правки (разбиение длинных ответов на атомарные документы,
перефразировки, слоты уточнений) лежат отдельно в data/kb/curated.json
и ссылаются на базовые документы полем "replaces".

Запуск:  python tools/build_kb.py
"""

from __future__ import annotations

import json
import os
import re
import sys

try:
    import openpyxl
except ImportError:  # pragma: no cover
    sys.exit("Требуется openpyxl:  pip install openpyxl")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "source", "khakaton.xlsx")
DST = os.path.join(ROOT, "data", "kb", "base_from_xlsx.json")

# Соответствие «номер строки Excel → идентификатор и интент документа».
ROW_MAP = {
    1: ("q01.benefit.categories", "benefit_categories"),
    2: ("q02.benefit.cost", "benefit_cost"),
    3: ("q03.routes.scheme", "routes_scheme"),
    4: ("q04.fare.cost", "fare_cost"),
    5: ("q05.apps.tracking", "apps_tracking"),
    6: ("q06.stops.new", "stops_new"),
    7: ("q07.payment.methods", "payment_methods"),
    8: ("q08.complaint.schedule", "complaint_schedule"),
    9: ("q09.benefit.suburban", "benefit_suburban"),
    10: ("q10.benefit.wearables", "benefit_wearables"),
    11: ("q11.payment.notification", "payment_notification"),
    12: ("q12.payment.double_charge", "payment_double_charge"),
    13: ("q13.routes.planning", "routes_planning"),
    14: ("q14.benefit.nonresident", "benefit_nonresident"),
    15: ("q15.benefit.where", "benefit_where"),
    16: ("q16.benefit.transfer", "benefit_transfer"),
    17: ("q17.routes.changes", "routes_changes"),
    18: ("q18.complaint.condition", "complaint_condition"),
    19: ("q19.validator.green", "validator_green"),
    20: ("q20.benefit.lost_card", "benefit_lost_card"),
    21: ("q21.benefit.wearables_charge", "benefit_wearables_charge"),
}


def clean(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_urls(value: str):
    if not value:
        return []
    parts = re.split(r"[\s,;]+", value)
    return [p.strip() for p in parts if p.strip().startswith("http")]


def main() -> None:
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb.worksheets[0]

    docs = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        try:
            num = int(str(row[0]).strip())
        except ValueError:
            continue
        doc_id, intent = ROW_MAP.get(num, (f"q{num:02d}.misc", "other"))
        docs.append(
            {
                "id": doc_id,
                "intent": intent,
                "source_row": num,
                "title": clean(row[1]),
                "question": clean(row[1]),
                "paraphrases": [],
                "answer": clean(row[2]),
                "npa": clean(row[3]),
                "urls": split_urls(clean(row[4])),
                "origin": "khakaton.xlsx",
            }
        )

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as fh:
        json.dump({"documents": docs}, fh, ensure_ascii=False, indent=2)
    print(f"Сохранено документов: {len(docs)} → {DST}")


if __name__ == "__main__":
    main()
