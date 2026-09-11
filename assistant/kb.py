"""Загрузка базы знаний.

База состоит из двух слоёв:
  data/kb/base_from_xlsx.json — автогенерация из исходного Excel (дословно);
  data/kb/curated.json        — редакторский слой: атомарные документы,
                                перефразировки, слоты уточнений.

Документ curated-слоя может перекрывать базовые документы полем "replaces",
чтобы в поисковом индексе не было дублей одного и того же факта.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

KB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kb"
)


@dataclass
class Document:
    id: str
    intent: str
    title: str
    question: str
    answer: str
    npa: str = ""
    urls: List[str] = field(default_factory=list)
    paraphrases: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    grounding: str = "official"  # npa | official | none
    slot_value: Optional[str] = None
    source_row: Optional[int] = None
    origin: str = "curated"

    @property
    def search_text(self) -> str:
        """Текст, по которому строится индекс: вопрос важнее тела ответа."""
        parts = [self.question, self.title]
        parts.extend(self.paraphrases)
        parts.extend(self.keywords)
        parts.append(self.answer)
        return "\n".join(p for p in parts if p)

    @property
    def has_source(self) -> bool:
        return bool(self.urls) or bool(self.npa)


def _to_doc(raw: dict, origin: str) -> Document:
    return Document(
        id=raw["id"],
        intent=raw.get("intent", "other"),
        title=raw.get("title") or raw.get("question", ""),
        question=raw.get("question", ""),
        answer=raw.get("answer", ""),
        npa=raw.get("npa", ""),
        urls=list(raw.get("urls", [])),
        paraphrases=list(raw.get("paraphrases", [])),
        keywords=list(raw.get("keywords", [])),
        grounding=raw.get("grounding", "official"),
        slot_value=raw.get("slot_value"),
        source_row=raw.get("source_row"),
        origin=origin,
    )


def load_documents(kb_dir: str = KB_DIR) -> List[Document]:
    base_path = os.path.join(kb_dir, "base_from_xlsx.json")
    cur_path = os.path.join(kb_dir, "curated.json")

    base_raw: List[dict] = []
    if os.path.exists(base_path):
        with open(base_path, encoding="utf-8") as fh:
            base_raw = json.load(fh).get("documents", [])

    cur_raw: List[dict] = []
    if os.path.exists(cur_path):
        with open(cur_path, encoding="utf-8") as fh:
            cur_raw = json.load(fh).get("documents", [])

    replaced = set()
    for raw in cur_raw:
        replaced.update(raw.get("replaces", []))

    docs: List[Document] = []
    for raw in base_raw:
        if raw["id"] in replaced:
            continue
        docs.append(_to_doc(raw, origin="xlsx"))
    for raw in cur_raw:
        docs.append(_to_doc(raw, origin="curated"))

    ids = [d.id for d in docs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Дублирующиеся id документов: {sorted(duplicates)}")
    return docs


def load_domain_lexicon(kb_dir: str = KB_DIR) -> List[str]:
    """Термины, по которым вопрос опознаётся как относящийся к теме."""
    path = os.path.join(kb_dir, "domain_lexicon.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(json.load(fh).get("terms", []))


def load_clarify_rules(kb_dir: str = KB_DIR) -> Dict[str, dict]:
    """Возвращает {"variants": {...}, "slots": {...}} из clarify.json."""
    path = os.path.join(kb_dir, "clarify.json")
    if not os.path.exists(path):
        return {"variants": {}, "slots": {}}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        "variants": data.get("variants", {}),
        "slots": data.get("slots", {}),
    }
