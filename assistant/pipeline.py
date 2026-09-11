"""Ядро ассистента: интент → поиск → политика ответа → ответ с источником.

Политика ответа (по порядку проверок):
  1. команда/приветствие                       → карточка возможностей;
  2. ответ на ранее заданное уточнение         → ответ выбранным документом;
  3. явная просьба позвать оператора           → эскалация;
  4. вопрос про другой регион                  → отказ по зоне ответственности;
  5. релевантность ниже нижнего порога         → «данных нет» + оператор;
  6. не хватает существенного параметра        → уточняющий вопрос с кнопками;
  7. два разных интента идут вплотную          → уточнение «что именно имелось в виду»;
  8. релевантность между порогами              → предложение формулировок;
  9. иначе                                     → ответ + НПА + ссылка.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import llm
from .config import settings
from .kb import Document, load_clarify_rules, load_documents
from .retriever import Hit, HybridRetriever
from .text import contains_any, has_keyword

GREETING_RE = re.compile(
    r"^\s*(/start|/help|start|help|привет|здравствуй\w*|добрый\s+(день|вечер|утро)|"
    r"что\s+ты\s+умеешь|помощь|меню)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

OPERATOR_WORDS = [
    "оператор", "живой человек", "живого человека", "специалист",
    "консультант", "позовите", "соедините", "human", "менеджер",
]

THANKS_RE = re.compile(r"^\s*(спасибо|благодарю|спс|пасиб\w*)[\s!.)]*$", re.IGNORECASE)


@dataclass
class Button:
    label: str
    payload: str


@dataclass
class Source:
    npa: str
    urls: List[str]

    def render(self) -> str:
        parts = []
        if self.npa:
            parts.append(f"📄 Источник: {self.npa}")
        for url in self.urls:
            parts.append(f"🔗 {url}")
        return "\n".join(parts)


@dataclass
class Reply:
    kind: str
    text: str
    intent: str = "other"
    confidence: float = 0.0
    doc_id: Optional[str] = None
    source: Optional[Source] = None
    buttons: List[Button] = field(default_factory=list)
    debug: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        if self.source and self.source.render():
            return f"{self.text}\n\n{self.source.render()}"
        return self.text


class Assistant:
    def __init__(self, kb_dir: Optional[str] = None):
        self.documents: List[Document] = (
            load_documents(kb_dir) if kb_dir else load_documents()
        )
        self.by_id: Dict[str, Document] = {d.id: d for d in self.documents}
        self.clarify_rules = load_clarify_rules(kb_dir) if kb_dir else load_clarify_rules()
        self.retriever = HybridRetriever(self.documents)

    # ------------------------------------------------------------------
    # публичный вход
    # ------------------------------------------------------------------
    def ask(self, query: str, state: Optional[dict] = None) -> Reply:
        state = state if state is not None else {}
        query = (query or "").strip()

        if not query:
            return self._meta_reply()

        # 2. пользователь нажал кнопку уточнения
        if query.startswith("clarify:"):
            return self._resolve_clarify_payload(query, state)

        # 1. команды и приветствия
        if GREETING_RE.match(query):
            state.pop("pending_clarify", None)
            return self._meta_reply()

        if THANKS_RE.match(query):
            return Reply(
                kind="smalltalk",
                text="Пожалуйста! Если появятся вопросы по транспорту — пишите.",
                intent="meta",
                confidence=1.0,
            )

        # 2b. текстовый ответ на заданное уточнение («городской», «школьник»)
        pending = state.get("pending_clarify")
        if pending:
            resolved = self._match_pending(query, pending)
            if resolved:
                state.pop("pending_clarify", None)
                return self._answer_with_doc(resolved, confidence=0.9, kind="answer")
            # иначе трактуем как новый вопрос
            state.pop("pending_clarify", None)

        # 3. просьба позвать оператора
        if contains_any(query, OPERATOR_WORDS):
            return self._escalate(
                "Передаю обращение оператору.",
                intent="escalation",
            )

        # 4. другой регион
        region_hit = self._other_region(query)
        if region_hit:
            return Reply(
                kind="out_of_scope",
                text=(
                    f"Я отвечаю только по общественному транспорту Тульской области, "
                    f"а вопрос касается другого региона ({region_hit}). "
                    f"По правилам и тарифам другого региона нужно обращаться к местному перевозчику."
                ),
                intent="out_of_scope",
                confidence=1.0,
            )

        # 5–9. поиск и политика
        hits = self.retriever.search(query, top_k=5)
        debug = [h.explain() for h in hits]
        if not hits:
            return self._no_data(query, debug)

        top = hits[0]
        if top.score < settings.threshold_refuse or self._out_of_domain(top):
            return self._no_data(query, debug, hits=hits)

        # 5b. выбор варианта внутри интента (город/пригород, школьник/взрослый)
        top = self._resolve_variant(query, top)

        # 6. не хватает существенного параметра
        if settings.clarify_enabled:
            clarify = self._need_slot(query, top)
            if clarify:
                state["pending_clarify"] = {
                    "intent": top.doc.intent,
                    "options": clarify["options"],
                }
                return Reply(
                    kind="clarify",
                    text=clarify["question"],
                    intent=top.doc.intent,
                    confidence=top.score,
                    buttons=[
                        Button(o["label"], f"clarify:{o['doc']}") for o in clarify["options"]
                    ],
                    debug=debug,
                )

            # 7. два разных интента вплотную
            if len(hits) > 1:
                second = hits[1]
                if (
                    second.doc.intent != top.doc.intent
                    and (top.score - second.score) < settings.min_margin
                    and top.score < 0.45
                ):
                    state["pending_clarify"] = {
                        "intent": top.doc.intent,
                        "options": [
                            {"label": h.doc.title, "doc": h.doc.id, "keywords": h.doc.keywords}
                            for h in (top, second)
                        ],
                    }
                    return Reply(
                        kind="clarify",
                        text="Уточните, какой вопрос имеется в виду:",
                        intent=top.doc.intent,
                        confidence=top.score,
                        buttons=[
                            Button(h.doc.title, f"clarify:{h.doc.id}") for h in (top, second)
                        ],
                        debug=debug,
                    )

        # 8. серая зона: отвечаем не наугад, а предлагаем формулировки
        if top.score < settings.threshold_answer:
            options = [h for h in hits[:3] if h.score >= settings.threshold_refuse]
            state["pending_clarify"] = {
                "intent": top.doc.intent,
                "options": [
                    {"label": h.doc.title, "doc": h.doc.id, "keywords": h.doc.keywords}
                    for h in options
                ],
            }
            listing = "\n".join(f"• {h.doc.title}" for h in options)
            return Reply(
                kind="clarify",
                text=(
                    "Не уверен, что понял вопрос правильно. Уточните, что именно интересует:\n"
                    f"{listing}\n\nИли переформулируйте вопрос — отвечу точнее."
                ),
                intent=top.doc.intent,
                confidence=top.score,
                buttons=[Button(h.doc.title, f"clarify:{h.doc.id}") for h in options]
                + [Button("Связаться с оператором", "operator")],
                debug=debug,
            )

        # 9. уверенный ответ
        return self._answer_with_doc(
            top.doc, confidence=top.score, kind="answer", question=query, hits=hits, debug=debug
        )

    # ------------------------------------------------------------------
    # вспомогательные шаги
    # ------------------------------------------------------------------
    def _answer_with_doc(
        self,
        doc: Document,
        confidence: float,
        kind: str,
        question: str = "",
        hits: Optional[List[Hit]] = None,
        debug: Optional[List[str]] = None,
    ) -> Reply:
        text = doc.answer

        # необязательная генерация LLM поверх контекста, с проверкой заземления
        if settings.llm_enabled and question:
            contexts = [d.answer for d in self._context_docs(doc, hits)]
            generated = llm.generate(question, contexts)
            if generated:
                text = generated

        if doc.grounding == "none":
            text += (
                "\n\n⚠️ Отдельной нормы по этому вопросу в доступных источниках нет — "
                "ответ основан на официальных разъяснениях и технической логике системы."
            )

        return Reply(
            kind=kind,
            text=text,
            intent=doc.intent,
            confidence=confidence,
            doc_id=doc.id,
            source=Source(npa=doc.npa, urls=doc.urls),
            buttons=[Button("Связаться с оператором", "operator")],
            debug=debug or [],
        )

    def _context_docs(self, doc: Document, hits: Optional[List[Hit]]) -> List[Document]:
        docs = [doc]
        for h in hits or []:
            if h.doc.id != doc.id and h.score >= settings.threshold_refuse:
                docs.append(h.doc)
        return docs[:3]

    def _out_of_domain(self, top: Hit) -> bool:
        """Совпадение есть, но не с сутью вопроса — отвечать нельзя."""
        if not top.on_topic and top.score < settings.ontopic_score_max:
            # ни одного термина предметной области: «сколько стоит бензин»
            return True
        return (
            top.score < settings.ood_score_max
            and top.domain < settings.ood_domain_min
        )

    def _resolve_variant(self, query: str, top: Hit) -> Hit:
        """Для интентов с вариантами выбирает нужный документ по признаку.

        Если дискриминирующего признака в запросе нет — отдаём сводный
        документ, который покрывает все варианты сразу: переспрашивать
        там, где можно ответить полностью, значит тратить ход пассажира.
        """
        rule = self.clarify_rules.get("variants", {}).get(top.doc.intent)
        if not rule:
            return top
        for option in rule.get("options", []):
            if any(has_keyword(query, kw) for kw in option.get("keywords", [])):
                doc = self.by_id.get(option["doc"])
                return Hit(doc, top.score, top.surface, top.body, top.coverage, top.unknown, top.matched) if doc else top
        default_doc = self.by_id.get(rule.get("default", ""))
        if default_doc:
            return Hit(default_doc, top.score, top.surface, top.body, top.coverage, top.unknown, top.matched)
        return top

    def _need_slot(self, query: str, top: Hit) -> Optional[dict]:
        """Уточнение там, где без параметра ответ будет неверным."""
        rule = self.clarify_rules.get("slots", {}).get(top.doc.intent)
        if not rule:
            return None
        low = query.lower().replace("ё", "е")
        if not any(trigger in low for trigger in rule.get("trigger", [])):
            return None
        if any(has_keyword(low, skip) for skip in rule.get("skip_if", [])):
            return None
        return rule

    def _match_pending(self, query: str, pending: dict) -> Optional[Document]:
        low = query.lower().replace("ё", "е")
        for option in pending.get("options", []):
            if option["label"].lower() in low or low in option["label"].lower():
                return self.by_id.get(option["doc"])
            for kw in option.get("keywords", []):
                if kw and has_keyword(low, kw):
                    return self.by_id.get(option["doc"])
        return None

    def _resolve_clarify_payload(self, payload: str, state: dict) -> Reply:
        state.pop("pending_clarify", None)
        doc_id = payload.split(":", 1)[1]
        doc = self.by_id.get(doc_id)
        if not doc:
            return self._no_data(payload, [])
        return self._answer_with_doc(doc, confidence=0.95, kind="answer")

    def _other_region(self, query: str) -> Optional[str]:
        for marker in settings.other_regions:
            if has_keyword(query, marker):
                return marker.capitalize()
        return None

    def _no_data(
        self, query: str, debug: List[str], hits: Optional[List[Hit]] = None
    ) -> Reply:
        near = [h.doc.title for h in (hits or []) if h.score >= settings.threshold_refuse * 0.7][:3]
        extra = ("\n\nВозможно, вас интересует:\n" + "\n".join(f"• {t}" for t in near)) if near else ""
        return Reply(
            kind="no_data",
            text=(
                "В доступных мне источниках нет подтверждённого ответа на этот вопрос, "
                "поэтому не буду отвечать наугад.\n\n"
                f"Обратитесь к оператору: {settings.operator_name}, телефон {settings.operator_phone} "
                f"({settings.operator_site})." + extra
            ),
            intent="no_data",
            confidence=hits[0].score if hits else 0.0,
            buttons=[Button("Связаться с оператором", "operator")],
            debug=debug,
        )

    def _escalate(self, prefix: str, intent: str) -> Reply:
        contacts = self.by_id.get("contacts.support")
        text = prefix + "\n\n" + (contacts.answer if contacts else "")
        return Reply(
            kind="escalate",
            text=text,
            intent=intent,
            confidence=1.0,
            doc_id="contacts.support",
            source=Source(npa=contacts.npa, urls=contacts.urls) if contacts else None,
        )

    def _meta_reply(self) -> Reply:
        doc = self.by_id.get("meta.capabilities")
        if not doc:
            return Reply(kind="meta", text="Задайте вопрос о транспорте.", intent="meta")
        return Reply(
            kind="meta",
            text=doc.answer,
            intent="meta",
            confidence=1.0,
            doc_id=doc.id,
            buttons=[
                Button("Стоимость проезда", "clarify:fare.city"),
                Button("Льготы", "clarify:benefit.categories"),
                Button("Связаться с оператором", "operator"),
            ],
        )

    # ------------------------------------------------------------------
    def operator_reply(self) -> Reply:
        return self._escalate("Контакты для связи со специалистом:", intent="escalation")
