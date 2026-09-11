"""Гибридный поиск по базе знаний без внешних зависимостей.

Документ представлен не одним «мешком слов», а набором поверхностей:
формулировка вопроса, заголовок, каждая перефразировка, ключевые слова —
и отдельно тело ответа. Запрос сравнивается с каждой поверхностью, берётся
лучшее совпадение: короткий разговорный вопрос не размывается объёмным
текстом ответа.

Итоговая релевантность — взвешенная сумма ограниченных [0..1] сигналов:

  surface  — лучшее совпадение с формулировкой (слова + символьные 3-граммы);
  body     — совпадение с телом ответа;
  coverage — доля «веса» запроса (по idf), покрытая документом;
  штраф    — за долю лексики запроса, которой нет в базе знаний вообще
             (именно он отделяет «не про транспорт» от «плохо сформулировано»).

Шкала ограничена сознательно: по абсолютному значению принимается решение
«отвечать / уточнить / признать нехватку данных», а не только порядок выдачи.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .kb import Document, load_domain_lexicon
from .text import char_ngrams, tokenize

W_SURFACE = 0.45
W_BODY = 0.15
W_COVER = 0.25
W_DOMAIN = 0.15
W_SURF_WORD = 0.65  # внутри поверхности: слова против символьных n-грамм
KEYWORD_BONUS = 0.05
UNKNOWN_PENALTY = 0.15  # мягкий штраф: длинный «человеческий» вопрос не должен наказываться
DOMAIN_REF_TERMS = 2  # сколько «ядерных» терминов считать полным узнаванием темы
PREFIX_LEN = 5        # длина префикса для нечёткого сопоставления основ


@dataclass
class Hit:
    doc: Document
    score: float
    surface: float
    body: float
    coverage: float
    unknown: float
    matched: List[str]
    core: float = 0.0    # доля веса запроса, попавшая в вопрос/ключевые слова документа
    domain: float = 0.0  # абсолютный доменный сигнал: сколько «ядерных» терминов узнано
    on_topic: bool = True  # в запросе есть хотя бы один термин предметной области

    # обратная совместимость с прежними полями
    @property
    def cos_word(self) -> float:
        return self.surface

    @property
    def cos_char(self) -> float:
        return self.body

    def explain(self) -> str:
        return (
            f"{self.doc.id}: score={self.score:.3f} (surface={self.surface:.3f}, "
            f"body={self.body:.3f}, cover={self.coverage:.3f}, core={self.core:.2f}, "
            f"domain={self.domain:.2f}, on_topic={int(self.on_topic)}, "
            f"unknown={self.unknown:.2f}, "
            f"matched={','.join(self.matched[:6])})"
        )


Vector = Dict[str, float]


def _tfidf(counts: Counter, idf: Dict[str, float], default_idf: float = 0.0) -> Vector:
    vec: Vector = {}
    for term, tf in counts.items():
        w = idf.get(term, default_idf)
        if not w:
            continue
        vec[term] = (1.0 + math.log(tf)) * w
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Vector, b: Vector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _idf(docs_terms: Sequence[Sequence[str]]) -> Dict[str, float]:
    n = len(docs_terms)
    df: Counter = Counter()
    for terms in docs_terms:
        df.update(set(terms))
    return {t: math.log((n + 1) / (c + 0.5)) + 1.0 for t, c in df.items()}


class HybridRetriever:
    def __init__(self, documents: List[Document], domain_terms: Optional[List[str]] = None):
        self.documents = documents
        terms = domain_terms if domain_terms is not None else load_domain_lexicon()
        self._domain_prefixes = {
            t[:PREFIX_LEN] for t in tokenize(" ".join(terms)) if len(t) >= PREFIX_LEN
        } | {t for t in tokenize(" ".join(terms)) if len(t) < PREFIX_LEN}

        doc_word_terms = [tokenize(d.search_text) for d in documents]
        doc_char_terms = [char_ngrams(d.search_text) for d in documents]
        self.word_idf = _idf(doc_word_terms)
        self.char_idf = _idf(doc_char_terms)
        self.max_idf = max(self.word_idf.values()) if self.word_idf else 1.0
        self.mean_idf = (
            sum(self.word_idf.values()) / len(self.word_idf) if self.word_idf else 1.0
        )
        self.vocab = set(self.word_idf)

        self._word_sets = [set(t) for t in doc_word_terms]
        self._keyword_sets = [set(tokenize(" ".join(d.keywords))) for d in documents]
        # «ядро» документа: формулировки вопроса и ключевые слова.
        # Совпадение только с телом ответа — слабый сигнал: длинный ответ
        # случайно содержит много общих слов.
        self._core_sets = [
            set(tokenize(" ".join([d.question, d.title, " ".join(d.paraphrases), " ".join(d.keywords)])))
            for d in documents
        ]
        # Префиксы ядерных терминов: стеммер оставляет «пенсионерк» и
        # «пенсионер» разными основами, поэтому узнавание темы проверяется
        # по совпадению первых пяти букв.
        self._core_prefixes = [
            {t[:PREFIX_LEN] for t in core if len(t) >= PREFIX_LEN}
            for core in self._core_sets
        ]
        self._body_word = [_tfidf(Counter(tokenize(d.answer)), self.word_idf) for d in documents]
        self._surfaces: List[List[Tuple[Vector, Vector]]] = []
        for doc in documents:
            surfaces = [doc.question, doc.title, ", ".join(doc.keywords)]
            surfaces.extend(doc.paraphrases)
            pack = []
            for text in surfaces:
                if not text or not text.strip():
                    continue
                pack.append(
                    (
                        _tfidf(Counter(tokenize(text)), self.word_idf),
                        _tfidf(Counter(char_ngrams(text)), self.char_idf),
                    )
                )
            self._surfaces.append(pack)

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Hit]:
        q_words = tokenize(query)
        if not q_words:
            return []
        q_word_vec = _tfidf(Counter(q_words), self.word_idf, default_idf=self.max_idf)
        q_char_vec = _tfidf(Counter(char_ngrams(query)), self.char_idf, default_idf=1.0)

        q_unique = list(dict.fromkeys(q_words))
        # для очень короткого запроса («потерял стк») достаточно одного
        # узнанного термина, для развёрнутого — минимум двух
        domain_ref = DOMAIN_REF_TERMS if len(q_unique) >= 3 else 1
        q_weights = {t: self.word_idf.get(t, self.max_idf) for t in q_unique}
        total_weight = sum(q_weights.values()) or 1.0
        unknown = sum(w for t, w in q_weights.items() if t not in self.vocab) / total_weight


        on_topic = any(
            (t[:PREFIX_LEN] in self._domain_prefixes if len(t) >= PREFIX_LEN else t in self._domain_prefixes)
            for t in q_unique
        )

        hits: List[Hit] = []
        for i, doc in enumerate(self.documents):
            best_surface = 0.0
            for w_vec, c_vec in self._surfaces[i]:
                sim = W_SURF_WORD * _cosine(q_word_vec, w_vec) + (
                    1 - W_SURF_WORD
                ) * _cosine(q_char_vec, c_vec)
                best_surface = max(best_surface, sim)
            body = _cosine(q_word_vec, self._body_word[i])
            matched = [t for t in q_unique if t in self._word_sets[i]]
            coverage = sum(q_weights[t] for t in matched) / total_weight
            core_terms = [
                t
                for t in q_unique
                if t in self._core_sets[i]
                or (len(t) >= PREFIX_LEN and t[:PREFIX_LEN] in self._core_prefixes[i])
            ]
            core = sum(q_weights[t] for t in core_terms) / total_weight
            # Узнавание темы считаем по числу совпавших «ядерных» терминов,
            # а не по их idf: самая частая лексика предметной области
            # («проезд», «льгота») имеет низкий idf, но именно она и означает,
            # что вопрос наш.
            domain = min(1.0, len(core_terms) / domain_ref)

            score = (
                W_SURFACE * best_surface
                + W_BODY * body
                + W_COVER * coverage
                + W_DOMAIN * domain
            )
            if self._keyword_sets[i] & set(matched):
                score += KEYWORD_BONUS
            score *= 1.0 - UNKNOWN_PENALTY * unknown

            hits.append(
                Hit(
                    doc=doc,
                    score=max(0.0, min(score, 1.0)),
                    surface=best_surface,
                    body=body,
                    coverage=coverage,
                    unknown=unknown,
                    matched=matched,
                    core=core,
                    domain=domain,
                    on_topic=on_topic,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
