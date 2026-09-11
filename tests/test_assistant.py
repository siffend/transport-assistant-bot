"""Тесты ядра. Запуск:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.kb import load_documents  # noqa: E402
from assistant.llm import grounding_check  # noqa: E402
from assistant.pipeline import Assistant  # noqa: E402
from assistant.telegram_bot import build_keyboard  # noqa: E402
from assistant.text import has_keyword, stem, tokenize  # noqa: E402


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.docs = load_documents()

    def test_no_duplicate_ids(self):
        ids = [d.id for d in self.docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_document_has_answer_and_source(self):
        for doc in self.docs:
            with self.subTest(doc=doc.id):
                self.assertTrue(doc.answer.strip(), "пустой ответ")
                self.assertTrue(doc.has_source, "нет ни НПА, ни ссылки")

    def test_all_21_source_rows_covered(self):
        rows = {d.source_row for d in self.docs if d.source_row}
        self.assertEqual(rows, set(range(1, 22)))


class TestText(unittest.TestCase):
    def test_stemmer_groups_word_forms(self):
        self.assertEqual(stem("проездного"), stem("проездной"))
        self.assertEqual(stem("списания"), stem("списание"))

    def test_synonyms_normalize_colloquial(self):
        self.assertIn("стоимост", tokenize("скока стоит"))
        self.assertIn("приложен", tokenize("какая прога"))

    def test_keyword_respects_word_start(self):
        self.assertTrue(has_keyword("еду в город", "город"))
        self.assertFalse(has_keyword("еду в пригород", "город"))


class TestPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = Assistant()

    def test_answers_with_source(self):
        r = self.a.ask("сколько стоит проезд", {})
        self.assertEqual(r.kind, "answer")
        self.assertTrue(r.source.npa or r.source.urls)
        self.assertIn("36", r.text)

    def test_refuses_when_no_data(self):
        r = self.a.ask("сколько стоит бензин на заправке", {})
        self.assertEqual(r.kind, "no_data")
        self.assertIn("оператор", r.text.lower())

    def test_other_region_is_out_of_scope(self):
        r = self.a.ask("сколько стоит проезд в Костроме", {})
        self.assertEqual(r.kind, "out_of_scope")

    def test_clarifies_when_category_unknown(self):
        r = self.a.ask("мне положена льгота?", {})
        self.assertEqual(r.kind, "clarify")
        self.assertTrue(r.buttons)

    def test_clarification_button_resolves_to_answer(self):
        state = {}
        first = self.a.ask("мне положена льгота?", state)
        payload = first.buttons[0].payload
        second = self.a.ask(payload, state)
        self.assertEqual(second.kind, "answer")
        self.assertTrue(second.doc_id)

    def test_variant_selected_by_context(self):
        self.assertEqual(self.a.ask("стоимость проезда в городе", {}).doc_id, "fare.city")
        self.assertEqual(
            self.a.ask("стоимость проезда в пригородном автобусе", {}).doc_id, "fare.suburban"
        )

    def test_operator_request_escalates(self):
        r = self.a.ask("позовите оператора", {})
        self.assertEqual(r.kind, "escalate")
        self.assertIn("705-507", r.text)

    def test_grounding_note_for_documents_without_npa(self):
        r = self.a.ask("почему поздно приходит уведомление об оплате", {})
        self.assertIn("нет", r.text.lower())

    def test_paraphrase_matches_same_document(self):
        pairs = [
            ("Что делать, если потерял карту с льготным абонементом", "потерял соцкарту что делать"),
            ("Почему оплата списалась несколько раз", "деньги сняли два раза за поездку"),
        ]
        for canonical, colloquial in pairs:
            with self.subTest(q=colloquial):
                self.assertEqual(
                    self.a.ask(canonical, {}).doc_id, self.a.ask(colloquial, {}).doc_id
                )


class TestLLMGuards(unittest.TestCase):
    def test_grounding_rejects_invented_numbers(self):
        ok, bad = grounding_check("Проезд стоит 99 рублей", "Проезд стоит 36 рублей")
        self.assertFalse(ok)
        self.assertIn("99", bad)

    def test_grounding_accepts_numbers_from_context(self):
        ok, _ = grounding_check("Стоит 36 ₽ по карте", "Проезд 36 ₽ по транспортной карте")
        self.assertTrue(ok)


class TestTelegramLayer(unittest.TestCase):
    def test_keyboard_built_from_buttons(self):
        a = Assistant()
        reply = a.ask("мне положена льгота?", {})
        kb = build_keyboard(reply)
        self.assertTrue(kb)
        self.assertIn("callback_data", kb[0][0])
        self.assertLessEqual(len(kb[0][0]["callback_data"]), 64)


if __name__ == "__main__":
    unittest.main()
