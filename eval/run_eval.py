#!/usr/bin/env python3
"""Оценка качества ассистента на eval/testset.json.

Метрики:
  Doc@1        — доля вопросов, где топ-1 документ совпал с эталонным;
  Doc@3        — эталонный документ в первых трёх результатах поиска;
  Policy       — доля случаев, где выбран верный тип реакции
                 (ответ / уточнение / отказ / вне зоны ответственности);
  Refusal      — доля отказов там, где ответа в базе нет (нет галлюцинаций);
  Source cov.  — доля ответов, снабжённых НПА или ссылкой;
  Latency      — время ответа (медиана и 95-й перцентиль).

Запуск:  python eval/run_eval.py [--verbose] [--json отчет.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.pipeline import Assistant  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWER_KINDS = {"answer", "escalate", "meta"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="показать каждый случай")
    ap.add_argument("--json", dest="json_path", help="сохранить отчёт в JSON")
    ap.add_argument("--testset", default="testset.json", help="файл набора (в каталоге eval/)")
    args = ap.parse_args()

    testset_path = args.testset if os.path.isabs(args.testset) else os.path.join(HERE, args.testset)
    with open(testset_path, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]

    assistant = Assistant()

    stats = defaultdict(lambda: {"n": 0, "doc1": 0, "doc3": 0, "policy": 0, "with_doc": 0})
    latencies = []
    sourced = 0
    answered = 0
    failures = []

    for case in cases:
        query = case["query"]
        group = case.get("group", "other")
        expect_docs = set(case.get("expect_doc", []))
        expect_kind = case.get("expect_kind")

        started = time.perf_counter()
        reply = assistant.ask(query, {})
        latencies.append((time.perf_counter() - started) * 1000)

        row = stats[group]
        row["n"] += 1

        # тип реакции
        kind_ok = True
        if expect_kind:
            if expect_kind == "answer":
                kind_ok = reply.kind in ANSWER_KINDS
            else:
                kind_ok = reply.kind == expect_kind
        row["policy"] += int(kind_ok)

        # точность поиска (только там, где эталон задан)
        doc1_ok = doc3_ok = None
        if expect_docs:
            doc1_ok = reply.doc_id in expect_docs
            top3 = {h.doc.id for h in assistant.retriever.search(query, top_k=3)}
            doc3_ok = bool(top3 & expect_docs) or doc1_ok
            row["with_doc"] += 1
            row["doc1"] += int(doc1_ok)
            row["doc3"] += int(doc3_ok)

        if reply.kind == "answer":
            answered += 1
            if reply.source and (reply.source.npa or reply.source.urls):
                sourced += 1

        if not kind_ok or doc1_ok is False:
            failures.append(
                {
                    "query": query,
                    "group": group,
                    "expect_kind": expect_kind,
                    "got_kind": reply.kind,
                    "expect_doc": sorted(expect_docs),
                    "got_doc": reply.doc_id,
                    "confidence": round(reply.confidence, 3),
                }
            )

        if args.verbose:
            mark = "ok " if kind_ok and doc1_ok is not False else "ERR"
            print(f"[{mark}] {group:13} conf={reply.confidence:.2f} "
                  f"{reply.kind:12} {str(reply.doc_id):24} | {query}")

    total = sum(r["n"] for r in stats.values())
    doc_total = sum(r["with_doc"] for r in stats.values())
    doc1 = sum(r["doc1"] for r in stats.values())
    doc3 = sum(r["doc3"] for r in stats.values())
    policy = sum(r["policy"] for r in stats.values())

    def pct(x: int, n: int) -> str:
        return f"{(100.0 * x / n):5.1f}%" if n else "  n/a"

    print("\n" + "=" * 66)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 66)
    print(f"Всего случаев:            {total}")
    print(f"Doc@1 (поиск документа):  {pct(doc1, doc_total)}  ({doc1}/{doc_total})")
    print(f"Doc@3:                    {pct(doc3, doc_total)}  ({doc3}/{doc_total})")
    print(f"Policy (тип реакции):     {pct(policy, total)}  ({policy}/{total})")
    print(f"Ответы с источником:      {pct(sourced, answered)}  ({sourced}/{answered})")
    print(f"Латентность, мс:          медиана {statistics.median(latencies):.1f}, "
          f"p95 {sorted(latencies)[int(len(latencies) * 0.95) - 1]:.1f}")
    print("-" * 66)
    print(f"{'группа':<16}{'n':>4}{'Doc@1':>10}{'Doc@3':>10}{'Policy':>10}")
    for group in sorted(stats):
        r = stats[group]
        has_doc = r["with_doc"]
        print(f"{group:<16}{r['n']:>4}{pct(r['doc1'], has_doc):>10}"
              f"{pct(r['doc3'], has_doc):>10}{pct(r['policy'], r['n']):>10}")

    if failures:
        print("-" * 66)
        print(f"Расхождения ({len(failures)}):")
        for f in failures:
            print(f"  • «{f['query']}»")
            print(f"    ожидали {f['expect_kind']}/{f['expect_doc']}, "
                  f"получили {f['got_kind']}/{f['got_doc']} (conf={f['confidence']})")

    if args.json_path:
        report = {
            "total": total,
            "doc1": doc1,
            "doc3": doc3,
            "doc_total": doc_total,
            "policy": policy,
            "sourced": sourced,
            "answered": answered,
            "latency_ms_median": round(statistics.median(latencies), 2),
            "groups": {g: dict(r) for g, r in stats.items()},
            "failures": failures,
        }
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\nОтчёт сохранён: {args.json_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
