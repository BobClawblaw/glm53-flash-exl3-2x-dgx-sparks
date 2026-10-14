"""GPQA-Diamond: 198 graduate-level biology, chemistry and physics questions written to resist lookup, four choices.

The choices are shuffled per question from a fixed seed (the correct one is not always A), as OpenAI's simple-evals
does; the reply's last 'Answer: X' is graded. Chance is 25%. The CSV is simple-evals' public copy, pinned by its hash.
"""

from __future__ import annotations

import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, fetch, strip_think
from .mmlu import letter

URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
PROMPT = ("Answer the following multiple choice question. Think step by step, then give the answer on the last line "
          "as 'Answer: X' where X is one of A, B, C, D.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")


def questions(seed: int = 2026) -> list[dict]:
    rows = list(csv.DictReader(fetch(URL, "gpqa_diamond.csv", SHA256).open(newline="", encoding="utf-8")))
    out = []
    for i, r in enumerate(rows):
        choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(seed + i).shuffle(order)
        shown = [choices[j].strip() for j in order]
        out.append({"q": r["Question"].strip(), "choices": shown, "answer": "ABCD"[order.index(0)],
                    "domain": r.get("High-level domain", "")})
    return out


def run(client: Client, *, workers: int = 16, max_tokens: int = 4096, thinking: bool = False) -> Result:
    rows = questions()
    t0 = time.time()

    def one(r: dict) -> tuple[bool, str | None]:
        a, b, c, d = r["choices"]
        text = strip_think(client.ask(PROMPT.format(q=r["q"], a=a, b=b, c=c, d=d), max_tokens=max_tokens, thinking=thinking))
        got = letter(text)
        return got == r["answer"], got

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, rows))
    ok = sum(1 for hit, _ in outs if hit)
    by: dict[str, list[int]] = {}
    for r, (hit, _) in zip(rows, outs):
        v = by.setdefault(r["domain"] or "?", [0, 0])
        v[0] += int(hit)
        v[1] += 1
    unanswered = sum(1 for _, got in outs if got is None)
    return Result("gpqa", ok / len(rows), len(rows), ok, time.time() - t0,
                  [", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by.items()))
                   + (f", {unanswered} without an answer" if unanswered else "")],
                  {"thinking": thinking, "by_domain": by, "unanswered": unanswered,
                   "answers": [got for _, got in outs]})
