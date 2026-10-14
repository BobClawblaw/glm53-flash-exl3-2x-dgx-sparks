"""Real pictures: ChartQA (questions on bar, line and pie charts) and DocVQA (questions on scanned documents).

The vision check draws its own shapes and colours; these read printed numbers, axis labels and small text, where a
vision path that mostly works still loses points. ChartQA is graded by relaxed accuracy (a number within 5% of the
label, else the same text); DocVQA by ANLS (normalised Levenshtein similarity to the nearest answer, 0 below 0.5),
the datasets' own metrics. Both are parquet files with the images inside, pinned to commits (pyarrow reads them).
"""

from __future__ import annotations

import base64
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, fetch, strip_think

CHARTQA = ("https://huggingface.co/datasets/HuggingFaceM4/ChartQA/resolve/b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5/"
           "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet",
           "chartqa-test.parquet", "efcf31333368c485eb569ec7bd4cc3694bd0e8e1a06c503124b51d27c924491d")
DOCVQA = ("https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/resolve/539088ef8a8ada01ac8e2e6d4e372586748a265e/"
          "DocVQA/validation-00000-of-00006.parquet",
          "docvqa-val0.parquet", "a31507f0b700ac64f3ead52057c5dc3ccfb0baadcc62bc0d19d17159f080a4d8")
CHART_PROMPT = "{q}\nAnswer with a single word or number only."
DOC_PROMPT = "{q}\nAnswer with the exact words or number from the document only."


def _rows(src: tuple[str, str, str], n: int) -> list[dict]:
    import pyarrow.parquet as pq                          # noqa: PLC0415 (only these checks need it)
    t = pq.read_table(fetch(*src))
    step = max(1, t.num_rows // n)                        # spread over the file: both of ChartQA's halves
    return [t.slice(i, 1).to_pylist()[0] for i in range(0, t.num_rows, step)][:n]


def _data_url(img: dict) -> str:
    b = img["bytes"]
    mime = "image/png" if b[:4] == b"\x89PNG" else "image/jpeg" if b[:2] == b"\xff\xd8" else "image/webp"
    return f"data:{mime};base64," + base64.b64encode(b).decode()


REFUSED = "(refused: "  # a reply the server refused (HTTP 4xx, e.g. an image over its pixel limit): a miss, counted


def _ask(client: Client, img: dict, text: str, max_tokens: int) -> str:
    try:
        d = client.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(img)}},
                                                      {"type": "text", "text": text}]}], max_tokens=max_tokens)
    except urllib.error.HTTPError as e:  # one refused picture must not void the check
        return f"{REFUSED}HTTP {e.code})"
    reply = strip_think(d["choices"][0]["message"].get("content") or "").strip()
    return reply.splitlines()[-1].strip() if reply else ""


def _num(s: str) -> float | None:
    s = s.strip().rstrip(".").replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def relaxed(pred: str, gold: str) -> bool:
    """ChartQA's relaxed accuracy: within 5% for numbers, else equal ignoring case."""
    p, g = _num(pred), _num(gold)
    if p is not None and g is not None:
        return abs(p - g) <= 0.05 * abs(g) if g != 0 else p == 0
    return pred.strip().strip(".").lower() == gold.strip().lower()


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: list[str], tau: float = 0.5) -> float:
    p = re.sub(r"\s+", " ", pred.strip().lower())
    best = 0.0
    for g in golds:
        g = re.sub(r"\s+", " ", g.strip().lower())
        d = _lev(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - d if d < tau else 0.0)
    return best


def run(client: Client, *, n: int = 200, workers: int = 16, max_tokens: int = 64) -> list[Result]:
    out = []
    t0 = time.time()
    charts = _rows(CHARTQA, n)

    def chart(r: dict) -> tuple[bool, str]:
        got = _ask(client, r["image"], CHART_PROMPT.format(q=r["query"]), max_tokens)
        return relaxed(got, r["label"][0]), got

    with ThreadPoolExecutor(workers) as pool:
        cs = list(pool.map(chart, charts))
    ok = sum(1 for hit, _ in cs if hit)
    human = [hit for r, (hit, _) in zip(charts, cs) if r["human_or_machine"] == 0]
    machine = [hit for r, (hit, _) in zip(charts, cs) if r["human_or_machine"] == 1]
    refused = sum(1 for _, got in cs if got.startswith(REFUSED))
    out.append(Result("chartqa", ok / len(charts), len(charts), ok, time.time() - t0,
                      [f"relaxed accuracy; human-written {sum(human)}/{len(human)}, generated {sum(machine)}/{len(machine)}"
                       + (f"; {refused} refused by the server" if refused else "")],
                      {"misses": [{"q": r["query"], "gold": r["label"][0], "got": got}
                                  for r, (hit, got) in zip(charts, cs) if not hit][:40]}))
    t1 = time.time()
    docs = _rows(DOCVQA, n)

    def doc(r: dict) -> tuple[float, str]:
        got = _ask(client, r["image"], DOC_PROMPT.format(q=r["question"]), max_tokens)
        return anls(got, list(r["answers"])), got

    with ThreadPoolExecutor(workers) as pool:
        ds = list(pool.map(doc, docs))
    mean = sum(s for s, _ in ds) / len(ds)
    good = sum(1 for s, _ in ds if s >= 0.5)
    refused = sum(1 for _, got in ds if got.startswith(REFUSED))
    out.append(Result("docvqa", mean, len(docs), good, time.time() - t1,
                      [f"ANLS {mean:.3f} (score); passed = ANLS >= 0.5" + (f"; {refused} refused by the server" if refused else "")],
                      {"misses": [{"q": r["question"], "gold": list(r["answers"]), "got": got}
                                  for r, (s, got) in zip(docs, ds) if s < 0.5][:40]}))
    return out
