"""Long-context tasks past retrieval (after NVIDIA's RULER): the model must find, follow or count across the window.

Every document is generated from a seed (needles.document's filler), so no dataset is needed and a run repeats:

* mk-niah: eight keys, each with a 7-digit number, spread through the document; one key is asked for (the others
  distract). One case.
* multi-value: one key given four different numbers at four depths; all four asked for. Four cases.
* var-track: a chain of five assignments (VAR A = 31415, VAR B = VAR A, ...) spread through the document, with a
  decoy chain; every variable that holds the value asked for. Five cases (one per variable in the chain, none for
  the decoys named).
* common-words: a long numbered word list where ten words recur far more often than the rest; the ten asked for.
  Ten cases.

Lengths reach the 1M window. At the two longest only mk-niah and var-track run (a 1M prompt takes minutes to fill).
The sparse indexer picks which tokens attention reads past 2k; these are its checks across the window.
"""

from __future__ import annotations

import random
import re
import time

from .common import Client, Result, strip_think
from .needles import document

WORDS = ("amber anchor apron arrow badge basil beacon bison bracket breeze cabin cactus candle canyon cedar cobalt comet "
         "copper coral cradle crater crystal dagger delta dune ember falcon fennel ferry fiddle flint forge fossil galaxy "
         "garnet glacier granite harbor hazel helmet heron hollow island ivory jasper kettle lantern lava ledger lily "
         "linen lotus magnet maple marble meadow mesa mortar nectar nickel nutmeg oasis onyx orbit otter paddle pebble "
         "pepper pillar pine plaza pocket prism quartz quill raven reef ribbon ridge saddle sage satin scarf shadow "
         "sierra silver sphinx spruce summit swallow tango thistle thunder timber topaz tundra velvet violet walnut "
         "willow zephyr").split()
KEYS = ("blue-heron", "quiet-forge", "amber-tide", "silver-pass", "cedar-gate", "north-lamp", "iron-brook", "glass-fern")
CHAIN = ("XKQ", "PLM", "RTV", "ZWB", "HNC")
DECOY = ("DFG", "SJY", "MUE")


def _place(paras: list[str], items: list[tuple[float, str]]) -> str:
    for depth, text in sorted(items, key=lambda x: -x[0]):            # deepest first: earlier indices stay put
        paras.insert(int(len(paras) * depth), text + "\n\n")
    return "".join(paras)


def _ask(client: Client, doc: str, question: str, max_tokens: int) -> tuple[str, int, float]:
    prompt = "Read this document carefully.\n\n" + doc + "\n\nQuestion: " + question
    d = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return strip_think(d["choices"][0]["message"].get("content") or ""), d.get("usage", {}).get("prompt_tokens") or 0, d["_seconds"]


def mk_niah(client: Client, tokens: int, seed: int) -> dict:
    rng = random.Random(seed)
    vals = {k: rng.randint(1_000_000, 9_999_999) for k in KEYS}
    want = KEYS[seed % len(KEYS)]
    items = [((i + 0.5) / len(KEYS), f"One of the special magic numbers for {k} is: {vals[k]}.") for i, k in enumerate(rng.sample(KEYS, len(KEYS)))]
    text, pt, s = _ask(client, _place(document(tokens, seed), items),
                       f"What is the special magic number for {want} mentioned in the document? Answer with the number only.", 24)
    hit = str(vals[want]) in re.sub(r"[^0-9]", " ", text).split()
    return {"task": "mk-niah", "tokens": pt, "seconds": s, "passed": int(hit), "cases": 1, "reply": text[:80]}


def multi_value(client: Client, tokens: int, seed: int) -> dict:
    rng = random.Random(seed)
    key = KEYS[(seed + 3) % len(KEYS)]
    vals = rng.sample(range(1_000_000, 9_999_999), 4)
    items = [(d, f"One of the special magic numbers for {key} is: {v}.") for d, v in zip((0.12, 0.38, 0.63, 0.88), vals)]
    text, pt, s = _ask(client, _place(document(tokens, seed), items),
                       f"What are all the special magic numbers for {key} mentioned in the document? List each number "
                       "once, separated by commas, nothing else.", 64)
    got = set(re.sub(r"[^0-9]", " ", text).split())
    hits = sum(1 for v in vals if str(v) in got)
    return {"task": "multi-value", "tokens": pt, "seconds": s, "passed": hits, "cases": 4, "reply": text[:120]}


def var_track(client: Client, tokens: int, seed: int) -> dict:
    rng = random.Random(seed)
    value, decoy = rng.randint(10_000, 99_999), rng.randint(10_000, 99_999)
    items = [(0.05 + 0.9 * i / len(CHAIN), (f"VAR {v} = {value}" if i == 0 else f"VAR {v} = VAR {CHAIN[i - 1]}") + ".")
             for i, v in enumerate(CHAIN)]
    items += [(0.1 + 0.8 * i / len(DECOY) + 0.03, (f"VAR {v} = {decoy}" if i == 0 else f"VAR {v} = VAR {DECOY[i - 1]}") + ".")
              for i, v in enumerate(DECOY)]
    text, pt, s = _ask(client, _place(document(tokens, seed), items),
                       f"Find every variable that is assigned the value {value}, directly or through other variables. "
                       "List the variable names only, separated by commas.", 64)
    names = set(re.findall(r"\b[A-Z]{3}\b", text))
    hits = sum(1 for v in CHAIN if v in names) - sum(1 for v in DECOY if v in names)
    return {"task": "var-track", "tokens": pt, "seconds": s, "passed": max(0, hits), "cases": len(CHAIN), "reply": text[:120]}


def common_words(client: Client, tokens: int, seed: int) -> dict:
    rng = random.Random(seed)
    top = rng.sample(WORDS, 10)
    rest = [w for w in WORDS if w not in top]
    n = int(tokens / 4.5)                                      # "1234. word " is about 4.5 tokens
    # the top ten each take 3% of the list; the rest share what is left (each well under 1%)
    seq = [top[i % 10] if rng.random() < 0.30 else rng.choice(rest) for i in range(n)]
    doc = "".join(f"{i + 1}. {w} " for i, w in enumerate(seq))
    text, pt, s = _ask(client, doc, "Above is a numbered list of words. What are the 10 most common words in the list? "
                                    "Answer with the ten words only, separated by commas.", 96)
    got = set(re.findall(r"[a-z]+", text.lower()))
    hits = sum(1 for w in top if w in got)
    return {"task": "common-words", "tokens": pt, "seconds": s, "passed": hits, "cases": 10, "reply": text[:160]}


# (target tokens, tasks): every task to 128k, the retrieval and tracking tasks at 512k and the full window
LENGTHS = ((32_000, "all"), (128_000, "all"), (512_000, "long"), (900_000, "long"))
TASKS = {"all": (mk_niah, multi_value, var_track, common_words), "long": (mk_niah, var_track)}


def run(client: Client, *, max_tokens: int | None = None) -> Result:
    """``max_tokens`` caps the document lengths (``--long-max``), as the needles check does."""
    t0 = time.time()
    rows = []
    for i, (n, which) in enumerate(LENGTHS):
        if max_tokens and n > max_tokens:
            continue
        for j, task in enumerate(TASKS[which]):
            try:
                rows.append(task(client, n, seed=101 + 17 * i + j))
            except Exception as exc:                       # noqa: BLE001 (a refused or failed prompt scores 0)
                rows.append({"task": task.__name__.replace("_", "-"), "tokens": n, "seconds": 0, "passed": 0,
                             "cases": {"mk_niah": 1, "multi_value": 4, "var_track": 5, "common_words": 10}[task.__name__],
                             "reply": f"failed: {str(exc)[:80]}"})
    passed = sum(r["passed"] for r in rows)
    cases = sum(r["cases"] for r in rows)
    note = ", ".join(f"{r['task']} {r['passed']}/{r['cases']}@{max(r['tokens'], 1) // 1000}k" for r in rows)
    return Result("long-tasks", passed / cases if cases else 0, cases, passed, time.time() - t0, [note], {"rows": rows})
