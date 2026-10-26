#!/usr/bin/env python3
"""Run the quality suite against a server and write a report.

    python3 -m quality.suite --url http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --out evidence/run1
    python3 -m quality.suite ... --only gsm8k tools          # a subset
    python3 -m quality.suite --compare a/replies.json b/replies.json   # two servers' reply sets (ranks, quantizations)

Checks: drafts, behaviour (streaming, multiturn, stop, thinking, copy, max-tokens), tools, json (asked in the prompt),
ifeval, gsm8k, mgsm, mmlu, math500, gpqa (diamond), humaneval (sandboxed), repetition, long-context (needles at the
edges, spread facts), long-tasks (RULER-style multi-key, multi-value, variable tracking, word counts to 1M), soak
(chats resuming long kept states turn after turn beside a long reply, 20 rounds; the server serves them one at a time),
snapshot (the reply set for comparisons), vision, docvision (ChartQA, DocVQA), concurrent, vision-concurrent. Each writes its details to <out>/<check>.json; <out>/report.md and report.json hold the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import (behaviour, docvision, exactness, gpqa, gsm8k, humaneval, ifeval, json_schema, longtasks, math500, mgsm, mmlu,
               needles, repetition, soak, tools, vision)
from .common import Client, Result

CHECKS = ("drafts", "behaviour", "tools", "json", "ifeval", "gsm8k", "mgsm", "mmlu", "math500", "gpqa", "humaneval",
          "repetition", "long-context", "long-tasks", "soak", "snapshot", "vision", "docvision", "concurrent",
          "vision-concurrent")


def run_check(name: str, client: Client, out: Path, a) -> Result:
    if name == "drafts":
        return exactness.drafts(client)
    if name == "tools":
        return tools.run(client, workers=a.workers)
    if name == "json":
        return json_schema.run(client, guided=a.guided_json, workers=a.workers)
    if name == "ifeval":
        return ifeval.run(client, n=a.ifeval_n, workers=a.workers)
    if name == "gsm8k":
        return gsm8k.run(client, n=a.gsm8k_n, workers=a.workers, thinking=a.thinking, reasoning_budget=a.reasoning_budget)
    if name == "repetition":
        return repetition.run(client, workers=a.workers)
    if name == "long-context":
        return needles.run(client, max_tokens=a.long_max)
    if name == "soak":
        return soak.run(client, chats=a.soak_chats, rounds=a.soak_rounds, background=bool(a.soak_background))
    if name == "snapshot":
        return exactness.snapshot(client, out / "replies.json")
    if name == "mgsm":
        return mgsm.run(client, n=a.mgsm_n, workers=a.workers)
    if name == "mmlu":
        return mmlu.run(client, per_subject=a.mmlu_per_subject, workers=a.workers)
    if name == "math500":
        return math500.run(client, n=a.math_n, workers=a.workers, thinking=a.thinking)
    if name == "gpqa":
        return gpqa.run(client, workers=a.workers, thinking=a.thinking)
    if name == "long-tasks":
        return longtasks.run(client, max_tokens=a.long_max)
    if name == "docvision":
        return docvision.run(client, n=a.docvision_n, workers=a.workers)
    if name == "humaneval":
        return humaneval.run(client, image=a.sandbox_image, workers=a.workers)
    if name == "behaviour":
        return behaviour.run(client)
    if name == "vision":
        return vision.run(client)
    if name == "concurrent":
        return exactness.concurrency(client, streams=a.streams)
    if name == "vision-concurrent":
        return vision.concurrent(client, streams=a.streams)
    raise ValueError(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="default")
    ap.add_argument("--out", type=Path, default=Path("quality-out"))
    ap.add_argument("--only", nargs="+", choices=CHECKS)
    ap.add_argument("--skip", nargs="+", choices=CHECKS, default=[])
    ap.add_argument("--workers", type=int, default=16, help="parallel requests where a check allows it (the server decodes 16 at once)")
    ap.add_argument("--streams", type=int, default=16, help="streams the concurrency checks send at once")
    ap.add_argument("--gsm8k-n", type=int, default=250)
    ap.add_argument("--reasoning-budget", type=int, default=49152, help="GSM8K with --thinking: reasoning tokens a reply may take before its answer")
    ap.add_argument("--ifeval-n", type=int, default=150)
    ap.add_argument("--mgsm-n", type=int, default=30, help="problems per MGSM language")
    ap.add_argument("--long-max", type=int, default=0, help="skip long-context and long-tasks documents above this many tokens (0: all, to 1M)")
    ap.add_argument("--mmlu-per-subject", type=int, default=6)
    ap.add_argument("--math-n", type=int, default=500, help="MATH-500 problems (in order)")
    ap.add_argument("--docvision-n", type=int, default=200, help="ChartQA and DocVQA questions each (spread over the files)")
    ap.add_argument("--sandbox-image", default=None, help="image HumanEval code runs in (no network); default: recipe.toml's image")
    ap.add_argument("--soak-chats", type=int, default=3, help="chats resuming long kept states at once")
    ap.add_argument("--soak-background", type=int, choices=(0, 1), default=1, help="a long decode beside each round")
    ap.add_argument("--soak-rounds", type=int, default=20)
    ap.add_argument("--thinking", action="store_true", help="GSM8K, MATH-500 and GPQA with thinking on")
    ap.add_argument("--guided-json", action="store_true", help="json through response_format (not served by this recipe yet)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    a = ap.parse_args(argv)
    if a.compare:
        r = exactness.compare(*a.compare)
        print(r.line())
        return 0 if r.score == 1.0 else 1
    a.out.mkdir(parents=True, exist_ok=True)
    client = Client(a.url, a.model)
    names = [c for c in (a.only or CHECKS) if c not in a.skip]
    results: list[Result] = []
    print(f"quality suite: {a.url} as {a.model}; checks: {', '.join(names)}")
    for name in names:
        t0 = time.time()
        try:
            r = run_check(name, client, a.out, a)
        except Exception as exc:                 # noqa: BLE001
            r = Result(name, 0.0, 0, 0, time.time() - t0, [f"failed: {str(exc)[:100]}"])
        for one in (r if isinstance(r, list) else [r]):      # behaviour returns one result per sub-check
            results.append(one)
            print(one.line(), flush=True)
        (a.out / f"{name}.json").write_text(json.dumps([x.as_dict() for x in r] if isinstance(r, list) else r.as_dict(), indent=1))
    lines = ["| Check | Passed | Score | Time | Note |", "|---|---:|---:|---:|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.passed}/{r.cases} | {100 * r.score:.1f}% | {r.seconds:.0f}s | {r.notes[0] if r.notes else ''} |")
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    (a.out / "report.json").write_text(json.dumps({"url": a.url, "model": a.model, "results": [r.as_dict() for r in results]}, indent=1))
    print(f"report: {a.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
