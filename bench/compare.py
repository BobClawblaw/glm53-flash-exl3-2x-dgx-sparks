#!/usr/bin/env python3
"""Two measurement files side by side, with the difference in percent: the README's comparison tables.

    python3 bench/compare.py ruler OURS.json THEIRS.json [--names "ours" "theirs"] [--field per_stream|aggregate|wall]
    python3 bench/compare.py night OURS.json THEIRS.json [--names ...]

``ruler``: two files written by bench/ruler.py --json (cells by concurrency and phase).
``night``: two files written by tonyd2wild's bench_tp2_night.py (single-stream decode by prompt, C1..C6 aggregate,
cold prefill, long context). The percent column is (ours - theirs) / theirs; a tool-call row compares end to end
(its decode rate is meaningless on a server that sends the call as one chunk). Stdlib only; prints Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys


def pct(a, b) -> str:
    if a is None or b is None or not b:
        return "n/a"
    d = (a - b) / b * 100
    return f"{d:+.1f}%"


def num(x, digits=1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def ruler(ours: dict, theirs: dict, names: list[str], field: str) -> None:
    ta = {(c["concurrency"], c["phase"]): c for c in ours["cells"]}
    tb = {(c["concurrency"], c["phase"]): c for c in theirs["cells"]}
    print(f"| streams | phase | {names[0]} | {names[1]} | difference |")
    print("|---|---|---|---|---|")
    order = {k: i for i, k in enumerate(list(ta) + [k for k in tb if k not in ta])}  # the files' own phase order
    for key in sorted(set(ta) | set(tb), key=lambda k: (k[0], order[k])):
        a = ta.get(key, {}).get(field)
        b = tb.get(key, {}).get(field)
        print(f"| {key[0]} | {key[1]} | {num(a)} | {num(b)} | {pct(a, b)} |")
    print(f"\n{field} tok/s; ours {ours.get('runs')} run(s), theirs {theirs.get('runs')} run(s).")


def night(ours: dict, theirs: dict, names: list[str]) -> None:
    sa, sb = ours.get("single", {}), theirs.get("single", {})
    if sa or sb:
        print(f"| prompt | {names[0]} | {names[1]} | difference |")
        print("|---|---|---|---|")
        for name in [n for n in sa] + [n for n in sb if n not in sa]:
            a, b = sa.get(name, {}), sb.get(name, {})
            key, note = ("e2e_tok_s", " (end to end)") if name == "tooluse" else ("decode_tok_s", "")
            x = a.get(key, {}).get("median")
            y = b.get(key, {}).get("median")
            print(f"| {name}{note} | {num(x)} | {num(y)} | {pct(x, y)} |")
        print("\nSingle stream, median decode tok/s of the harness's repeats, greedy.\n")
    wa, wb = ours.get("sweep", {}), theirs.get("sweep", {})
    if wa or wb:
        print(f"| streams | {names[0]} median / peak | {names[1]} median / peak | difference (median) |")
        print("|---|---|---|---|")
        for level in [l for l in wa] + [l for l in wb if l not in wa]:
            a, b = wa.get(level, {}).get("agg_tok_s", {}), wb.get(level, {}).get("agg_tok_s", {})
            print(f"| {level.upper()} | {num(a.get('median'))} / {num(a.get('peak'))} | {num(b.get('median'))} / {num(b.get('peak'))} | {pct(a.get('median'), b.get('median'))} |")
        print("\nAggregate tok/s over mixed prompts, the harness's rounds.\n")
    pa, pb = ours.get("prefill"), theirs.get("prefill")
    if pa or pb:
        rows_a = {int(k): r for k, r in (pa or {}).items()}  # keyed by the harness's nominal size
        rows_b = {int(k): r for k, r in (pb or {}).items()}
        print(f"| prompt (nominal) | {names[0]} prefill tok/s (TTFT) | {names[1]} prefill tok/s (TTFT) | difference |")
        print("|---|---|---|---|")
        for k in sorted(set(rows_a) | set(rows_b)):
            a, b = rows_a.get(k, {}), rows_b.get(k, {})
            x, y = a.get("prefill_tok_s"), b.get("prefill_tok_s")
            ta_, tb_ = a.get("ttft_s"), b.get("ttft_s")
            print(f"| {k} | {num(x, 0)} ({num(ta_)} s) | {num(y, 0)} ({num(tb_)} s) | {pct(x, y)} |")
        print("\nCold prefill (a unique salt heads every prompt).\n")
    la, lb = ours.get("longctx"), theirs.get("longctx")
    if la or lb:
        print(f"| long context | {names[0]} | {names[1]} | difference |")
        print("|---|---|---|---|")
        for k in sorted(set(la or {}) | set(lb or {})):
            x = (la or {}).get(k, {}).get("per_stream_decode")
            y = (lb or {}).get(k, {}).get("per_stream_decode")
            print(f"| {k} | {num(x)} | {num(y)} | {pct(x, y)} |")
        print("\nDecode tok/s per stream on 32K-token prompts.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["ruler", "night"])
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--names", nargs=2, default=["ours", "theirs"])
    ap.add_argument("--field", default="per_stream", choices=["per_stream", "aggregate", "wall"])
    a = ap.parse_args()
    with open(a.ours) as f:
        ours = json.load(f)
    with open(a.theirs) as f:
        theirs = json.load(f)
    if a.kind == "ruler":
        ruler(ours, theirs, a.names, a.field)
    else:
        night(ours, theirs, a.names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
