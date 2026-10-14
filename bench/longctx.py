#!/usr/bin/env python3
"""Long prompts at exact token counts: 64K, 128K and 256K by default. Each is a repeatable document (two facts buried
at a third and two thirds) and a question about them; the server's `/tokenize` sizes the document to the target
within 1%. Reports the prompt tokens, the time to the first token (the prefill rate), the decode rate after it, and
whether the reply has both facts. `--streams N` sends N such prompts at once (different documents). Writes
`longctx-<tokens>[-sN].json` to `--out`. Stdlib only.

usage: longctx.py --url http://127.0.0.1:8888 --out evidence/<day>/longctx [--sizes 65536,131072,262144] [--streams 1]
"""

import argparse
import json
import random
import threading
import time
import urllib.request

WORDS = ("harbor lantern meadow copper signal orbit candle marble thunder river stone cloud garden violet forest engine "
         "ledger compass glacier pepper saddle quarry ribbon timber velvet walnut anchor bramble cinder").split()
QUESTION = ("\n\nTwo facts are stated somewhere in the notes above: a vault code and the minute past the hour at which "
            "the ferry leaves. Give the vault code and that minute first, then describe the notes in about 150 words.")


def document(words: int, seed: int) -> str:
    rng = random.Random(seed)
    body = [rng.choice(WORDS) for _ in range(words)]
    body.insert(words // 3, "The vault code is 4471.")
    body.insert(2 * words // 3, "The ferry leaves at 17 past the hour.")
    return " ".join(body)


def post(url: str, path: str, body: dict, timeout: float):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def count(url: str, model: str, text: str) -> int:
    with post(url, "/tokenize", {"model": model, "messages": [{"role": "user", "content": text}]}, 600) as r:
        return len(json.load(r)["tokens"])


def sized(url: str, model: str, target: int, seed: int) -> tuple[str, int]:
    """A document whose rendered prompt is within 1% of ``target`` tokens (two rounds of /tokenize)."""
    probe = 4000
    ratio = count(url, model, document(probe, seed) + QUESTION) / probe
    words = int(target / ratio)
    for _ in range(3):
        text = document(words, seed) + QUESTION
        n = count(url, model, text)
        if abs(n - target) <= target // 100:
            return text, n
        words = int(words * target / n)
    return text, n


def ask(url: str, model: str, text: str, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    first = None
    out = []
    usage = None
    with post(url, "/v1/chat/completions", body, timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                piece = c.get("delta", {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    out.append(piece)
    end = time.time()
    reply = "".join(out)
    completion = usage["completion_tokens"] if usage else len(out)
    prompt = usage["prompt_tokens"] if usage else None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if usage else 0
    ttft = (first or end) - t0
    decode = (completion - 1) / (end - first) if first and end > first and completion > 1 else 0.0
    return {"prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": completion, "ttft_s": round(ttft, 2),
            "prefill_tok_s": round(prompt / ttft, 1) if prompt and ttft > 0 else None, "decode_tok_s": round(decode, 1),
            "wall_s": round(end - t0, 2), "needle": ("4471" in reply and "17" in reply), "reply": reply[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--sizes", default="65536,131072,262144")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    for size in [int(x) for x in a.sizes.split(",")]:
        texts = [sized(a.url, a.model, size, 11 + 100 * a.streams + i) for i in range(a.streams)]  # no prompt shared with another run
        results = [None] * a.streams
        def run(i: int) -> None:
            results[i] = ask(a.url, a.model, texts[i][0], a.max_tokens, a.timeout)
        t0 = time.time()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(a.streams)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        row = {"target_tokens": size, "streams": a.streams, "wall_s": round(wall, 2), "results": results}
        name = f"longctx-{size}" + (f"-s{a.streams}" if a.streams > 1 else "")
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(row, f, indent=1)
        for i, r in enumerate(results):
            print(f"{size:>7} tokens{(' stream ' + str(i)) if a.streams > 1 else ''}: prompt {r['prompt_tokens']} (cached {r['cached_tokens']}), "
                  f"ttft {r['ttft_s']} s = {r['prefill_tok_s']} tok/s prefill, decode {r['decode_tok_s']} tok/s over "
                  f"{r['completion_tokens']} tokens, needle {'found' if r['needle'] else 'MISSED'}", flush=True)
        if a.streams > 1:
            print(f"{size:>7} tokens x {a.streams} streams: wall {wall:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
