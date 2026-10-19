"""MATH-500: competition maths (the PRM800K test subset of Hendrycks' MATH), the last \\boxed{} answer compared after
normalising its LaTeX (the reference's own string rules, plus numeric equality for plain numbers and fractions).

Also accepted (2026-10-19), forms reasoning replies write and the first rules missed:
- a trailing unit in \\text{} or \\mbox{} after a number (`2000 \\text{ calories}`, `864 \\mbox{ inches}^2`);
- `{,}` thousands separators (`58{,}500`);
- set braces around a list (`\\{3,5,7\\}`), a choice letter in parentheses (`(C)` and `C`);
- a bare list of values in any order (`1,-2` and `-2, 1`); a tuple or interval in brackets keeps its order;
- a one-word answer in any case (`East` and `\\text{east}`).
Fixed: `\\ ` (a LaTeX space) was removed even as the second half of a matrix row break `\\\\ `, so a gold matrix
written with spaces matched no answer. Every miss is kept with its answer (it was the first 40), so a run can be
regraded later.

GSM8K sits near its ceiling for this model; these problems separate quantizations where GSM8K cannot.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from .common import Client, Result, fetch, strip_think

# HuggingFaceH4/MATH-500, pinned to a commit
URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl"
SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
PROMPT = "{problem}\n\nSolve the problem step by step. Put the final answer, simplified, in \\boxed{{}}."


def boxed(text: str) -> str | None:
    """The content of the last \\boxed{...} (or \\fbox{...}), braces balanced."""
    at = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if at < 0:
        return None
    i = text.find("{", at)
    if i < 0:
        m = re.match(r"\\boxed\s+(\S+)", text[at:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _fix_fracs(s: str) -> str:
    # \frac12 -> \frac{1}{2}, \frac1{72} -> \frac{1}{72}
    out, i = "", 0
    while i < len(s):
        if s.startswith("\\frac", i) and i + 5 < len(s) and s[i + 5] != "{":
            a = s[i + 5]
            rest = s[i + 6:]
            if rest.startswith("{"):
                out += f"\\frac{{{a}}}"
                i += 6
            elif rest:
                out += f"\\frac{{{a}}}{{{rest[0]}}}"
                i += 7
            else:
                out += s[i:]
                break
        else:
            out += s[i]
            i += 1
    return out


def _unit(m: re.Match) -> str:
    # a trailing \text{words} after something with a digit is a unit: dropped; anything else is kept as it was
    return m.group(1) if re.search(r"\d", m.group(1)) else m.group(0)


def normalise(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(.+?)\s*\\(?:text|mbox)\{\s*[A-Za-z][A-Za-z ]*\}(?:\^\{?\d\}?)?$", _unit, s)
    s = re.sub(r"^\\\{(.*)\\\}$", r"\1", s)                  # \{a,b\} -> a,b
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    for a, b in (("\n", ""), ("{,}", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("tfrac", "frac"), ("dfrac", "frac"),
                 ("\\left", ""), ("\\right", ""), ("^{\\circ}", ""), ("^\\circ", ""), ("\\$", ""), ("$", ""),
                 ("\\%", ""), ("%", ""), ("\\displaystyle", "")):
        s = s.replace(a, b)
    s = re.sub(r"(?<!\\)\\ ", "", s)                           # a LaTeX space, not a row break's second half
    s = re.sub(r"(degrees?|units?|cm|inches|meters?|dollars?)$", "", s.strip())
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)                     # "x = 5" -> "5"
    s = s.replace(" ", "")
    if s.startswith("."):
        s = "0" + s
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)              # 1,000 -> 1000
    s = _fix_fracs(s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = re.sub(r"^\(([A-E])\)$", r"\1", s)                   # (C) -> C
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = re.sub(r"^(-?\d+)\.0+$", r"\1", s)
    return s


def _number(s: str) -> Fraction | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(3)) != 0:
        v = Fraction(int(m.group(2)), int(m.group(3)))
        return -v if m.group(1) else v
    m = re.fullmatch(r"-?\d+/\d+", s)
    if m:
        a, b = s.split("/")
        return Fraction(int(a), int(b)) if int(b) else None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _items(s: str) -> list[str] | None:
    """A bare list's values split at its top-level commas; None for one value or a bracketed tuple or interval."""
    if s[:1] in "([{" or s[-1:] in ")]}":
        return None
    out, depth, at = [], 0, 0
    for i, ch in enumerate(s):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            out.append(s[at:i])
            at = i + 1
    return out + [s[at:]] if out else None


def same(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = normalise(pred), normalise(gold)
    if a == b:
        return True
    if a.isalpha() and b.isalpha():
        return a.lower() == b.lower()                         # a word answer: East, \\text{east}
    xa, xb = _items(a), _items(b)
    if xa and xb and len(xa) == len(xb):
        return sorted(xa) == sorted(xb)
    x, y = _number(a), _number(b)
    return x is not None and y is not None and abs(x - y) <= Fraction(1, 10 ** 6) * max(1, abs(y))


def run(client: Client, *, n: int = 500, workers: int = 16, max_tokens: int = 4096, thinking: bool = False) -> Result:
    rows = [json.loads(line) for line in fetch(URL, "math500-test.jsonl", SHA256).read_text().splitlines() if line.strip()]
    rows = rows[:n]
    t0 = time.time()

    def one(row: dict) -> tuple[bool, str | None]:
        text = strip_think(client.ask(PROMPT.format(problem=row["problem"]), max_tokens=max_tokens, thinking=thinking))
        got = boxed(text)
        return same(got, row["answer"]), got

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, rows))
    ok = sum(1 for hit, _ in outs if hit)
    by_level: dict[int, list[int]] = {}
    for row, (hit, _) in zip(rows, outs):
        v = by_level.setdefault(int(row["level"]), [0, 0])
        v[0] += int(hit)
        v[1] += 1
    unboxed = sum(1 for _, got in outs if got is None)
    levels = " ".join(f"L{lv} {a}/{b}" for lv, (a, b) in sorted(by_level.items()))
    return Result("math500", ok / len(rows), len(rows), ok, time.time() - t0,
                  [levels + (f", {unboxed} without a boxed answer" if unboxed else "")],
                  {"n": len(rows), "thinking": thinking, "by_level": by_level, "unboxed": unboxed,
                   "misses": [{"i": i, "gold": r["answer"], "got": got} for i, (r, (hit, got)) in enumerate(zip(rows, outs))
                              if not hit]})
