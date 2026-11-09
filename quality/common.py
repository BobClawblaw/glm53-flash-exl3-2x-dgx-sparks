"""Shared pieces of the quality suite: the chat client, dataset fetching with pinned revisions, result records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

CACHE = Path(os.environ.get("QUALITY_CACHE", Path.home() / ".cache" / "glm53-flash-quality"))


class Client:
    """OpenAI-style chat completions against one server; greedy by default, thinking off unless asked."""

    def __init__(self, url: str, model: str = "default", timeout: int = 7200) -> None:
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self._switch: bool | None = None  # the served model's thinking_switch capability, read on first use

    def template_kwargs(self, thinking: bool) -> dict:
        """``chat_template_kwargs`` for a request: thinking off is sent only to a model whose template can switch it
        off (TensorFold's ``thinking_switch`` capability; a server that doesn't say gets the flag). A model with no
        switch, such as Recursive Flint, refuses an explicit off and reasons as it always does without the flag."""
        if thinking:
            return {"enable_thinking": True}
        if self._switch is None:
            self._switch = True
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as r:
                    for m in json.load(r).get("data") or []:
                        if m.get("id") == self.model and (m.get("capabilities") or {}).get("thinking_switch") is False:
                            self._switch = False
            except Exception:
                pass
        return {"enable_thinking": False} if self._switch else {}

    def answer_tokens(self, n: int) -> int:
        """A thinking-off check's ``max_tokens``: ``n`` for the answer, plus room for the reasoning a model with no
        switch writes first (its ``content`` is empty when the reasoning alone fills ``n``)."""
        self.template_kwargs(False)
        return n + 4096 if self._switch is False else n

    def chat(self, messages: list[dict], *, max_tokens: int = 512, temperature: float = 0.0, thinking: bool = False,
             tools: list | None = None, response_format: dict | None = None, draft: bool | None = None,
             extra: dict | None = None) -> dict:
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                      "stream": False, "chat_template_kwargs": self.template_kwargs(thinking)}
        if tools:
            body["tools"] = tools
        if response_format:
            body["response_format"] = response_format
        if draft is not None:
            body["draft"] = draft
        if extra:
            body.update(extra)
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.load(r)
        d["_seconds"] = time.monotonic() - t0
        return d

    def ask(self, prompt: str, **kw) -> str:
        d = self.chat([{"role": "user", "content": prompt}], **kw)
        return text_of(d)


def text_of(d: dict) -> str:
    msg = d["choices"][0]["message"]
    return msg.get("content") or ""


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def fetch(url: str, name: str, sha256: str | None = None) -> Path:
    """A dataset file by URL, cached under QUALITY_CACHE; pinned by a commit in the URL or by ``sha256`` of its bytes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(p)
    if sha256 is not None:
        have = hashlib.sha256(p.read_bytes()).hexdigest()
        if have != sha256:
            raise ValueError(f"{name}: sha256 {have} is not the pinned {sha256}; the source changed, re-pin deliberately")
    return p


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


@dataclass
class Result:
    """One check's outcome: a name, a score (0..1), a count of cases, and notes a reader can act on."""

    name: str
    score: float
    cases: int
    passed: int
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"{self.name:<14} {self.passed:>5}/{self.cases:<5} {100 * self.score:6.1f}%  ({self.seconds:.0f}s)" + \
            (f"  {self.notes[0]}" if self.notes else "")

    def as_dict(self) -> dict:
        return asdict(self)
