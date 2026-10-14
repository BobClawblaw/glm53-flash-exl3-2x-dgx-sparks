"""The checkpoint on disk: the pinned snapshot (served as published) and the worker's copy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Settings
from .nodes import Node, log

REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
            "model.safetensors.index.json")


def snapshot_problems(folder: Path) -> list[str]:
    """Why a checkpoint folder is incomplete: missing files, or a shard whose header does not end at its size."""
    bad = [name for name in REQUIRED if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
    index = folder / "model.safetensors.index.json"
    if not index.is_file():
        return bad
    try:
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    except (OSError, ValueError, KeyError) as exc:
        return bad + [f"{index.name} ({exc})"]
    for shard in shards:
        p = folder / shard
        try:
            with p.open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            if 8 + n + end != p.stat().st_size:
                bad.append(f"{shard} (truncated)")
        except (OSError, ValueError, struct.error) as exc:
            bad.append(f"{shard} ({exc.__class__.__name__})")
    return bad


def hf_token(s: Settings) -> str | None:
    """The Hugging Face token the download uses: HF_TOKEN, else the one `hf auth login` stored under hf_cache or
    ~/.cache/huggingface. Never printed, never put on a command line."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    for p in (s.hf_cache / "token", Path.home() / ".cache" / "huggingface" / "token"):
        if p.is_file() and p.read_text().strip():
            return p.read_text().strip()
    return None


GATED_HELP = ("The Ablit weights ({repo}) are gated on Hugging Face.\n"
              "  1. Open https://huggingface.co/{repo}, log in and request access (Mia's AI Lab grants it per account).\n"
              "  2. With the same account, create a token with read access at https://huggingface.co/settings/tokens.\n"
              "  3. Set HF_TOKEN=hf_... in the environment (or `hf auth login`), then `python3 -m spark up` again.\n"
              "  Or set ablit = false (cluster.toml) or ABLIT=0 to serve the published checkpoint.")


def check_gated_access(s: Settings) -> None:
    """Before a gated download: a token is set and reaches the repository's files (one small file, HEAD only).
    401: Hugging Face rejects the token; 403: the account has no access yet (or the token cannot read gated repos)."""
    repo = s.repo
    token = hf_token(s)
    if not token:
        raise RuntimeError("no Hugging Face token (HF_TOKEN unset, nothing stored by `hf auth login`). " + GATED_HELP.format(repo=repo))
    url = f"https://huggingface.co/{repo}/resolve/{s.revision}/config.json"
    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.HTTPError as e:
        why = {401: "Hugging Face does not accept the token (401)",
               403: "the token's account has not been granted access to the repository yet, or a fine-grained token "
                    "lacks read access to gated repositories (403)"}.get(e.code, f"HTTP {e.code}")
        raise RuntimeError(f"cannot reach the gated files of {repo}: {why}. " + GATED_HELP.format(repo=repo)) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach huggingface.co to check access to {repo}: {e.reason}") from None
    log(f"Hugging Face token reaches {repo} (gated)")


def ensure_snapshot(s: Settings, *, download: bool = True) -> None:
    """The pinned revision, complete, under hf_cache; downloaded with the hf CLI when it is not (the Ablit
    weights: after the gated-access check, with the token in the download's environment only)."""
    problems = snapshot_problems(s.snapshot_dir)
    if not problems:
        log(f"Snapshot {s.repo}@{s.revision[:8]} complete ({s.snapshot_dir})")
        return
    if not download:
        raise RuntimeError(f"snapshot incomplete: {', '.join(problems[:6])}")
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        raise RuntimeError(f"no hf CLI on PATH and the snapshot is incomplete ({', '.join(problems[:4])}); "
                           "pip install -U huggingface_hub")
    env = {**os.environ, "HF_HOME": str(s.hf_cache), "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1")}
    if s.ablit:
        check_gated_access(s)
        env["HF_TOKEN"] = hf_token(s) or ""
    log(f"Downloading {s.repo}@{s.revision[:8]} (about 176 GB)")
    rc = subprocess.call([hf, "download", s.repo, "--revision", s.revision], env=env)
    if rc:
        raise RuntimeError(f"hf download failed ({rc})")
    problems = snapshot_problems(s.snapshot_dir)
    if problems:
        raise RuntimeError(f"snapshot still incomplete after the download: {', '.join(problems[:6])}")


def worker_complete(s: Settings, worker: Node) -> bool:
    """The worker's copy of the snapshot is complete: every shard the index names, at the head's sizes."""
    # dotfiles (.gitattributes) skipped on both sides: the worker's `*` does not match them
    want = sorted(f"{p.stat().st_size} {p.name}" for p in s.snapshot_dir.iterdir()
                  if p.is_file() and not p.name.startswith("."))
    out = worker.script(f"cd {s.snapshot_dir} 2>/dev/null && for f in *; do [ -f \"$f\" ] && echo \"$(stat -L -c %s \"$f\") $f\"; done",
                        check=False)
    return sorted(line for line in out.stdout.splitlines() if line.strip()) == want


def sync_snapshot(s: Settings, head: Node, worker: Node) -> None:
    """The worker serves the same snapshot, byte for byte: the repo's HF cache folder copied over the link once
    (about 176 GB), unless the worker downloaded it itself."""
    if worker_complete(s, worker):
        log(f"{worker.name}: snapshot {s.repo}@{s.revision[:8]} complete")
        return
    repo_dir = s.hf_cache / "hub" / f"models--{s.repo.replace('/', '--')}"
    log(f"Copying the snapshot to {worker.name} (about 176 GB over the link)")
    worker.run(["mkdir", "-p", str(repo_dir)])
    rc = head.stream(["rsync", "-a", "--partial", "--info=progress2", f"{repo_dir}/", f"{worker.ssh}:{repo_dir}/"])
    if rc or not worker_complete(s, worker):
        raise RuntimeError(f"rsync of the snapshot to {worker.name} failed ({rc})")


DFLASH2_NOTICE = ("NOTICE: drafter = dflash2 (the default) uses IncoAI's incoai/GLM-5.3-Flash-DFlash2 weights, licensed "
                  "CC BY-NC-ND 4.0: NON-COMMERCIAL USE ONLY. For commercial use set drafter = \"mtp\" (cluster.toml or "
                  "DRAFTER=mtp); commercial licences for DFlash2: contact@inco.ai")


def ensure_dflash2(s: Settings, *, download: bool = True) -> None:
    """drafter=dflash2: IncoAI's pinned drafter under hf_cache (its model.safetensors at the pinned sha256),
    downloaded from IncoAI's repository when it is not; nothing of it is changed on disk."""
    log(DFLASH2_NOTICE)
    weights = s.dflash2_dir / "model.safetensors"
    if not (weights.is_file() and (s.dflash2_dir / "config.json").is_file()):
        if not download:
            raise RuntimeError(f"drafter=dflash2 and {s.dflash2_dir} is incomplete")
        hf = shutil.which("hf") or shutil.which("huggingface-cli")
        if hf is None:
            raise RuntimeError("drafter=dflash2 needs the hf CLI to download it: pip install -U huggingface_hub")
        log(f"Downloading {s.dflash2_repo}@{s.dflash2_revision[:8]} (about 2.3 GB)")
        rc = subprocess.call([hf, "download", s.dflash2_repo, "--revision", s.dflash2_revision],
                             env={**os.environ, "HF_HOME": str(s.hf_cache), "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1")})
        if rc:
            raise RuntimeError(f"hf download of {s.dflash2_repo} failed ({rc})")
    h = hashlib.sha256()
    with weights.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != s.dflash2_sha256:
        raise RuntimeError(f"{weights} has sha256 {h.hexdigest()}, not the pinned {s.dflash2_sha256}")
    log(f"DFlash2 {s.dflash2_repo}@{s.dflash2_revision[:8]} verified ({s.dflash2_dir})")


def sync_dflash2(s: Settings, head: Node, worker: Node) -> None:
    """The worker loads its own copy of the drafter (rank 1 drafts too): the snapshot folder copied once."""
    want = (s.dflash2_dir / "model.safetensors").stat().st_size
    out = worker.script(f"stat -L -c %s {s.dflash2_dir}/model.safetensors 2>/dev/null", check=False)
    if out.stdout.strip() == str(want):
        return
    repo_dir = s.hf_cache / "hub" / f"models--{s.dflash2_repo.replace('/', '--')}"
    log(f"Copying the DFlash2 drafter to {worker.name} (about 2.3 GB)")
    worker.run(["mkdir", "-p", str(repo_dir)])
    rc = head.stream(["rsync", "-a", "--partial", f"{repo_dir}/", f"{worker.ssh}:{repo_dir}/"])
    if rc:
        raise RuntimeError(f"rsync of the DFlash2 drafter to {worker.name} failed ({rc})")
