"""The settings guards and the engine command line, with no Docker and no GPU: python3 -m unittest discover -s tests -q"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spark import config  # noqa: E402

CLEAN = {k: v for k, v in os.environ.items() if k not in {f.name.upper() for f in config.Settings.__dataclass_fields__.values()}}


def load(**env: str) -> config.Settings:
    return config.load(cluster_file="none", env={**CLEAN, **env})


class GuardTests(unittest.TestCase):
    def refused(self, needle: str, **env: str) -> None:
        with self.assertRaises(config.ConfigError) as got:
            load(**env)
        self.assertIn(needle, str(got.exception), env)

    def test_defaults_load_and_name_the_pinned_patch(self) -> None:
        s = load()
        self.assertEqual((s.context, s.drafts, s.thinking, s.vision), (1048576, True, False, True))
        self.assertEqual((s.dense_bits, s.head_bits, s.draft_head_bits, s.parallel), (4, 8, 4, 16))
        self.assertEqual(s.model_in_container, f"/cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit/snapshots/{s.revision}")
        self.assertEqual(hashlib.sha256(s.patch_path.read_bytes()).hexdigest(), s.patch_sha256)

    def test_ablit_is_the_default_and_the_published_checkpoint_the_switch(self) -> None:
        s = load()
        self.assertTrue(s.ablit)
        self.assertEqual((s.repo, s.revision[:8]), ("Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit", "57edefd2"))
        self.assertIn("(ablit)", s.summary())
        p = load(ABLIT="0")
        self.assertFalse(p.ablit)
        self.assertEqual((p.repo, p.revision), (s.published_repo, s.published_revision))
        self.assertEqual(p.revision[:8], "76c0b517")
        self.assertNotIn("(ablit)", p.summary())
        self.assertNotEqual(s.snapshot_dir, p.snapshot_dir)   # both checkpoints keep their own cache folder
        self.refused("published_revision", PUBLISHED_REVISION="main")

    def test_packed_bits_and_vision(self) -> None:
        env = load(DENSE_BITS="4", HEAD_BITS="0").container_env()
        self.assertEqual((env["TF_GLM_DENSE_BITS"], "TF_GLM_HEAD_BITS" in env, env["TF_GLM_DRAFT_HEAD_BITS"]), ("4", False, "4"))
        self.refused("must be 0 (bf16), 4, 6 or 8", DENSE_BITS="5")
        self.assertIn("--vision", load().native_args(0))
        self.assertNotIn("--vision", load(VISION="0").native_args(0))
        self.refused("extra_args sets --vision", EXTRA_ARGS="--vision")

    def test_windows(self) -> None:
        self.assertEqual(load(CONTEXT="1048576").context, 1048576)
        self.assertEqual(load(CONTEXT="65536").context, 65536)
        self.refused("trained window", CONTEXT="1048832")
        self.refused("trained window", CONTEXT="2048")
        self.refused("multiple of 256", CONTEXT="65537")

    def test_integers_and_flags(self) -> None:
        self.refused("not a decimal integer", MAX_TOKENS="0100")
        self.refused("not a decimal integer", PORT="x")
        self.refused("outside", PORT="70000")
        self.refused("0/1 or true/false", DRAFTS="yes")
        self.refused("40-hex", REVISION="main")
        self.refused("0/1 or true/false", MEMGUARD="maybe")

    def test_extra_args_cannot_reset_owned_flags(self) -> None:
        for word in ("--tp 1", "--context=8192", "--paral 8", "--no-dr", "--port 1", "--tool-sys x"):
            self.refused("which the recipe passes itself", EXTRA_ARGS=word)
        self.assertEqual(load(EXTRA_ARGS="--temperature 0.6").extra_args, "--temperature 0.6")
        self.refused("not KEY=VALUE", EXTRA_ENV="NCCL_PROTO")

    def test_parallel(self) -> None:
        a = load(PARALLEL="4").native_args(0)
        self.assertEqual(a[a.index("--parallel") + 1], "4")
        self.refused("within [1, 16]", PARALLEL="17")
        self.refused("within [1, 16]", PARALLEL="0")

    def test_patch_pin(self) -> None:
        self.refused("not a sha256", PATCH_SHA256="abc")
        self.refused("not the pinned", PATCH_SHA256="b" * 64)
        self.refused("is missing", PATCH="nope.patch")


class ServeArgsTests(unittest.TestCase):
    def test_both_ranks_share_the_engine_settings(self) -> None:
        s = load()
        r0, r1 = s.native_args(0), s.native_args(1)
        for flag in ("--context", "--tp", "--master", "--master-port"):
            self.assertEqual(r0[r0.index(flag) + 1], r1[r1.index(flag) + 1], flag)
        self.assertEqual(r1[r1.index("--rank") + 1], "1")
        for flag in ("--name", "--host", "--port", "--max-tokens", "--no-thinking", "--tool-system", "--parallel"):
            self.assertIn(flag, r0)
            self.assertNotIn(flag, r1)                   # rank 1 serves no HTTP
        self.assertEqual(r0[r0.index("--parallel") + 1], "16")  # sixteen requests at once

    def test_switches(self) -> None:
        self.assertIn("--no-drafts", load(DRAFTS="0").native_args(1))
        self.assertIn("--no-drafts", load(DRAFTS="0").native_args(0))
        self.assertNotIn("--no-drafts", load().native_args(0))
        self.assertIn("--thinking", load(THINKING="1").native_args(0))
        self.assertNotIn("--tool-system", load(TOOL_SYSTEM="").native_args(0))
        self.assertEqual(load(EXTRA_ARGS="--temperature 0").native_args(0)[-2:], ["--temperature", "0"])

    def test_model_folder(self) -> None:
        self.assertTrue(load().model_in_container.startswith("/cache/huggingface/hub/models--Mia-AiLab--"))   # the snapshot as published

    def test_drafter(self) -> None:
        m = load(DRAFTER="mtp")
        self.assertNotIn("TF_GLM_DRAFTER", m.container_env())
        self.assertNotIn("TF_GLM_DRAFTER", load(DRAFTS="0").container_env())   # no drafts: no drafter, no download
        d = load()
        self.assertEqual(d.drafter, "dflash2")   # the default: non-commercial weights (README, NOTICE)
        env = d.container_env()
        self.assertEqual(env["TF_GLM_DRAFTER"], "dflash2")
        self.assertTrue(env["TF_GLM_DFLASH2"].endswith("/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/" + d.dflash2_revision))
        for needle, env in (("mtp or dflash2", {"DRAFTER": "eagle"}), ("40-hex", {"DFLASH2_REVISION": "main"})):
            with self.assertRaises(config.ConfigError) as got:
                load(**env)
            self.assertIn(needle, str(got.exception))

    def test_container_env(self) -> None:
        env = load(EXTRA_ENV="NCCL_PROTO=LL", MEMORY_RESERVE_GIB="12").container_env()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["NCCL_PROTO"], "LL")
        self.assertEqual(env["TENSORFOLD_MEMORY_RESERVE_GIB"], "12.0")
        self.assertEqual(env["NCCL_IB_HCA"], load().hca)


if __name__ == "__main__":
    unittest.main()
