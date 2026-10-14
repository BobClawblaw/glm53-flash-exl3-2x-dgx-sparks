# Working on this recipe

Serve Mia's AI Lab's EXL3 GLM-5.3-Flash as published on two DGX Sparks with TensorFold's native engine at TP=2, up to
16 requests at once: by default the published `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`; with `ablit = true`
(needs `HF_TOKEN`) the gated Ablit weights `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit`.
Stdlib Python only, no harness dependencies. Say which checkpoint every measurement used (the README's numbers so far
are the published checkpoint's).

- **Entry point:** `python3 -m spark up|down|status|logs|image|validate`.
  - Settings come from `recipe.toml` (defaults), then `cluster.toml` per cluster (git-ignored), then environment
    variables in upper case.
  - `python3 -m spark validate` is the dry run.
- **The patch** (`docker/patches/tensorfold-native-v1.0.11-glm.patch`, `git diff v1.0.11 <head>` of the port) is
  pinned by sha256 in `recipe.toml`. Never edit it in place. Change the port's branch, run `tf-cuda-test glm-batch`
  on both ranks (16 streams together equal to each alone, serial and drafted; resumed equal to fresh) and the ruler,
  then regenerate, check that `git apply --check` passes on a clean v1.0.11, re-pin, `spark image --rebuild`, and
  rerun the suite.
- **Drafts never change a reply.** Copy drafts and DFlash2 only propose, and the target verifies. Any change to the
  drafting paths must keep `glm-batch` passing.
- **DFlash2 is the default and non-commercial.** Its weights are CC BY-NC-ND 4.0: download them from IncoAI at the
  pinned revision, check the sha256, and never commit, convert on disk or redistribute them (packing in memory at
  load is fine). Keep the README's licence notice, NOTICE and the `spark up` notice saying so, and keep
  `drafter = "mtp"` working as the commercial-use setting.
- **Measure before you claim.** The suite (`python3 -m quality.suite`) and the ruler (`bench/ruler.py`) write JSON
  under the folder you name. Commit those under `evidence/<date>/` and point the README at the files. Report
  per-stream and wall throughput: a server that runs streams one after another looks fast in summed rates.
- **Memory is unified:** read it with `free -h`, not `nvidia-smi`. The engine sizes one pool from the smaller node
  and its watchdog exits a rank below 2 GiB available; a stalled GB10 does not recover on its own.
- **Restarts:** `down` stamps the stop, and `up` waits out `settle_seconds` before rank 1 starts. It retries, with a
  memory compaction, when the worker refuses NCCL's first memory registration (GB10: `ibv_reg_mr` ENOMEM after a
  large free).
- **Builds:** never build an image or load one onto a node while a rank runs there (unified memory). Stop the pair
  first.
