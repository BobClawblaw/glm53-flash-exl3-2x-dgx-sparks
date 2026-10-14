# Changelog

## 2026-10-14

- **Prompts 3.6x faster: the EXL3 experts' prompt kernel in the transposed design** (`TF_GLM_EXL3_PROMPT=fused`, the
  default; after Mia's AI Lab's TensorFold patches, Apache-2.0). The old path ran prompt chunks through the grouped
  decode kernel: 32K prompt tokens spent 87 s of a 32K profile in it (74.5% of the prompt's time). The new kernels
  (`glm_exl3_prompt_gu`, `glm_exl3_prompt_dn`) decode each trellis tile once into the tensor cores' A operand and
  stream the item's 64 rows past it from shared memory, fuse gate and up with the SwiGLU epilogue and the down
  projection with its own, and keep every output in one accumulation chain: 7.2 s for the same 32K tokens
  (`evidence/dev/sweep-2026-10-14/prof-dev6-fused-n32768.rank0.log`), and a resumed prompt equals a fresh one bit
  for bit. Replies change only by rounding order: teacher-forced on the old kernel's own greedy replies the top token
  agrees at 107/114 and 130/135 positions (the old kernel against itself 114/114, 135/135), and the suite's reasoning
  and instruction checks move both ways within noise (GSM8K 243 -> 244, MMLU 294 -> 292, HumanEval 157 -> 156,
  IFEval strict 109 -> 111, MGSM 217 -> 223; `evidence/2026-10-14/suite-ab/`). `tf-cuda-test glm-batch` passes on
  both ranks.
  Served: cold prefill 991 / 1,001 / 957 tok/s on 6K / 30K / 114K-token prompts (272 / 270 / 268 before; Mia-AiLab's
  recipe 627-697; tonyd2wild's NVFP4 recipe 1,121-1,225), decode unchanged within noise (`evidence/2026-10-14/`).
- **A conversation's next turn resumes its kept state** (`TF_GLM_KEEP_HEADROOM`, 8192): a kept state is allocated
  8,192 cache slots beyond its request, kept states giving way when the pool is full of them (the first form of the
  change counted them as occupied, so a pool full of kept states denied every headroom and nothing changed). The next
  turn (longer by the reply and the new message) resumes it instead of recomputing the whole prompt: turn 2 of a
  67K-token chat starts in 0.20 s instead of 67.7 s, turn 3 in 0.16 s; a 4.2K-token chat's turns 2 and 3 resume 4,196
  and 4,214 tokens (`evidence/2026-10-14/chat-resume-shipped-*.txt`). Before, turn 2 of a 15K-token chat took 55 s to
  its first token.
- **The dev harness** ran NCCL on sockets from 06:04 to 10:20 UTC (the container's SPCX plugin and libibverbs on the
  bare library path left NCCL without an InfiniBand provider): the 32K profiles in `evidence/dev/sweep-2026-10-14/`
  from that window carry a 27.7 s all-gather instead of 3-4 s; their kernel rows are unaffected.
- **The Ablit weights are the default** (`ablit = true`, `[model]` in `recipe.toml`):
  [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit)
  at `57edefd2`, the revision Mia's AI Lab's recipe pins for its `ABLIT=1` mode. The repository is gated: `up` needs
  `HF_TOKEN` (or `hf auth login`) and checks, with one `HEAD` request before any download, that the token reaches the
  pinned files (401 / 403 explained with the steps); the token stays in the download's environment. `ablit = false`
  (`cluster.toml`, or `ABLIT=0`) serves the published checkpoint (`published_repo` / `published_revision`, unchanged
  pins); the two checkpoints keep separate cache folders. Thinking stays off by default. The README's numbers are the
  published checkpoint's until the Ablit evidence lands (README, "Ablit weights").
- **Quality.** The README's quality table carries the whole suite on the shipped image (`evidence/2026-10-13/suite/`,
  `suite-soak/`, `suite-docvision/`): ChartQA 147/200, DocVQA 112/200 at ANLS 0.5, long-tasks 36/40, concurrent
  12/12. The soak's replies are 60/60 equal and its resumes 0/60: a chat's next turn does not resume its kept state on
  this build (the state is sized for the turn that made it); the fix is in the next engine patch.
- **Quality harness.** `quality/soak.py`: the `background` flag shadowed the helper that sends the side decode, so
  every round's background request failed and the check scored itself 0 (the replies were 60/60 equal); the helper is
  `story()` now. `quality/docvision.py`: a picture the server refuses (HTTP 400, e.g. over its 16-megapixel decode
  limit) counts as a miss and is reported, instead of aborting the whole check.

## 2026-10-13

- **Speed.** One stream, greedy, 16 prompts a phase: prose 42.3 -> 45.8, code 66.1 -> 70.3, structured 53.6 -> 57.2,
  list 52.3 -> 56.2 tok/s; replies and drafts landed unchanged (`evidence/dev/sweep-2026-10-13/`).
  - The DFlash2 round, 8.5 -> 4.3 ms: the MTP head no longer runs while DFlash2 drafts alone; the drafter's MLP
    linears are split across the two ranks (gate and up by columns, down by K, one fp32 gather a layer; 0.56 -> 0.37
    GiB a rank); its attention runs 16 key chunks a kv head with a merge instead of one serial block (0.67 -> 0.13 ms
    a round); its candidates come from the 4-bit draft head (`TF_GLM_DFLASH2_HEAD=draft`, now the recipe's setting:
    +3% at equal acceptance); the drafter mix is off (`TF_GLM_MIX=0`: the mix gave the MTP head 188 of 260 rounds
    and cost 3.7% on code).
  - Sampled requests: the 64 candidates a row by radix selection (3.6 -> 0.35 ms a verify, the same bytes), full-row
    draws by a counting sort. The candidate buffers were 64 ints a row short (132 needed) and overflowed past 62
    sampled rows a step.
  - Measured and left out: a lower chain cut (0.4: slower on every phase), noise-aware draft picks (no gain), a
    per-draft chance floor (`TF_GLM_DRAFT_MARGINAL` 0.4-0.6: slower, each verified row costs 6.9 ms), reply copies at
    8 tokens (none fire on the ruler).
- **Against other recipes** (`evidence/dev/sweep-2026-10-13/`). Mia-AiLab's Python recipe, one stream: prose +1.6%,
  code +1.7%, structured -6.5%, list -3.6%. tonyd2wild's vLLM NVFP4 recipe on its own harness at 262K: decode ahead on
  every prompt but JSON (-11%), prose +23%, tool calls +57% end to end, 3 to 6 streams +31% to +74%, a 114K-token
  prompt's decode +55%; cold prefill 270 tok/s against its 1,120-1,225 and the Python recipe's 630-700. A 32K-token
  prompt's profile (`evidence/dev/sweep-2026-10-13/prompt-profile-32k.txt`) puts 74.5% of its time in the EXL3
  routed experts' prompt kernel: the next target. 1M and 262K context settings measure the same within 2%.
- `tf-cuda-test glm-prof` profiles a prompt of `TF_PROF_PROMPT` tokens (its wall time, then its kernels).
- **Eight streams: +9% to +43%** (`TF_ROCE_MAX_KB=4096` in `extra_env`). The one-shot RoCE path carried gathers up to
  512 KiB; a verify step's fp32 partial sums are 1 MiB at 8 streams, so every one of them went to NCCL. Wall tok/s
  over the wave: prose 94 -> 102, code 109 -> 136, structured 78 -> 111, list 97 -> 120 (two runs, shipped image).
- `bench/compare.py`: two measurement files side by side with the difference in percent.
- Credits name every contributor the work builds on.

## 2026-10-12

- **First version.** Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold on two DGX Sparks with TensorFold v1.0.7's native
  engine and this recipe's patch.
  - **The model:** a GLM-5.3-Flash family for two CUDA ranks that reads the EXL3 checkpoint as published, with every
    kernel our own.
  - **Serving:** up to 16 requests in one batched step, the 1M window, images and video, and kept prompt states.
  - **Packing:** dense linears packed at load to 4 bits, the head to 8.
- **Speed.**
  - Small all-gathers go as one-shot RDMA writes over both halves of the RoCE port (b12x's exchange; `TF_ROCE=0`
    keeps NCCL).
  - The MoE router runs a warp per K quarter, and the dense matmuls are dependent launches (their weights stream in
    under the kernel before).
  - One stream decodes at 46-56 tok/s on the ruler; 8 streams at 98-121 tok/s wall.
- **Drafts.**
  - **Copy drafts** (prompt lookup) on top of the MTP head, verified in windows of up to 16 rows. A repeated or
    edited passage runs at 15 tokens a round.
  - **IncoAI's DFlash2 block drafter, the default** (`drafter = "dflash2"`). Its weights are **non-commercial
    (CC BY-NC-ND 4.0)**, downloaded from IncoAI and checked by sha256; `drafter = "mtp"` is the commercial-use
    setting. Our implementation after z-lab's reference (matching it on the same inputs): one 5-layer pass drafts 7
    tokens, the linears are packed to 4 bits at load, and a round takes about 5 ms. At one stream it is 21% faster
    than MTP on code and level elsewhere.
- **Memory safety.**
  - The two ranks size one stream pool from the smaller node.
  - The decode graph cache is bounded (96, recycled by in-place updates).
  - A watchdog on each rank refuses new requests below 5 GiB available and exits below 2 GiB, or when its peer is
    gone, instead of letting a GB10 stall.
- **The quality suite:**
  - checks for MATH-500, GPQA-Diamond, RULER-style long-context tasks, ChartQA and DocVQA;
  - `bench/ruler.py` reports wall throughput beside the summed per-stream rates.
