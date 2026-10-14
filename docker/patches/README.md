# docker/patches/

The image build (`docker/Dockerfile.native`) applies `tensorfold-native-v1.0.11-glm.patch` to TensorFold at the release
`recipe.toml` pins (`[engine] tensorfold_sha`, v1.0.11). The patch's sha256 is pinned too (`patch_sha256`). The
build refuses a file that differs, the image carries the hash as a label, and `spark up` refuses an image whose label
differs. A changed patch is a new pin and new evidence.

## What the patch adds

It is `git diff v1.0.11 <head>` of this project's port of GLM-5.3-Flash to TensorFold's native CUDA engine. It adds:

- `zig/src/families/glm_cuda/`: GLM-5.3-Flash on two CUDA ranks.
  - The loader reads the EXL3 checkpoint as published.
  - The forward covers KDA layers, DSA on the latent cache with the indexer and the sparse pools,
    hyper-connections, the EXL3 routed experts and the shared expert, and the MTP head.
  - A batched step serves up to 16 streams, each with its own caches out of one pool.
  - Kept prompt states, greedy and sampled draws, and the native family interface (rank 0's lane core, rank 1
    replaying its calls).
  - Copy drafts (`copy.zig`), the DFlash2 block drafter (`dflash.zig`, opt-in) and the memory guard (`guard.zig`).
- Its kernels (`zig/kernels/cuda/glm/`): every launch is ours. That covers the dense matmul at 4/6/8/16 bits, the
  EXL3 trellis experts, latent attention, the indexer, KDA, the hyper-connections, the vision tower's helpers and the
  DFlash2 drafter's kernels.
- The EXL3 experts' prompt kernels (`glm_exl3_prompt_gu`, `glm_exl3_prompt_dn` in `glm/exl3.cu`): the transposed
  design from Mia's AI Lab's TensorFold patches (Apache-2.0), written for this engine. The decoded trellis tiles are
  the tensor cores' A operand and an item's 64 prompt rows the B operand, gate and up run in one block with the
  SwiGLU as the epilogue, and the down projection carries its own epilogue; no partial sums touch memory, and every
  output is one accumulation chain, so a resumed prompt equals a fresh one bit for bit. 32K prompt tokens' experts:
  87 s with the grouped decode kernel, 7.2 s here.
- `zig/src/cuda/roce.zig` and `zig/kernels/cuda/roce_gather.cu`: small all-gathers as one-shot RDMA writes over both
  halves of the RoCE port (b12x's exchange, two ranks).
- `-Dglm` in the build and the CUDA registry entry, and `tf-cuda-test glm-*` checks: `glm-batch` (16 streams
  together equal to each alone, serial and drafted, resumed equal to fresh) and `glm-prof` (step times and a
  per-kernel profile).

## The engine's switches

Set through `extra_env`. The recipe sets the `*_BITS`, `TF_GLM_DRAFTER` and `TF_GLM_DFLASH2` itself.

| Variable | Default | Effect |
|---|---|---|
| `TF_GLM_DENSE_BITS` / `TF_GLM_HEAD_BITS` / `TF_GLM_DRAFT_HEAD_BITS` | recipe | dense linears / head / the drafts' head packed at load |
| `TF_GLM_DRAFTER` | (mtp) | `dflash2`: IncoAI's DFlash2 drafter from `TF_GLM_DFLASH2` (non-commercial weights) |
| `TF_GLM_DFLASH2_BITS` | 4 | the drafter's linears packed at load: 4, 6, 8, or 0 (bf16) |
| `TF_GLM_COPY` | 1 | `0`: no copy drafts |
| `TF_GLM_WINDOW` | 16 (recipe: 8) | a stream's verify window at most, 2-16 rows (8: JSON +17%, code and SQL +4%; repeating a passage verbatim 27% slower) |
| `TF_GLM_DRAFT_CONFIDENCE` | 0.7 | a draft chain stops below this probability (0.4: slower on every phase) |
| `TF_GLM_DRAFT_MARGINAL` | 0 | DFlash2: a chain stops at the first draft whose own chance is below this instead (0.4-0.6 measured slower: a verified row costs 6.9 ms) |
| `TF_GLM_DFLASH2_HEAD` | recipe: `draft` | DFlash2's candidates from the 4-bit draft head (`full`: the 8-bit verify head, 1.1 ms more a round at equal acceptance) |
| `TF_GLM_DFLASH2_SPLIT` | 1 | DFlash2's MLP linears split across the ranks (gate and up by columns, down by K, one fp32 gather a layer); `0`: whole on each rank |
| `TF_GLM_DFLASH2_ATTN` | chunked | DFlash2's attention as 16 key chunks a kv head and a merge; `serial`: one block a kv head (0.5 ms more a round) |
| `TF_GLM_DFLASH2_PROF` | off | `1`: each DFlash2 phase synchronized and timed; `2`: every kernel kind (rank 1 logs its own table) |
| `TF_GLM_MIX` | recipe: `0` | `1`: with DFlash2 loaded, each stream picks DFlash2 or the MTP head a round by measured tokens a ms (-3.7% on code); `0`: DFlash2 every round and the MTP head never runs |
| `TF_GLM_COPY_MIN` / `TF_GLM_COPY_MIN_REPLY` | 8 / 16 | tokens a copy match takes in the prompt / inside the reply |
| `TF_GLM_EXL3_PROMPT` | `fused` | the EXL3 experts' prompt kernel: `fused` (the transposed design), `staged` (128-pair items, rows and words through shared memory, K split `TF_GLM_EXL3_PROMPT_SK` ways), `wide` (its 8-warp block), `old` (the grouped decode kernel: 12x slower at 32K tokens) |
| `TF_GLM_EXL3_PROMPT_SK` | 2 | the staged kernel's K splits (1, 2, 4); one value for every chunk, so a resumed prompt equals a fresh one |
| `TF_GLM_PREFILL_ROWS` | 2048 | rows a prompt chunk takes |
| `TF_GLM_KEEP_HEADROOM` | 8192 | cache slots a kept state is allocated beyond its request when the pool has them, so a conversation's next turn (longer by the reply and the new message) resumes it; `0`: none, and only a prompt of the same size resumes |
| `TF_GLM_GRAPHS` | 96 | decode graphs kept instantiated |
| `TF_MEM_FLOOR_GIB` | 2 | a rank exits below this much available memory (no new requests below the floor + 3) |
| `TF_ROCE` | 1 | `0`: NCCL carries every all-gather |
| `TF_ROCE_MAX_KB` | 512 (recipe: 4096) | the largest all-gather the one-shot RoCE path carries; larger ones go to NCCL (at 8 streams a verify's fp32 partials are 1 MiB: 512 cost 24-43% there) |
| `TF_GLM_TRACE` | off | every 256 rounds: verify and draft times, drafts landed, memory, graphs |

## Changing the patch

1. Change the port's branch and run `tf-cuda-test glm-batch` on both ranks.
2. Run the ruler, then regenerate the patch: `git diff <v1.0.11 sha> <head> > tensorfold-native-v1.0.11-glm.patch`.
3. Check it applies to a clean v1.0.11 checkout (`git apply --check`).
4. Re-pin `patch_sha256`, run `spark image --rebuild`, and rerun the suite.
