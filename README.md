# GLM-5.3-Flash EXL3 on two DGX Sparks (TensorFold native)

Serve Mia's AI Lab's EXL3 quantization of GLM-5.3-Flash (4-bit "mcg" trellis routed experts, every other weight
BF16; a quantization of zai-org's GLM-5.3-Flash, MIT) across two NVIDIA DGX Spark (GB10, 128 GB) nodes with
[TensorFold](https://github.com/ashhart/TensorFold)'s native (Zig) engine at tensor-parallel 2. By default the recipe
serves the published [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)
(Apache-2.0), no account needed; `ablit = true` serves the **Ablit weights**,
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit)
(the same checkpoint, abliterated; a gated repository, see [Ablit weights](#ablit-weights)). Either way:

- up to 16 requests decoding together, each with the model's full 1,048,576-token window;
- image and video inputs;
- IncoAI's DFlash2 block drafter by default (**non-commercial weights**, see below), or the checkpoint's own MTP head,
  plus copy drafts, verified in windows of up to 8 rows;
- kept prompt states for multi-turn chats and agents, tool calls, and an OpenAI- and Anthropic-compatible API;
- a memory guard that stops a rank cleanly before the unified memory runs out;
- a systemd unit.

> **Licence: the default configuration is for non-commercial use only.** By default the recipe drafts with IncoAI's
> DFlash2 weights ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)), which are
> licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/): non-commercial use (research and
> evaluation) only, no derivatives distributed. `spark up` downloads them from IncoAI, prints this notice and never
> changes or redistributes them. **For any commercial use, set `drafter = "mtp"`** (in `cluster.toml`, or
> `DRAFTER=mtp`): the recipe then uses only the checkpoint's own MTP head and nothing of IncoAI's is downloaded.
> Commercial licences for DFlash2 are IncoAI's to grant (contact@inco.ai). The recipe's own files are MIT and the
> engine patch is Apache-2.0, whichever drafter you pick ([NOTICE](NOTICE)). This is not legal advice.

TensorFold v1.0.11 serves GLM-5.3-Flash natively on Macs only. This recipe adds a GLM-5.3-Flash family for two CUDA
ranks to the native engine (the patch, [`docker/patches/README.md`](docker/patches/README.md)) and serves Mia-AiLab's
checkpoint **as published**: the engine reads the EXL3 trellises and scales where the checkpoint keeps them. At load
it packs the dense linears to 4 bits and the head to 8 (symmetric, a scale per 64 inputs; `dense_bits`, `head_bits`,
`draft_head_bits`). The approach and the target numbers follow Mia-AiLab's own recipe for this checkpoint on
TensorFold's Python engine ([MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold));
this port runs on TensorFold's current release instead, with its own kernels.

## Quick start

```bash
cp cluster.example.toml cluster.toml        # the pair's addresses, the RoCE devices, the port
python3 -m spark validate                   # the settings, checked; touches nothing
python3 -m spark up                         # image, checkpoint, both ranks, ready
python3 -m spark status
python3 -m spark down
```

`up` builds the image on the head (TensorFold v1.0.11 with the patch, `docker/Dockerfile.native`), downloads the
pinned snapshot (about 176 GB), copies the image and the snapshot to the worker, and starts rank 1 there and rank 0 on
the head. Requirements: two GB10 nodes linked by ConnectX-7 (RoCE), Docker with the NVIDIA runtime on both,
passwordless ssh from the head to the worker, about 210 GB free on each node and the `hf` CLI on the head. No
account or token: the published checkpoint is public. Only the opt-in [Ablit weights](#ablit-weights) need a Hugging
Face token whose account has access to their gated repository.

The API is OpenAI-style (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and Anthropic-style
(`/v1/messages`) at `http://<head>:<port>`; the model is named `GLM-5.3-Flash-EXL3`.

### Both halves of the port

A GB10's QSFP port shows up as two PCIe halves (`rocep1s0f1` and `roceP2p1s0f1`), each with its own address. Name
both in `hca` and the engine stripes its small all-gathers over the two (b12x's one-shot RDMA exchange); NCCL takes
the rest.

## Ablit weights

`ablit = true` (in `cluster.toml`, or `ABLIT=1`; the default is `false`) serves
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit):
Mia's AI Lab's abliteration of the same EXL3 checkpoint (the refusal direction removed from the weights; the method,
its verification and the responsible-use terms are on the model's page), at the revision pinned in `recipe.toml`
(`57edefd2`), the one Mia's AI Lab's own recipe pins for its `ABLIT=1` mode. Everything else, drafts, the window, the
kernels, the API, is the same. Thinking is off by default either way (`thinking = false`); a request can still ask to
think (`reasoning_effort`, or `chat_template_kwargs.enable_thinking`). The repository is gated:

1. Open [its page on Hugging Face](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit), log in
   and request access; Mia's AI Lab grants it per account.
2. With the same account, create a token with read access at <https://huggingface.co/settings/tokens>.
3. Give it to the head node: `HF_TOKEN=hf_...` in the environment of `python3 -m spark up`, or `hf auth login` once.
   The token goes into the download's environment only, never on a command line or into a log.

`up` sends one `HEAD` request for the pinned `config.json` before anything else: `401` means Hugging Face rejects
the token, `403` that the account has no access yet (or a fine-grained token cannot read gated repositories); it
stops with those steps. The first `up` then downloads the Ablit checkpoint (about 176 GB, beside the published one
in the cache) and copies it to the worker. Without `ablit = true` the published checkpoint is served and nothing is
gated; the two switch without a download once both are in the cache.

**The speed tables in this README are the published checkpoint's** (the default), so they compare with
Mia-AiLab's and tonyd2wild's numbers on the same weights. The Ablit weights, measured the same way on the same image (`evidence/2026-10-14/ablit/`):

| Ablit weights | prose | code | structured | list |
|---|---|---|---|---|
| one stream, greedy, tok/s (wall) | 46.1 (44.7) | 65.7 (63.3) | 57.8 (52.2) | 56.2 (56.1) |
| one stream, sampled | 42.5 | 62.9 | 53.0 | 53.3 |
| 8 streams, wall | 99.6 | 130.1 | 110.9 | 120.8 |
| 16 streams, wall | 127.9 | 168.9 | 142.6 | 158.0 |

Cold prefill 991 / 1,001 / 962 tok/s on the 6K / 30K / 114K-token prompts (the same as the published checkpoint's),
decode after the 114K prompt 37.9 tok/s (41.6), a 67K-token chat's turn 2 in 0.14 s. Prose, structured and list decode
the same as the published checkpoint; code about 5% slower, its replies shorter (247 tokens against 256) at the same
draft rounds a reply. The greedy row is `ruler-ablit-c1-clean.json`, a clean boot; the first boot's ruler
(`ruler-ablit-c1.json`, taken while a compile ran) read 64.4 / 50.9 on code and structured, and the sampled, 8- and
16-stream rows come from that boot. The same recipe on another pair (xspark03 and xspark04, the Ablit weights)
measures 46.7 / 66.3 / 58.8 / 57.1 from this head (`pair-252/ruler-c1.json`).

The quality checks on the Ablit weights (`ablit/suite/`), against the published checkpoint on the same image:
GSM8K 246/250 (244), MMLU 292/342 (292), HumanEval 159/164 (156), IFEval 118/150 loose and 109 strict (118 / 111),
tool calls 59/60 (59), JSON 30/30 (30), MGSM 221/240 (223). The abliteration costs nothing on these.

## What it serves

- **16 requests at once** (`parallel`): their windows verify together in one batched step. Each request's caches
  are sized to its prompt and `max_tokens` out of one pool, the memory the model leaves on the smaller of the two
  nodes, and a request waits while the pool is full.
- **A 1,048,576-token window** (`context`). The caches take about 13 KB a token on each node, so one full-window
  request fits beside the model.
- **Images and videos** in chat messages (`vision = true`), as OpenAI `image_url` / `video_url` parts (data URLs, or
  public HTTPS with `--vision-urls`). They're prepared as GLM's processor does (videos sampled at 2 frames a second)
  and encoded by the vision tower on the head node.
- **Drafts** (`drafts`; replies equal undrafted ones): the checkpoint's MTP head plus copy drafts, or DFlash2
  (below).
- **Kept prompt states**: a turn that extends an earlier prompt resumes from its kept state instead of reading the
  whole history again, and its reply equals a fresh prompt's.
- **Greedy and sampled replies**, and **thinking** (`thinking = false`, the default, closes the template's thinking
  block for a direct answer).
- **Not yet**: structured output (`response_format`); a prompt with images keeps no prompt state.

## Drafters

`drafter` picks who drafts; replies are the same either way (the target verifies every draft).

- **`dflash2`** (the default; **non-commercial use only**, see the licence notice above):
  [IncoAI's DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), a 5-layer block drafter that
  drafts 7 tokens in one pass from the target's hidden states.
  - The recipe downloads it from IncoAI at a pinned revision and checks its sha256.
  - The engine packs its linears to 4 bits in memory at load (`TF_GLM_DFLASH2_BITS`); nothing is written back.
  - The drafter's code in the patch is our own, written from z-lab's DFlash reference (MIT). It matches that
    reference on the same inputs: hidden rows at 0.9997+ cosine, the same drafts.
  - It is fastest on code (67.5 against 55.8 tok/s at one stream).
- **`mtp`** (for commercial use): the checkpoint's own MTP head, under the recipe's own licences only. About 18% slower
  on code, as fast or faster on prose and lists.

Both run with **copy drafts**. When the last 8 tokens of a stream's history occurred earlier in its prompt (16
inside the reply), the tokens that followed are drafted. **A verify window holds 8 rows** (`TF_GLM_WINDOW=8` in the
recipe's `extra_env`): the drafter proposes more than lands on JSON and code, and each extra row costs a verify step
its bytes, so 8 rows decode JSON 17% faster than 16 and code and SQL 4% (`evidence/dev/sweep-2026-10-14/depth/`).
The price is verbatim repetition: repeating a 640-token passage runs at 95 tok/s against 131 with 16-row windows,
which suit an editing workload better (`TF_GLM_WINDOW=16` in `cluster.toml`'s `extra_env`, keeping
`TF_ROCE_MAX_KB=4096` beside it).

## Measured on this pair

Two DGX Sparks (GB10, 128 GB each) over one ConnectX-7 RoCE link, TensorFold v1.0.11 native with this recipe's patch,
the defaults (1,048,576-token window, 4-bit dense / 8-bit head, EXL3 experts, DFlash2 with its 4-bit draft head,
copy drafts, 8-row verify windows, one-shot RoCE gathers up to 4 MiB, vision on), on the published checkpoint
(the default; [Ablit weights](#ablit-weights) for the abliterated form). The ruler is `bench/ruler.py`:
greedy, 256-token replies, thinking off, a distinct prompt per stream. "Wall" is every token of a wave over its wall
time. The shipped image, `evidence/2026-10-14/tf-1.0.11/ruler-shipped-addfa527-c{1,8,16}.json`; the previous
image's numbers (TensorFold v1.0.7, the old prompt kernel, 16-row windows, 2026-10-13) are `evidence/2026-10-13/`.

One stream, every prompt of a phase in turn (`--concurrency 1 --waves 16`), tok/s per stream (wall):

| | prose | code | structured | list |
|---|---|---|---|---|
| this release | 45.6 (45.0) | 70.1 (66.1) | 57.0 (53.0) | 56.8 (55.7) |
| 2026-10-13 | 45.9 (44.6) | 69.8 (65.1) | 56.7 (52.0) | 56.1 (54.8) |
| | -0.7% | +0.3% | +0.6% | +1.2% |

Streams together (`--concurrency 8 --waves 2 --runs 2`, `--concurrency 16 --runs 2`), tok/s over the wave's wall time:

| | prose | code | structured | list |
|---|---|---|---|---|
| 8 streams, this release | 100.9 | 133.4 | 113.0 | 119.3 |
| 8 streams, 2026-10-13 | 101.8 | 134.5 | 112.5 | 120.6 |
| | -0.9% | -0.9% | +0.5% | -1.1% |
| 16 streams, this release | 128.7 | 168.1 | 143.2 | 154.6 |
| 16 streams, 2026-10-13 | 129.8 | 173.7 | 140.5 | 156.0 |
| | -0.8% | -3.2% | +1.9% | -0.9% |

`drafter = "mtp"` (commercial use) was 46.1 / 55.8 / 54.0 / 55.7 per stream on 2026-10-12 and is unchanged by this
release's drafter work.

| | |
|---|---|
| A verify step, one stream | 22 ms + 6.9 ms a row (4.2 rows a round on the ruler) |
| A DFlash2 draft round, one stream | 4.3 ms (8.5 on 2026-10-12): the MTP head not run, the MLP split across the ranks, chunked attention, the 4-bit draft head |
| First token (short prompt) | 0.19-0.29 s |
| Cold prefill | about 1,000 tok/s at 6K-114K tokens (see below) |
| Memory a node | model 81.6 GiB (the head's includes the vision tower), stream pool ~13.8 GiB, reserve 12 GiB |

**Where a step's time goes** (one row): the routed experts' trellises, 8 of 288 an MoE layer over 42 layers (about
10 ms); the dense linears (about 9 ms); the hyper-connections, the attention and the small kernels; and two
all-gathers a layer.

**Each extra row in a verify window** reads the experts its token routes to: 4 rows take 30 ms of expert reads and
8 rows 49 ms, near the memory's bandwidth. So a window's rows cost their bytes, and drafts pay only when they land.

## Against other recipes

Numbers from `evidence/2026-10-14/tf-1.0.11/` (Mia-AiLab's rulers from `evidence/2026-10-13/`); `python3 bench/compare.py` renders each table from its two files. The
percentage is this recipe against the other.

**Mia-AiLab's recipe for the same checkpoint** (TensorFold's Python engine v0.6.0 with their patches, on another
pair), the same ruler, one stream:

| | prose | code | structured | list |
|---|---|---|---|---|
| greedy, this recipe / theirs | 45.6 / 45.1 (**+1.1%**) | 70.1 / 69.1 (**+1.3%**) | 57.0 / 61.2 (-6.8%) | 56.8 / 58.3 (-2.5%) |
| sampled (temperature 1, top_p 0.95), this recipe / theirs | 45.1 / 45.7 (-1.3%) | 66.1 / 57.4 (**+15.0%**) | 55.8 / 57.2 (-2.3%) | 55.4 / 55.5 (-0.2%) |

Eight streams: that server runs them one after another (41.8 / 62.2 / 52.4 / 60.4 wall); this one runs them
together (see above).

**tonyd2wild's vLLM NVFP4 recipe** ([GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark),
knapcio's stack, 262K context) on **its own harness and prompts** (`bench_tp2_night.py`: greedy, `reasoning_effort:
low`, median of 3), their published lane A against this recipe's branch head at `--context 262144`
(`tonyd-harness/`). Decode tok/s, one stream:

| prompt | this recipe | theirs | |
|---|---|---|---|
| count to 100 / 300 | 98.2 / 95.9 | 96.4 / 93.0 | +1.8% / +3.1% |
| code | 75.6 | 68.9 | **+9.7%** |
| json | 74.0 | 71.0 | **+4.2%** |
| sql | 65.0 | 60.3 | +7.7% |
| tool call (end to end) | 70.9 | 40.4 | +75.5% |
| math | 66.9 | 65.4 | +2.2% |
| prose / narrative / summary | 42.2 / 40.7 / 45.7 | 35.3 / 35.3 / 41.8 | +19.4% / +15.3% / +9.4% |

Mixed prompts together, aggregate tok/s (median of 3 rounds):

| streams | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| this recipe | 71.0 | 61.3 | 69.4 | 96.6 | 81.8 | 86.8 |
| theirs | 63.3 | 57.2 | 46.4 | 71.5 | 46.3 | 65.7 |
| | **+12.2%** | +7.1% | **+49.6%** | **+35.2%** | **+76.8%** | **+32.2%** |

Long prompts:

| | this recipe | theirs | |
|---|---|---|---|
| cold prefill, 6K / 30K / 114K-token prompt | 991 / 1,002 / 950 tok/s (6.0 / 29.8 / 119.9 s) | 1,121 / 1,200 / 1,225 (5.3 / 24.9 / 93.0 s) | −11.6% / −16.5% / −22.5% |
| decode after a 114K-token prompt, 1 / 2 at once | 42.0 / 15.3 | 26.9 / 14.0 | +56.0% / +9.2% |

At 64K, 128K and 256K tokens (`bench/longctx.py`: a document sized to the token count with two facts buried at a
third and two thirds, `evidence/2026-10-14/tf-1.0.11/longctx/`):

| prompt | prefill | time to first token | both facts in the reply |
|---|---|---|---|
| 65,526 tokens | 1,000 tok/s | 65.5 s | yes |
| 131,073 tokens | 959 tok/s | 136.6 s | yes |
| 262,144 tokens | 902 tok/s | 290.6 s | yes |
| 2 x 65K tokens at once | 998 tok/s each | 65.7 s, both | yes, both |

The previous image prefilled at 272 / 270 / 268 tok/s (`evidence/2026-10-13/tonyd-harness/`): 3.6x slower. Mia-AiLab's
recipe prefills at 627-697 tok/s on the same prompts (30K and 114K), so this recipe is 43-60% faster than theirs
there. What changed is the EXL3 routed experts' prompt kernel (`evidence/2026-10-14/prompt-profile-32k.txt` against
`prompt-profile-32k-old-kernel.txt`): a 32K-token prompt's experts took 87.2 s in the grouped decode kernel and take
7.2 s in the fused prompt kernels, so the dense 4-bit matmul (5.5 s) is now the largest kernel of a prompt, with the
attention, the router and the hyper-connections at 1-2.5 s each. The 1M and 262K context settings measure the same
within 2% on every row (`evidence/2026-10-13/tonyd-harness/ours-ctx1048576.json`).

## Quality

`python3 -m quality.suite` (greedy, thinking off) on the published checkpoint (the default),
`evidence/2026-10-13/suite/` (`report.md`, the previous image); the soak and the document checks reran after harness
fixes in `suite-soak/` and `suite-docvision/`. The fused prompt kernel's image reran the reasoning and instruction
checks (`evidence/2026-10-14/suite-ab/`): the second number where there is one. The kernel changes a prompt's rounding
order (its top token differs from the old kernel's at 4-6% of a greedy reply's positions) and the scores move both
ways within noise.

| Check | Result |
|---|---|
| drafts / streaming / multiturn / stop / thinking / copy / max-tokens | 12/12, 4/4, 8/8, 3/3, 10/10, 2/2, 3/3 |
| tool calls (60) | 59/60 (98.3%); fused 59/60 |
| JSON (30, `response_format`) | 30/30; fused 30/30 |
| IFEval (150) | 118/150 loose (78.7%), 109/150 strict; fused 118 loose, 111 strict |
| GSM8K (250) | 243/250 (97.2%); fused 244/250 (97.6%) |
| MGSM (8 languages x 30) | 217/240 (90.4%); fused 223/240 (92.9%) |
| MMLU (57 subjects x 6) | 294/342 (86.0%); fused 292/342 (85.4%) |
| MATH-500 | 442/500 (88.4%) |
| GPQA-Diamond | 140/198 (70.7%) |
| HumanEval (164, sandboxed) | 157/164 (95.7%) pass@1; fused 156/164 (95.1%) |
| repetition | 8/8: repeated 4-grams 0.25% mean, 0.82% max |
| long-context | 42/42: needles 12/12 to 102k, spread facts 10/10 at 25k, 51k, 102k |
| long-tasks (RULER-style) | 36/40 (90.0%): var-track 3/5 at 25k and 102k, the rest 10/10 |
| soak | 60/60 replies equal to the same request alone, 0 errors; 0/60 turns resumed a kept state, so the check scores 0/60 (see below) |
| snapshot | 12/12 (`replies.json`) |
| vision (6) / vision-concurrent (16) | 6/6, 16/16 |
| ChartQA (200) | 147/200 (73.5%) relaxed accuracy: human-written 79/105, generated 68/95 |
| DocVQA (200) | 112/200 at ANLS >= 0.5, mean ANLS 0.557; 5 pictures refused (over the decode limit) |
| concurrent | 12/12: 16 streams together equal each alone |

The soak's resumes: a chat's second turn never resumed its kept state on the previous image (`cached_tokens` 0, the
prompt recomputed: 55 s to the first token of a 15K-token chat's turn 2 with the old prompt kernel, 15 s with the
fused one; `evidence/dev/sweep-2026-10-14/chat-resume-shipped.txt`), because a kept state was sized for the turn that
made it and the engine skips a kept state smaller than the request. This image allocates a kept state 8,192 slots
beyond its request (`TF_GLM_KEEP_HEADROOM`; kept states give way when the pool is full of them): turn 2 of a 67K-token
chat starts in 0.20 s instead of 67.7 s and turn 3 in 0.16 s (`evidence/2026-10-14/chat-resume-shipped-*.txt`).

## Memory

The GB10's memory is the host's. When it runs out the machine stalls (no OOM kill comes), and the peer rank then
waits in a collective. The engine guards against both:

- **One pool for both ranks.** At startup each rank measures what the model leaves; the pair uses the smaller.
- **Bounded graphs.** At most 96 decode graphs stay instantiated (`TF_GLM_GRAPHS`); a new shape takes an old graph's
  place by an in-place update.
- **A watchdog on each rank.** A thread reads `MemAvailable` every 100 ms. Below 5 GiB no new request is admitted;
  below 2 GiB (`TF_MEM_FLOOR_GIB`) the rank exits, which frees its memory. A rank whose peer is gone (the control
  link closed or timed out, or NCCL reports an error) exits too, so `spark up` or the unit can start the pair again.
- **Prefer `free -h`** to `nvidia-smi` for reading a node's memory.

## Settings

`recipe.toml` holds the defaults, and `cluster.toml` (git-ignored) overrides them for a pair. An environment
variable of the same name in upper case overrides both (`CONTEXT=262144 python3 -m spark up`).
`python3 -m spark validate` checks every value before anything runs. The engine's own switches (`TF_GLM_*`, set
through `extra_env`) are listed in [`docker/patches/README.md`](docker/patches/README.md).

## Tests

- `python3 -m unittest discover -s tests -p "test_*.py"`: the settings, the orchestrator, the quality checkers.
- `python3 -m quality.suite --url http://127.0.0.1:<port> --model GLM-5.3-Flash-EXL3 --out <dir>`: drafts equal
  undrafted, the serving behaviours, tool calls, JSON, IFEval, GSM8K, MGSM, MMLU, MATH-500, GPQA-Diamond, HumanEval
  (sandboxed), repetition, long-context needles, long-context tasks (RULER-style), the kept-state soak, vision,
  ChartQA and DocVQA, and the concurrency checks.
- `bench/ruler.py --url ... --concurrency 1 8`: the decode ruler.
- `bench/longctx.py --url ... --out DIR [--sizes 65536,131072,262144] [--streams N]`: prompts at exact token
  counts with two buried facts: prefill rate, time to the first token, decode after, whether the reply has the facts.
- The engine's own checks, on the patched TensorFold tree: `tf-cuda-test glm-batch` (16 streams together equal to
  each alone, serial and drafted; resumed equal to fresh).

## Credits

This recipe is a thin layer over other people's work. Each line names what was taken and its licence.

- **The model:** [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai (MIT).
- **The checkpoint:** [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)
  by Mia's AI Lab (Apache-2.0), served as published by default, and its abliterated form, the `ablit = true` option,
  [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit)
  (gated; the abliteration, its verification and the responsible-use terms are Mia's AI Lab's, on the model's page).
  `ablit = true` follows the `ABLIT=1` mode of their Python-engine recipe (v1.7): the same pinned revision, thinking
  off by default.
- **The recipe this one follows:** Mia's AI Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (Apache-2.0) for TensorFold's Python engine: the target numbers, the 16-row verify windows and copy drafts, the
  DFlash2 chain cut by the drafts' probability product (their `fnc7:0.3`; its noise-aware form was measured on this
  engine and left out, no gain), and the per-reply `tensorfold` statistics the comparisons in `evidence/` read.
  **The EXL3 experts' prompt kernel follows their design** (their patches 0004 and 0009, Apache-2.0): the decoded
  trellis tiles as the tensor cores' A operand with the prompt rows streamed past them, gate and up fused with the
  SwiGLU, the down projection with its epilogue; written for this engine's kernels (`docker/patches/README.md`).
- **The engine:** [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and the TensorFold contributors
  (Apache-2.0): the native engine and server the patch extends, its lane core (the draft depth rule follows the Python
  engine's `family_depth.py`), and the Python engine's GLM kernels the CUDA kernels were checked against.
- **The EXL3 format:** [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT); the expert kernels
  read its trellis format.
- **The RoCE all-gather:** [b12x](https://github.com/local-inference-lab/b12x) by local-inference-lab (Apache-2.0):
  the one-shot "RoCEnante" RDMA exchange the ranks' small gathers follow.
- **DFlash2:** the architecture from [z-lab/dflash](https://github.com/z-lab/dflash) by Z Lab (MIT); the drafter
  weights [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) by IncoAI
  (CC BY-NC-ND 4.0, non-commercial; the default drafter, downloaded from IncoAI at a pinned revision;
  `drafter = "mtp"` for commercial use).
- **The cross-recipe benchmark:** the prompt set and harness (`bench_tp2_night.py`) of
  [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  by tonyd2wild, whose speed stack is
  [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) by knapcio
  (MIT). Their published results are the vLLM NVFP4 numbers in `evidence/dev/sweep-2026-10-13/tonyd-harness/`;
  the harness is run from their repository and not redistributed here.
- **Datasets:** the quality suite's datasets and their licences are listed in NOTICE.

This recipe's files are MIT (LICENSE); the patch is Apache-2.0 like the code it changes (NOTICE).
