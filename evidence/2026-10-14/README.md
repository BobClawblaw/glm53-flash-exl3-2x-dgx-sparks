# evidence/2026-10-14

The shipped image `tf-glm53-exl3:native-v1.0.7` with the fused EXL3 prompt kernel and the kept-state headroom (patch
sha256 14fb445e, port head c9ea9c2a), served by `spark up` on this pair, `recipe.toml` defaults, the published
checkpoint (`ablit = false`) unless a file name says otherwise. Files named `f95f2f70` are the same measurements on the
first build of the day (port head 09e84096: the fused kernel without the headroom fix; the decode and prompt paths are
the same code), kept as a second sample.

- `batch-default.rank0.log`: `tf-cuda-test glm-batch` on the build with no switches set (the fused kernel is the
  default): 16 streams together equal to each alone, serial and drafted, resumed equal to fresh; both ranks pass.
- `ruler-shipped-14fb445e-c1.json`, `-c8.json`, `-c16.json`: `bench/ruler.py`, greedy, one stream over all 16
  prompts a phase; eight streams, two waves, two runs; sixteen streams, two runs.
- `ruler-shipped-14fb445e-c1-sampled.json`: `bench/ruler_sampled.py` (temperature 1, top_p 0.95), one stream.
- `ruler-252-*.json`: Mia-AiLab's recipe on another pair (192.0.2.252, TensorFold's Python engine v0.6.0 with their
  patches), the same rulers, 2026-10-12 and -13 (copied from `../2026-10-13/`).
- `tonyd-harness/ours-shipped-14fb445e.json`: tonyd2wild's `bench_tp2_night.py` (greedy, `reasoning_effort: low`,
  single, sweep, prefill and longctx suites) against the shipped image at the recipe's 1M context; their published
  lane A result is `../2026-10-13/tonyd-harness/tonyd2wild-laneA-thinking-low-kv6.json`.
- `prompt-profile-32k.txt`: `tf-cuda-test glm-prof` with `TF_PROF_PROMPT=32768` on the shipped engine: a 32K-token
  prompt's wall time and its kernels (the experts' prompt kernels 7.2 s); `prompt-profile-32k-old-kernel.txt`: the
  same with `TF_GLM_EXL3_PROMPT=old`, the previous image's prompt path (87.2 s in the grouped kernel).
- `chat-resume-shipped-14fb445e.txt`: a chat's turns 2 and 3 resume turn 1's kept state (`cached_tokens`, time to
  the first token): a 4.2K-token chat, then 3,000- and 12,000-word documents; `-f95f2f70.txt` shows the first build
  never resuming (the headroom denied by a pool full of kept states).
- `long-shipped-14fb445e.json`: greedy replies to 1.2K-32K-token documents, compared with the dev build's.
- `suite-ab/`: the quality suite's reasoning and instruction checks on the first build (f95f2f70, the same prompt and
  decode code), against `../2026-10-13/suite/`
  (the old prompt kernel): the fused kernel's effect on scores.
- `../dev/sweep-2026-10-14/`: the day's dev points: the staged and fused prompt kernels' batch checks, profiles
  (`prof-*.rank0.log`), teacher-forced agreement with the old kernel's greedy replies (`tf-*.rank0.log`: fused
  107/114 and 130/135, the old kernel against itself 114/114 and 135/135), dev-server rulers, prefill suites and
  long replies. Profiles between 06:04 and 10:20 UTC ran NCCL on sockets (a 27.7 s all-gather); kernel rows unaffected.
- `ablit/`: the same measurements and the whole quality suite on the Ablit weights (the recipe's default).
