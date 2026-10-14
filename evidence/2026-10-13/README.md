# evidence/2026-10-13

The shipped image `tf-glm53-exl3:native-v1.0.7` (patch sha256 6861d48f, branch head 6bfd5bec), served by `spark up`
on this pair, `recipe.toml` defaults unless a file name says otherwise.

- `ruler-deployed-6861d48f-c1.json`, `-c8.json`: `bench/ruler.py`, greedy, one stream over all 16 prompts a phase;
  eight streams, two waves, two runs.
- `ruler-deployed-6861d48f-c1-sampled.json`: `bench/ruler_sampled.py` (temperature 1, top_p 0.95), one stream.
- `ruler-deployed-6861d48f-roce2048-*.json`: the same with `TF_ROCE_MAX_KB=2048`.
- `ruler-252-c1.json`, `ruler-252-c8.json`, `ruler-252-c1-sampled.json`: Mia-AiLab's recipe on another pair
  (192.0.2.252, TensorFold's Python engine v0.6.0 with their patches), the same rulers, 2026-10-12 and -13.
- `tonyd-harness/ours-ctx262144.json`, `ours-ctx1048576.json`: tonyd2wild's `bench_tp2_night.py` (greedy,
  `reasoning_effort: low`, single, sweep, prefill and longctx suites) against the branch head's dev build at
  `--context 262144` and 1048576; `tonyd2wild-laneA-thinking-low-kv6.json` is their published lane A result
  (vLLM, NVFP4, 2x DGX Spark, 262K), copied from their repository for the comparison.
- `prompt-profile-32k.txt`: `tf-cuda-test glm-prof` with `TF_PROF_PROMPT=32768` on the branch head: a 32K-token
  prompt's wall time and its kernels.
- `../dev/sweep-2026-10-13/`: the day's A/B points on dev builds (draft rules, the drafter's split and attention,
  the draft head, the mix, copy thresholds), each a 16-prompt one-stream ruler with the server's trace log.

Render a comparison: `python3 bench/compare.py ruler OURS.json THEIRS.json` or `... night OURS.json THEIRS.json`.
