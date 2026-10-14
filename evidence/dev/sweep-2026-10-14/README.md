# evidence/dev/sweep-2026-10-14

The day's dev points on bare builds of the port (`~/tf-zig-glm107`, branch `glm-v1.0.7`; `tf-cuda-test` and the
native server started by the dev scripts, not the image), on the published checkpoint. Profiles and batch checks
between 06:04 and 10:20 UTC ran NCCL on sockets (the container's SPCX plugin on the bare library path): their
all-gather rows read 27.7 s instead of 3-4 s; kernel rows are unaffected.

- `batch-*.rank0.log`: `tf-cuda-test glm-batch` on each kernel variant (old = the grouped decode kernel on prompt
  chunks, new/staged = the staged kernel, fused = the transposed design): all passed.
- `prof-*.rank0.log`: `glm-prof` with `TF_PROF_PROMPT=N` (256, 1024, 32768 tokens): the prompt's wall time and its
  kernels. The 32K experts: old 87.2 s (`prof-old-32k-roce`), staged 51.4 s (`prof-new-4096`), fused 7.2 s
  (`prof-dev6-fused-n32768`).
- `tf-*.rank0.log`: `glm-tf`, teacher-forced on the old kernel's own greedy replies (`ref-w1200.json`,
  `ref-w4000.json` from `long-old-2048.json` through `/tokenize`): fused 107/114 and 130/135 positions, the old
  kernel against itself 114/114 and 135/135.
- `ruler-*.json`, `ruler-*.rank0.log`: one-stream rulers on dev servers: the old, staged (`new-2048`, `new-4096`
  = `TF_GLM_PREFILL_ROWS`), fused (`dev6-fused`) kernels; `dev7-bare` (the fused default build, code and structured
  only); `dev8-headroom{0,8192,2048}` (`TF_GLM_KEEP_HEADROOM`, prose and code: the same within noise).
- `prefill-*.json`: tonyd2wild's harness, prefill suite, on the dev servers (fused: 993 / 1,003 / 964 tok/s).
- `long-*.json`: greedy replies to 1.2K-32K-token documents (`longcheck.py`); the staged and fused kernels' replies
  equal each other byte for byte and differ from the old kernel's by its summation order.
- `chat-resume-shipped.txt`: the previous image never resuming a chat's next turn; `chat-probe-dev8.txt`: the fix
  (turns 2 and 3 resume; a 67K-token chat's turn 2 in 0.20 s).
- `json/`: the JSON decode lever sweep (`json_exp2.sh`: selector temperature, copy drafts in the reply, the
  confidence cut, the window) on the dev8 server.
