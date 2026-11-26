# vLLM `prompt_logprobs` Spillover Benchmark: Can Auditing 5% of Agent Turns Slow the Other 95%?

Empirical vLLM `v0.34.0` serving benchmark measuring cross-cohort tail-latency spillover when a sample of production agent turns requests full-prompt log probabilities (`prompt_logprobs`).

## Mechanism & Hypothesis

- **Mechanism (vLLM v0.34.0)**:
  - When a request sets `prompt_logprobs`, `Request.get_prefix_cache_hit_limit()` returns `0`, causing `KVCacheManager.get_computed_blocks()` to bypass prefix-cache reads and recompute full-prompt prefill.
  - `KVCacheManager.record_prefix_cache_stats()` skips requests where `prefix_cache_lookup_enabled(request)` is `False`, excluding those skipped lookups from `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`.
  - Meanwhile, `prompt_logprobs > 0` is rejected when `stream=True` (`vllm/entrypoints/openai/completion/protocol.py`), so audited turns and their matched controls must use the nonstreaming `/v1/completions` endpoint.
- **Hypothesis**:
  - At **75% of measured baseline capacity**, auditing **5% of turns** raises **p95 TTFT** for a separate, unscored streaming cohort by **at least 20%**, while the reported prefix-cache hit ratio moves by **less than 3 percentage points**.

## Experimental Design

- **Server**: `vllm==0.34.0` serving `Qwen/Qwen2.5-3B-Instruct` in `bfloat16` on 1× NVIDIA A100-SXM4-80GB (`GPU 0`, port `8025`).
- **Workload**: 24 seeded, 32-turn sessions (768 total turns), each with a distinct 4,096-token starting history (primed during fixed warm-up) and 128 new tokens appended per turn (`4,224 -> 8,192` prompt tokens), fixed `output_tokens=64`, `ignore_eos=True`, `temperature=0.0`.
- **Cohort Isolation**:
  - **Sessions 0–15 (512 turns)**: Separate, unscored streaming measurement cohort (`stream=True`, never audited across any condition).
  - **Sessions 16–23 (256 turns)**: Nonstreaming audit/control pool (`stream=False` in all conditions). Audited turns are nested subsets (`1% ⊂ 5% ⊂ 10%`) of this pool, paired against exact nonstreaming matched controls at `0%`.
- **Arrival Schedule**: Stationary round-robin turn-shuffled Poisson arrivals (fixed per arrival seed: `101, 202, 303`) with boundary separation so consecutive turns of the same session do not collide and there is no ramp-down tail bias.
- **Metrics**:
  - Client-side per-request TTFT, E2E, TPOT, TTFB, body transfer time, and response payload bytes (`requests.jsonl`).
  - Server-side Prometheus `/metrics` deltas before/after each replay: `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (reported hit ratio), `vllm:request_prefill_kv_computed_tokens_sum` & `vllm:prompt_tokens_by_source{source="computed"}` (newly computed prefill tokens), `vllm:num_preemptions`, and fine-grained latency histograms (`server_ttft_s`, `server_queue_s`, `server_prefill_s`, `server_e2e_s`).

## Environment Requirements

- Linux host with 1× NVIDIA A100-SXM4-80GB (or comparable GPU; configurable via `GPU_ID`)
- Python 3.12 virtual environment (`.venv`) with `vllm==0.34.0`, `httpx`, `numpy`, and `matplotlib`
- Access to `Qwen/Qwen2.5-3B-Instruct` on Hugging Face (or set `MODEL` / `--model`)

## Quick Start

1. **Start vLLM `v0.34.0` on GPU 0 (in `tmux`/`screen` or a dedicated terminal)**:
   ```bash
   GPU_ID=0 PORT=8025 ./start_server.sh
   ```

2. **Run the full experiment (calibration → main 4×3 sweep → 3 hunts → plots)**:
   ```bash
   ./run_experiment.sh
   ```

3. **Or run individual stages**:
   ```bash
   # Baseline capacity calibration (0% audit)
   .venv/bin/python benchmark.py calibrate --rps-grid 4,8,12,16,20,24,28,32,36,40 --slo-p95-ms 500

   # Main sweep at 75% capacity (example: --rps 18.0)
   .venv/bin/python benchmark.py sweep --rps 18.0 --audit-fracs 0.0,0.01,0.05,0.10 --seeds 101,202,303

   # Three follow-up hunts:
   #   1) early_late: 5/10 late-turn audits vs 24/36 early-turn audits
   #   2) serialization: prompt_logprobs k=0 vs k=1 vs k=5 (client TTFT vs server queue time)
   #   3) load_boundary: 0% vs 5% auditing across 35%, 50%, 75%, 90% of capacity
   .venv/bin/python benchmark.py hunts --rps-75pct 18.0 --seeds 101,202,303

   # Plot Money Chart & Hunt diagnostics
   .venv/bin/python plot_results.py --results-dir results --out-dir plots
   ```

4. **Optional: Sync code to a remote GPU server / pull results locally**:
   ```bash
   REMOTE="user@gpu-host" REMOTE_DIR="/path/to/remote/benchmark" ./sync_to_server.sh push
   REMOTE="user@gpu-host" REMOTE_DIR="/path/to/remote/benchmark" ./sync_to_server.sh pull
   ```
   You can also store `REMOTE` and `REMOTE_DIR` in a local `.env.sync` file (ignored by git).
