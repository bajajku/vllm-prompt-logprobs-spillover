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

## Empirical Results (`vllm==0.34.0`, `Qwen/Qwen2.5-3B-Instruct`, 1× A100-SXM4-80GB)

All **73 benchmark runs** (56,064 total requests across calibration, main sweep, and 3 diagnostic hunts) completed with **0 errors** and **0 KV-cache preemptions**.

- **Baseline Capacity Calibration (0% audit, SLO p95 TTFT ≤ 500 ms):** **`24.0 req/s`** (`p95 = 124.1 ms` at `24 req/s`, crossing the SLO at `28 req/s` with `p95 = 506.4 ms`).
- **Target 75% Load:** **`18.0 req/s`**.

### Main Sweep: Audit Fraction vs. Unscored Streaming Cohort (`18.0 req/s`, 3 seeds)

![Money Chart](plots/money_chart.png)

| Audit % | Unscored p50 TTFT | Unscored p95 TTFT (mean ± SD) | vs. 0% Baseline | Reported Hit % | True Effective Hit % | Newly Computed Prefill |
|---:|---:|---:|---:|---:|---:|---:|
| **0%** | 61.6 ms | **93.8 ± 9.1 ms** | +0.0% | **97.94%** | 97.94% | 98,304 tok (1.0×) |
| **1%** | 64.6 ms | **244.3 ± 34.5 ms** | **+160.3%** | **97.94%** | 96.98% | 144,000 tok (1.5×) |
| **5%** | 164.6 ms | **757.0 ± 212.0 ms** | **+706.6%** | **97.94%** | 93.16% | 326,144 tok (3.3×) |
| **10%** | 440.3 ms | **10,209.7 ± 1,615.3 ms** | **+10,778.9%** | **97.94%** | 88.00% | 572,160 tok (5.8×) |

**Hypothesis verdict — strongly confirmed:** At 75% load, auditing **5% of turns** raised unscored streaming **p95 TTFT by +706.6% (8.1×)** while newly computed prefill tokens jumped **3.3×** and the reported prefix-cache hit rate moved by **0.00 percentage points (`97.94% → 97.94%`)**.

### Diagnostic Hunts

1. **Hunt 1 — Early vs. Late Turns ([hunt1_early_vs_late.png](plots/hunt1_early_vs_late.png)):** Auditing just **10 late turns** (1.3% of turns, `176K` new prefill tokens) raised unscored p95 TTFT to **665.9 ms (+592.3%)**—worse than auditing **24 early turns** (3.1% of turns, `209K` new prefill tokens, **422.4 ms / +339.1%**).
2. **Hunt 2 — GPU Contention vs. Logprob Serialization ([hunt2_serialization_gap.png](plots/hunt2_serialization_gap.png)):** Even at `prompt_logprobs=0` (`0.47 MB` payload), server Prometheus p95 TTFT jumped **10×** (`72.5 ms → 745.5 ms`, client p95 `856.8 ms`), proving GPU prefill/queue contention is the primary driver. Increasing to `k=5` (`2.41 MB` payload) kept server TTFT flat (`715.5 ms`) while pushing client p95 to `1,030.9 ms` and client p50 from `162.5 ms → 377.5 ms`, exposing a secondary API-server JSON serialization stall.
3. **Hunt 3 — Load Boundary ([hunt3_load_boundary.png](plots/hunt3_load_boundary.png)):** 5% auditing severely degraded unscored p95 TTFT across every tested load: **+388.5%** at 35% load (`70.0 → 341.9 ms`), **+455.6%** at 50% load (`75.1 → 417.5 ms`), **+682.4%** at 75% load (`97.7 → 764.3 ms`), and **+869.5%** at 90% load (`172.1 → 1,668.0 ms`).

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
