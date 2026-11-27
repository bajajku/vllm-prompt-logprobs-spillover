#!/usr/bin/env python3
"""
Empirical vLLM v0.34.0 prefix-cache bypass & tail-latency spillover benchmark.

Tests whether auditing a small fraction of agent turns with `prompt_logprobs`
(which bypasses prefix-cache reads and excludes skipped lookups from cache-hit
statistics in vLLM v0.34.0) degrades p95 TTFT for a separate, unscored
streaming cohort while leaving reported prefix-cache hit rate nearly unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
from multiprocessing.connection import wait as mp_wait
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8025"
PROM_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)$'
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
_WARMED_LOGPROBS_KS: set[int] = set()


@dataclass(frozen=True)
class TurnSpec:
    seq_idx: int
    session_id: int
    step_idx: int
    turn_depth: int  # Logical conversation depth in 0..num_turns-1 (stationary when staggered)
    scheduled_s: float
    cohort: str  # "streaming_unscored" | "nonstreaming_pool"
    audited: bool
    audit_action: str  # "none" | "logprobs" | "cache_miss"
    stream: bool
    prompt_len: int
    output_tokens: int


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus text exposition format into {metric_name: [(labels, val)]}."""
    metrics: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM_LINE_RE.match(line)
        if not m:
            continue
        name, label_str, val_str = m.groups()
        try:
            val = float(val_str)
        except ValueError:
            continue
        labels = dict(LABEL_RE.findall(label_str)) if label_str else {}
        metrics.setdefault(name, []).append((labels, val))
    return metrics


def prom_sum(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    candidates: list[str],
    label_filter: dict[str, str] | None = None,
) -> float | None:
    """Sum metric values matching the first present candidate name and optional labels."""
    for name in candidates:
        if name not in metrics:
            continue
        total = 0.0
        matched = False
        for labels, val in metrics[name]:
            if label_filter and any(labels.get(k) != v for k, v in label_filter.items()):
                continue
            total += val
            matched = True
        if matched:
            return total
    return None


def prom_hist_delta(
    before: dict[str, list[tuple[dict[str, str], float]]],
    after: dict[str, list[tuple[dict[str, str], float]]],
    base_name: str,
) -> dict[str, float | None]:
    """Compute exact delta mean and bucket-interpolated p50/p95/p99 from a Prometheus histogram."""
    b_sum = prom_sum(before, [f"{base_name}_sum"]) or 0.0
    a_sum = prom_sum(after, [f"{base_name}_sum"]) or 0.0
    b_cnt = prom_sum(before, [f"{base_name}_count"]) or 0.0
    a_cnt = prom_sum(after, [f"{base_name}_count"]) or 0.0
    d_sum = max(0.0, a_sum - b_sum)
    d_cnt = max(0.0, a_cnt - b_cnt)

    def collect_buckets(m: dict[str, list[tuple[dict[str, str], float]]]) -> dict[float, float]:
        out: dict[float, float] = {}
        for labels, val in m.get(f"{base_name}_bucket", []):
            le_str = labels.get("le")
            if le_str is None:
                continue
            le = math.inf if le_str == "+Inf" else float(le_str)
            out[le] = out.get(le, 0.0) + val
        return out

    b_bkt = collect_buckets(before)
    a_bkt = collect_buckets(after)
    bounds = sorted(set(a_bkt.keys()) | set(b_bkt.keys()))
    deltas = [(le, max(0.0, a_bkt.get(le, 0.0) - b_bkt.get(le, 0.0))) for le in bounds]

    def quantile(q: float) -> float | None:
        if d_cnt <= 0 or not deltas:
            return None
        target = q * d_cnt
        prev_le = 0.0
        prev_c = 0.0
        for le, c in deltas:
            if c >= target:
                if math.isinf(le):
                    return prev_le
                if c <= prev_c:
                    return le
                frac = (target - prev_c) / (c - prev_c)
                return prev_le + frac * (le - prev_le)
            prev_le = le
            prev_c = c
        return None

    return {
        "count": d_cnt,
        "sum": d_sum,
        "mean": (d_sum / d_cnt) if d_cnt > 0 else None,
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
    }


def summarize_metrics_delta(
    before: dict[str, list[tuple[dict[str, str], float]]],
    after: dict[str, list[tuple[dict[str, str], float]]],
) -> dict[str, Any]:
    """Extract prefix-cache, prefill-compute, and server-latency deltas."""
    def counter_delta(names: list[str], label_filter: dict[str, str] | None = None) -> float | None:
        a = prom_sum(after, names, label_filter)
        b = prom_sum(before, names, label_filter)
        if a is None:
            return None
        return max(0.0, a - (b or 0.0))

    pc_queries = counter_delta(["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"])
    pc_hits = counter_delta(["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"])
    prompt_total = counter_delta(["vllm:prompt_tokens_total", "vllm:prompt_tokens"])
    prompt_cached = counter_delta(["vllm:prompt_tokens_cached_total", "vllm:prompt_tokens_cached"])
    prompt_computed_src = counter_delta(
        ["vllm:prompt_tokens_by_source_total", "vllm:prompt_tokens_by_source"],
        {"source": "computed"},
    )
    gen_total = counter_delta(["vllm:generation_tokens_total", "vllm:generation_tokens"])
    preemptions = counter_delta(["vllm:num_preemptions_total", "vllm:num_preemptions"])
    prefill_kv_hist = prom_hist_delta(before, after, "vllm:request_prefill_kv_computed_tokens")

    reported_hit_ratio = (pc_hits / pc_queries) if (pc_queries and pc_queries > 0 and pc_hits is not None) else None
    newly_computed_prefill = (
        prefill_kv_hist["sum"]
        if prefill_kv_hist["count"] and prefill_kv_hist["count"] > 0
        else prompt_computed_src
    )
    true_hit_ratio = (
        1.0 - (newly_computed_prefill / prompt_total)
        if (prompt_total and prompt_total > 0 and newly_computed_prefill is not None)
        else None
    )

    return {
        "prefix_cache_queries": pc_queries,
        "prefix_cache_hits": pc_hits,
        "reported_cache_hit_ratio": reported_hit_ratio,
        "prompt_tokens_total": prompt_total,
        "prompt_tokens_cached_total": prompt_cached,
        "prompt_tokens_computed_by_source": prompt_computed_src,
        "prefill_kv_computed_tokens_sum": prefill_kv_hist["sum"],
        "newly_computed_prefill_tokens": newly_computed_prefill,
        "true_effective_cache_hit_ratio": true_hit_ratio,
        "generation_tokens_total": gen_total,
        "num_preemptions": preemptions,
        "kv_cache_usage_perc_after": prom_sum(after, ["vllm:kv_cache_usage_perc"]),
        "server_ttft_s": prom_hist_delta(before, after, "vllm:time_to_first_token_seconds"),
        "server_queue_s": prom_hist_delta(before, after, "vllm:request_queue_time_seconds"),
        "server_prefill_s": prom_hist_delta(before, after, "vllm:request_prefill_time_seconds"),
        "server_e2e_s": prom_hist_delta(before, after, "vllm:e2e_request_latency_seconds"),
    }


def compute_session_phases(num_sessions: int, num_turns: int, num_streaming_sessions: int) -> list[int]:
    """
    Assign staggered initial turn depths per session so that the aggregate
    prompt-length distribution is stationary across the entire replay window,
    uniformly covering depths 0..num_turns-1 in both cohorts.
    """
    phases = [0] * num_sessions
    for i in range(num_streaming_sessions):
        phases[i] = (i * num_turns) // max(1, num_streaming_sessions)
    pool_n = num_sessions - num_streaming_sessions
    for j in range(pool_n):
        offset = max(1, num_turns // max(1, 2 * pool_n))
        phases[num_streaming_sessions + j] = ((j * num_turns) // max(1, pool_n) + offset) % num_turns
    return phases


def generate_session_prompts(
    num_sessions: int,
    num_turns: int,
    num_streaming_sessions: int,
    init_tokens: int,
    turn_tokens: int,
    content_seed: int = 42,
    vocab_lo: int = 1000,
    vocab_hi: int = 100000,
) -> tuple[list[list[int]], list[list[list[int]]], list[int]]:
    """
    Generate deterministic, distinct per-session staggered conversations.

    Each session `s` starts at staggered depth `phase[s]` with a pre-primed
    conversation A of length `init_tokens + phase[s] * turn_tokens`. Once it
    reaches depth `num_turns - 1`, it rolls over to a second pre-primed 4K
    conversation B (`init_tokens`) so every single baseline turn hits the
    prefix cache and computes exactly `turn_tokens` (128) new tokens, while
    keeping the aggregate prompt-length distribution stationary across time.
    """
    rng = np.random.default_rng(content_seed)
    phases = compute_session_phases(num_sessions, num_turns, num_streaming_sessions)
    warmup_prefixes: list[list[int]] = []
    prompts: list[list[list[int]]] = []

    for s in range(num_sessions):
        p = phases[s]
        # Conversation A: already at depth `p` when replay starts.
        len_a = init_tokens + p * turn_tokens
        init_a = rng.integers(vocab_lo, vocab_hi, size=len_a, dtype=np.int64)
        init_a[0] = vocab_lo + s
        init_a[1] = vocab_lo + 500 + s
        init_a_list = init_a.tolist()
        warmup_prefixes.append(init_a_list)

        # Conversation B: fresh 4K history used after rollover (only needed when p > 0).
        init_b_list: list[int] | None = None
        if p > 0:
            init_b = rng.integers(vocab_lo, vocab_hi, size=init_tokens, dtype=np.int64)
            init_b[0] = vocab_lo + 2000 + s
            init_b[1] = vocab_lo + 2500 + s
            init_b_list = init_b.tolist()
            warmup_prefixes.append(init_b_list)

        s_prompts: list[list[int]] = []
        acc_a = list(init_a_list)
        acc_b = list(init_b_list) if init_b_list is not None else []
        for step in range(num_turns):
            chunk = rng.integers(vocab_lo, vocab_hi, size=turn_tokens, dtype=np.int64).tolist()
            if p + step < num_turns:
                acc_a = acc_a + chunk
                s_prompts.append(acc_a)
            else:
                acc_b = acc_b + chunk
                s_prompts.append(acc_b)
        prompts.append(s_prompts)

    return warmup_prefixes, prompts, phases


def build_schedule(
    num_sessions: int,
    num_turns: int,
    num_streaming_sessions: int,
    rps: float,
    arrival_seed: int,
    audit_frac: float,
    audit_mode: str,
    audit_count: int | None,
    output_tokens: int,
    init_tokens: int,
    turn_tokens: int,
    audit_seed: int | None = None,
    audit_action: str = "logprobs",
) -> list[TurnSpec]:
    """
    Build a fixed open-loop replay schedule:
    - Stationary Poisson aggregate arrival timestamps at rate `rps` (seeded by `arrival_seed`).
    - Round-robin turn-shuffled session assignment with boundary separation so consecutive
      turns of the same session do not arrive back-to-back and there is no tail ramp-down.
    - Staggered session turn depths so prompt lengths are stationary over the run.
    - Fixed streaming cohort (`sessions 0..num_streaming_sessions-1`) that is NEVER audited.
    - Nonstreaming pool (`sessions num_streaming_sessions..num_sessions-1`) where audited turns
      and their matched controls all use `stream=False`.
    - Strictly nested audit subsets per replicate (`1% ⊂ 5% ⊂ 10%`), with `audit_seed`
      varying across replicates (`9000 + arrival_seed` by default).
    """
    total_turns = num_sessions * num_turns
    phases = compute_session_phases(num_sessions, num_turns, num_streaming_sessions)
    effective_audit_seed = (9000 + arrival_seed) if audit_seed is None else audit_seed

    rng_arr = random.Random(arrival_seed)
    np_rng = np.random.default_rng(arrival_seed)

    # Generate stationary Poisson timestamps for all turns.
    inter_arrivals = np_rng.exponential(scale=1.0 / rps, size=total_turns)
    timestamps = np.cumsum(inter_arrivals)
    timestamps = timestamps - timestamps[0]

    # Order sessions per turn round, avoiding back-to-back session repeats across round boundaries.
    order: list[tuple[int, int]] = []
    prev_tail: set[int] = set()
    sep = min(4, num_sessions // 3)
    for step in range(num_turns):
        sess = list(range(num_sessions))
        for _ in range(50):
            rng_arr.shuffle(sess)
            if not (set(sess[:sep]) & prev_tail):
                break
        prev_tail = set(sess[-sep:]) if sep > 0 else set()
        for s in sess:
            order.append((s, step))

    # Determine which (session_id, step_idx) pairs are audited from the nonstreaming pool.
    pool_sessions = list(range(num_streaming_sessions, num_sessions))
    rng_aud = random.Random(effective_audit_seed)
    audited_set: set[tuple[int, int]] = set()

    if audit_mode == "uniform":
        target_n = audit_count if audit_count is not None else int(round(audit_frac * total_turns))
        # Stratified-shuffled permutation across logical turn depths so 1% ⊂ 5% ⊂ 10%
        # evenly spans early/mid/late prompt lengths AND wall-clock time.
        by_depth: dict[int, list[tuple[int, int]]] = {d: [] for d in range(num_turns)}
        for s in pool_sessions:
            for step in range(num_turns):
                depth = (phases[s] + step) % num_turns
                by_depth[depth].append((s, step))
        for d in range(num_turns):
            rng_aud.shuffle(by_depth[d])
        ranked: list[tuple[int, int]] = []
        depth_order = list(range(num_turns))
        rng_aud.shuffle(depth_order)
        for slot in range(len(pool_sessions)):
            for idx in range(num_turns):
                d = depth_order[(idx * 7 + slot * 3) % num_turns]
                if slot < len(by_depth[d]):
                    ranked.append(by_depth[d][slot])
        seen: set[tuple[int, int]] = set()
        unique_ranked = []
        for item in ranked:
            if item not in seen:
                seen.add(item)
                unique_ranked.append(item)
        if target_n > len(unique_ranked):
            raise ValueError(f"Requested {target_n} audits, but nonstreaming pool only has {len(unique_ranked)} turns.")
        audited_set = set(unique_ranked[:target_n])
    elif audit_mode in ("early", "late"):
        depth_range = set(range(0, min(8, num_turns))) if audit_mode == "early" else set(range(max(0, num_turns - 8), num_turns))
        candidates = [
            (s, step)
            for s in pool_sessions
            for step in range(num_turns)
            if ((phases[s] + step) % num_turns) in depth_range
        ]
        rng_aud.shuffle(candidates)
        target_n = audit_count if audit_count is not None else int(round(audit_frac * total_turns))
        if target_n > len(candidates):
            raise ValueError(f"Requested {target_n} {audit_mode} audits, but only {len(candidates)} candidate turns exist.")
        audited_set = set(candidates[:target_n])
    else:
        raise ValueError(f"Unknown audit_mode: {audit_mode}")

    specs: list[TurnSpec] = []
    for seq_idx, ((s, step), ts) in enumerate(zip(order, timestamps)):
        is_streaming = s < num_streaming_sessions
        is_audited = (s, step) in audited_set
        depth = (phases[s] + step) % num_turns
        assert not (is_streaming and is_audited), "Streaming cohort must never be audited."
        specs.append(
            TurnSpec(
                seq_idx=seq_idx,
                session_id=s,
                step_idx=step,
                turn_depth=depth,
                scheduled_s=float(ts),
                cohort="streaming_unscored" if is_streaming else "nonstreaming_pool",
                audited=is_audited,
                audit_action=audit_action if is_audited else "none",
                stream=is_streaming,
                prompt_len=init_tokens + (depth + 1) * turn_tokens,
                output_tokens=output_tokens,
            )
        )
    return specs


async def fetch_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    resp = await client.get(f"{base_url}/metrics", timeout=30.0)
    resp.raise_for_status()
    return parse_prometheus(resp.text)


async def reset_prefix_cache(client: httpx.AsyncClient, base_url: str) -> bool:
    """Attempt to reset vLLM's prefix cache via /reset_prefix_cache."""
    try:
        resp = await client.post(f"{base_url}/reset_prefix_cache", timeout=30.0)
        return resp.status_code == 200
    except Exception:
        return False


async def warmup_logprobs_kernels(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    ks: tuple[int, ...] = (0, 1, 5),
    prompt_lens: tuple[int, ...] = (4224, 8192),
) -> float:
    """
    Pre-compile vLLM Triton/Torch prompt_logprobs kernels (e.g., _topk_log_softmax_kernel)
    before resetting the prefix cache or starting timed measurement runs.
    """
    missing = [k for k in sorted(set(ks)) if k not in _WARMED_LOGPROBS_KS]
    if not missing:
        return 0.0
    t0 = time.monotonic()
    rng = np.random.default_rng(777)
    for k in missing:
        for idx, plen in enumerate(prompt_lens):
            prompt_ids = rng.integers(1000, 50000, size=plen, dtype=np.int64)
            prompt_ids[0] = 109000 + k * 10 + idx
            payload = {
                "model": model,
                "prompt": prompt_ids.tolist(),
                "max_tokens": 2,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": True,
                "stream": False,
                "prompt_logprobs": k,
            }
            resp = await client.post(f"{base_url}/v1/completions", json=payload, timeout=180.0)
            resp.raise_for_status()
        _WARMED_LOGPROBS_KS.add(k)
    return time.monotonic() - t0


async def warmup_sessions(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    warmup_prefixes: list[list[int]],
    warmup_output_tokens: int = 8,
    concurrency: int = 6,
) -> float:
    """Prime all staggered starting histories (and rollover bases) into the KV prefix cache."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()

    async def prime_one(prompt_ids: list[int]) -> None:
        payload = {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": warmup_output_tokens,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "ignore_eos": True,
            "stream": False,
        }
        async with sem:
            resp = await client.post(f"{base_url}/v1/completions", json=payload, timeout=180.0)
            resp.raise_for_status()

    await asyncio.gather(*(prime_one(h) for h in warmup_prefixes))
    return time.monotonic() - t0


async def send_turn(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    spec: TurnSpec,
    prompt_ids: list[int],
    audit_logprobs_k: int,
    t_run_start: float,
) -> tuple[dict[str, Any], bytes | None]:
    """
    Sleep until scheduled open-loop arrival time, then send request and record latencies.
    For nonstreaming requests, JSON decoding is strictly deferred post-run so large
    prompt_logprobs payloads never block the asyncio event loop during live replay.
    """
    now = time.monotonic()
    delay = (t_run_start + spec.scheduled_s) - now
    if delay > 0:
        await asyncio.sleep(delay)

    t_send = time.monotonic()
    dispatch_lag_ms = (t_send - (t_run_start + spec.scheduled_s)) * 1000.0

    # For cache_miss-only control, mutate token 0 to a unique uncached token ID
    # (outside [1000, 100000) and within Qwen2.5 vocab) without requesting prompt_logprobs.
    effective_prompt = prompt_ids
    if spec.audited and spec.audit_action == "cache_miss":
        effective_prompt = list(prompt_ids)
        effective_prompt[0] = 100500 + spec.seq_idx

    payload: dict[str, Any] = {
        "model": model,
        "prompt": effective_prompt,
        "max_tokens": spec.output_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "stream": spec.stream,
    }
    if spec.stream:
        payload["stream_options"] = {"include_usage": True}
    if spec.audited and spec.audit_action == "logprobs":
        payload["prompt_logprobs"] = audit_logprobs_k

    record: dict[str, Any] = asdict(spec)
    record["dispatch_lag_ms"] = dispatch_lag_ms
    record["audit_logprobs_k"] = audit_logprobs_k if (spec.audited and spec.audit_action == "logprobs") else None
    record["t_send_rel_s"] = t_send - t_run_start
    raw_body: bytes | None = None

    try:
        if spec.stream:
            t_first: float | None = None
            t_headers: float | None = None
            resp_bytes = 0
            usage: dict[str, Any] = {}
            chunk_count = 0
            async with client.stream(
                "POST", f"{base_url}/v1/completions", json=payload, timeout=300.0
            ) as resp:
                t_headers = time.monotonic()
                record["http_status"] = resp.status_code
                if resp.status_code != 200:
                    body = await resp.aread()
                    record["error"] = body.decode("utf-8", errors="replace")[:500]
                    return record, None
                async for line in resp.aiter_lines():
                    resp_bytes += len(line) + 1
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    obj = json.loads(data_str)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        txt = choices[0].get("text")
                        if txt is not None and txt != "":
                            chunk_count += 1
                            if t_first is None:
                                t_first = time.monotonic()
            t_end = time.monotonic()
            comp_tokens = usage.get("completion_tokens") or chunk_count
            record["ttfb_ms"] = (t_headers - t_send) * 1000.0 if t_headers else None
            record["ttft_ms"] = (t_first - t_send) * 1000.0 if t_first else None
            record["e2e_ms"] = (t_end - t_send) * 1000.0
            record["tpot_ms"] = (
                ((t_end - t_first) * 1000.0 / (comp_tokens - 1))
                if (t_first is not None and comp_tokens and comp_tokens > 1)
                else None
            )
            record["response_bytes"] = resp_bytes
            record["chunks"] = chunk_count
            record["prompt_tokens"] = usage.get("prompt_tokens")
            record["completion_tokens"] = usage.get("completion_tokens")
            ptd = usage.get("prompt_tokens_details") or {}
            record["cached_tokens"] = ptd.get("cached_tokens")
        else:
            async with client.stream(
                "POST", f"{base_url}/v1/completions", json=payload, timeout=300.0
            ) as resp:
                t_headers = time.monotonic()
                body = await resp.aread()
                t_end = time.monotonic()
                record["http_status"] = resp.status_code
                record["response_bytes"] = len(body)
                record["ttfb_ms"] = (t_headers - t_send) * 1000.0
                record["e2e_ms"] = (t_end - t_send) * 1000.0
                record["body_transfer_ms"] = (t_end - t_headers) * 1000.0
                if resp.status_code != 200:
                    record["error"] = body.decode("utf-8", errors="replace")[:500]
                    return record, None
                raw_body = body
    except Exception as exc:
        record["http_status"] = -1
        record["error"] = repr(exc)
    return record, raw_body


def _parse_nonstreaming_body(record: dict[str, Any], raw_body: bytes | None) -> dict[str, Any]:
    """Post-run JSON parse for nonstreaming responses (off the live replay critical path)."""
    if raw_body is None:
        return record
    try:
        obj = json.loads(raw_body)
        usage = obj.get("usage") or {}
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")
        ptd = usage.get("prompt_tokens_details") or {}
        record["cached_tokens"] = ptd.get("cached_tokens")
        choices = obj.get("choices") or []
        if choices and choices[0].get("prompt_logprobs") is not None:
            record["prompt_logprobs_len"] = len(choices[0]["prompt_logprobs"])
    except Exception as exc:
        record["parse_error"] = repr(exc)
    return record


def _cohort_worker_main(
    conn: Any,
    base_url: str,
    model: str,
    specs: list[TurnSpec],
    audit_logprobs_k: int,
    num_sessions: int,
    num_turns: int,
    num_streaming_sessions: int,
    init_tokens: int,
    turn_tokens: int,
    content_seed: int,
) -> None:
    """
    Isolated OS process for a single cohort (either streaming_unscored or nonstreaming_pool),
    eliminating any client-side event-loop, GC, or JSON-parsing cross-talk between cohorts.
    """
    _, prompts, _ = generate_session_prompts(
        num_sessions=num_sessions,
        num_turns=num_turns,
        num_streaming_sessions=num_streaming_sessions,
        init_tokens=init_tokens,
        turn_tokens=turn_tokens,
        content_seed=content_seed,
    )

    async def _run() -> list[dict[str, Any]]:
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
        async with httpx.AsyncClient(limits=limits) as client:
            conn.send("READY")
            t_run_start = float(conn.recv())
            tasks = [
                asyncio.create_task(
                    send_turn(
                        client=client,
                        base_url=base_url,
                        model=model,
                        spec=spec,
                        prompt_ids=prompts[spec.session_id][spec.step_idx],
                        audit_logprobs_k=audit_logprobs_k,
                        t_run_start=t_run_start,
                    )
                )
                for spec in specs
            ]
            raw_results = await asyncio.gather(*tasks)
        # Parse all nonstreaming JSON bodies AFTER replay finishes.
        return [_parse_nonstreaming_body(rec, body) for rec, body in raw_results]

    records = asyncio.run(_run())
    conn.send(records)
    conn.close()


def replay_isolated_cohorts(
    *,
    base_url: str,
    model: str,
    specs: list[TurnSpec],
    audit_logprobs_k: int,
    num_sessions: int,
    num_turns: int,
    num_streaming_sessions: int,
    init_tokens: int,
    turn_tokens: int,
    content_seed: int,
) -> tuple[list[dict[str, Any]], float]:
    """Run streaming and nonstreaming cohorts in separate OS processes synced on CLOCK_MONOTONIC."""
    streaming_specs = [s for s in specs if s.stream]
    nonstreaming_specs = [s for s in specs if not s.stream]

    ctx = multiprocessing.get_context("spawn")
    parent_s, child_s = ctx.Pipe()
    parent_n, child_n = ctx.Pipe()

    common_args = (
        base_url,
        model,
    )
    tail_args = (
        audit_logprobs_k,
        num_sessions,
        num_turns,
        num_streaming_sessions,
        init_tokens,
        turn_tokens,
        content_seed,
    )
    proc_s = ctx.Process(target=_cohort_worker_main, args=(child_s, *common_args, streaming_specs, *tail_args))
    proc_n = ctx.Process(target=_cohort_worker_main, args=(child_n, *common_args, nonstreaming_specs, *tail_args))
    proc_s.start()
    proc_n.start()
    child_s.close()
    child_n.close()

    # Wait for both processes to finish prompt setup and open HTTP pools.
    assert parent_s.recv() == "READY"
    assert parent_n.recv() == "READY"
    t_run_start = time.monotonic() + 0.5
    parent_s.send(t_run_start)
    parent_n.send(t_run_start)

    records: list[dict[str, Any]] = []
    pending = [parent_s, parent_n]
    while pending:
        for ready_conn in mp_wait(pending):
            records.extend(ready_conn.recv())
            ready_conn.close()
            pending.remove(ready_conn)

    wall_s = time.monotonic() - t_run_start
    proc_s.join()
    proc_n.join()
    if proc_s.exitcode != 0 or proc_n.exitcode != 0:
        raise RuntimeError(f"Cohort worker failed: streaming={proc_s.exitcode}, nonstreaming={proc_n.exitcode}")

    records.sort(key=lambda r: r["seq_idx"])
    return records, wall_s


def percentile_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


async def run_single(
    *,
    base_url: str,
    model: str,
    rps: float,
    audit_frac: float,
    arrival_seed: int,
    audit_mode: str = "uniform",
    audit_action: str = "logprobs",
    audit_count: int | None = None,
    audit_logprobs_k: int = 1,
    audit_seed: int | None = None,
    num_sessions: int = 24,
    num_turns: int = 32,
    num_streaming_sessions: int = 16,
    init_tokens: int = 4096,
    turn_tokens: int = 128,
    output_tokens: int = 64,
    content_seed: int = 42,
    out_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """Execute one full benchmark run (JIT warmup -> reset -> prefix warmup -> isolated replay -> metrics)."""
    effective_audit_seed = (9000 + arrival_seed) if audit_seed is None else audit_seed
    warmup_prefixes, _, phases = generate_session_prompts(
        num_sessions=num_sessions,
        num_turns=num_turns,
        num_streaming_sessions=num_streaming_sessions,
        init_tokens=init_tokens,
        turn_tokens=turn_tokens,
        content_seed=content_seed,
    )

    specs = build_schedule(
        num_sessions=num_sessions,
        num_turns=num_turns,
        num_streaming_sessions=num_streaming_sessions,
        rps=rps,
        arrival_seed=arrival_seed,
        audit_frac=audit_frac,
        audit_mode=audit_mode,
        audit_count=audit_count,
        output_tokens=output_tokens,
        init_tokens=init_tokens,
        turn_tokens=turn_tokens,
        audit_seed=effective_audit_seed,
        audit_action=audit_action,
    )

    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    async with httpx.AsyncClient(limits=limits) as client:
        vllm_ver = None
        try:
            vr = await client.get(f"{base_url}/version", timeout=10.0)
            if vr.status_code == 200:
                vllm_ver = vr.json().get("version")
        except Exception:
            pass

        # Pre-warm Triton logprobs kernels once per process before cache reset / timing.
        jit_warmup_s = await warmup_logprobs_kernels(
            client,
            base_url,
            model,
            ks=tuple(sorted({0, 1, 5, audit_logprobs_k})),
        )
        cache_reset_ok = await reset_prefix_cache(client, base_url)
        if not cache_reset_ok:
            raise RuntimeError(
                f"POST {base_url}/reset_prefix_cache failed. "
                "Start vLLM with VLLM_SERVER_DEV_MODE=1 to guarantee inter-run prefix-cache isolation."
            )
        warmup_s = await warmup_sessions(client, base_url, model, warmup_prefixes)
        await asyncio.sleep(2.0)
        metrics_before = await fetch_metrics(client, base_url)

    # Run streaming and nonstreaming cohorts in separate OS processes to prevent client cross-talk.
    records, wall_s = replay_isolated_cohorts(
        base_url=base_url,
        model=model,
        specs=specs,
        audit_logprobs_k=audit_logprobs_k,
        num_sessions=num_sessions,
        num_turns=num_turns,
        num_streaming_sessions=num_streaming_sessions,
        init_tokens=init_tokens,
        turn_tokens=turn_tokens,
        content_seed=content_seed,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        await asyncio.sleep(2.0)
        metrics_after = await fetch_metrics(client, base_url)

    metrics_delta = summarize_metrics_delta(metrics_before, metrics_after)
    ok_records = [r for r in records if r.get("http_status") == 200]
    err_records = [r for r in records if r.get("http_status") != 200]
    streaming_ok = [r for r in ok_records if r["cohort"] == "streaming_unscored"]
    audited_ok = [r for r in ok_records if r["audited"]]
    control_ok = [r for r in ok_records if r["cohort"] == "nonstreaming_pool" and not r["audited"]]
    achieved_rps = len(ok_records) / wall_s if wall_s > 0 else 0.0

    summary: dict[str, Any] = {
        "run_name": run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vllm_version": vllm_ver,
        "model": model,
        "cache_reset_ok": cache_reset_ok,
        "config": {
            "rps": rps,
            "audit_frac": audit_frac,
            "audit_mode": audit_mode,
            "audit_action": audit_action,
            "audit_count_actual": sum(1 for s in specs if s.audited),
            "audit_logprobs_k": audit_logprobs_k,
            "arrival_seed": arrival_seed,
            "audit_seed": effective_audit_seed,
            "staggered_depths": True,
            "session_phases": phases,
            "isolated_cohort_processes": True,
            "num_sessions": num_sessions,
            "num_turns": num_turns,
            "num_streaming_sessions": num_streaming_sessions,
            "init_tokens": init_tokens,
            "turn_tokens": turn_tokens,
            "output_tokens": output_tokens,
            "content_seed": content_seed,
        },
        "timing": {
            "jit_warmup_s": jit_warmup_s,
            "warmup_s": warmup_s,
            "schedule_span_s": specs[-1].scheduled_s if specs else 0.0,
            "wall_s": wall_s,
            "achieved_rps": achieved_rps,
            "unstable_overloaded": achieved_rps < 0.90 * rps,
        },
        "counts": {
            "total": len(records),
            "ok": len(ok_records),
            "errors": len(err_records),
            "streaming_ok": len(streaming_ok),
            "audited_ok": len(audited_ok),
            "control_ok": len(control_ok),
        },
        "streaming_unscored": {
            "ttft_ms": percentile_stats([r["ttft_ms"] for r in streaming_ok if r.get("ttft_ms") is not None]),
            "e2e_ms": percentile_stats([r["e2e_ms"] for r in streaming_ok if r.get("e2e_ms") is not None]),
            "tpot_ms": percentile_stats([r["tpot_ms"] for r in streaming_ok if r.get("tpot_ms") is not None]),
            "dispatch_lag_ms": percentile_stats([r["dispatch_lag_ms"] for r in streaming_ok]),
        },
        "nonstreaming_audited": {
            "e2e_ms": percentile_stats([r["e2e_ms"] for r in audited_ok if r.get("e2e_ms") is not None]),
            "ttfb_ms": percentile_stats([r["ttfb_ms"] for r in audited_ok if r.get("ttfb_ms") is not None]),
            "body_transfer_ms": percentile_stats(
                [r["body_transfer_ms"] for r in audited_ok if r.get("body_transfer_ms") is not None]
            ),
            "response_bytes": percentile_stats(
                [float(r["response_bytes"]) for r in audited_ok if r.get("response_bytes") is not None]
            ),
            "dispatch_lag_ms": percentile_stats([r["dispatch_lag_ms"] for r in audited_ok]),
        },
        "nonstreaming_control": {
            "e2e_ms": percentile_stats([r["e2e_ms"] for r in control_ok if r.get("e2e_ms") is not None]),
            "response_bytes": percentile_stats(
                [float(r["response_bytes"]) for r in control_ok if r.get("response_bytes") is not None]
            ),
            "dispatch_lag_ms": percentile_stats([r["dispatch_lag_ms"] for r in control_ok]),
        },
        "server_metrics": metrics_delta,
    }

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "requests.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    st = summary["streaming_unscored"]["ttft_ms"]
    dl = summary["streaming_unscored"]["dispatch_lag_ms"]
    sm = summary["server_metrics"]
    hit_pct = (sm["reported_cache_hit_ratio"] * 100.0) if sm["reported_cache_hit_ratio"] is not None else float("nan")
    true_pct = (
        (sm["true_effective_cache_hit_ratio"] * 100.0)
        if sm["true_effective_cache_hit_ratio"] is not None
        else float("nan")
    )
    prefill_tok = sm["newly_computed_prefill_tokens"] or 0.0
    flag = " [UNSTABLE]" if summary["timing"]["unstable_overloaded"] else ""
    print(
        f"[{run_name}] ok={len(ok_records)}/{len(records)} | "
        f"unscored p50={st['p50']:.1f}ms p95={st['p95']:.1f}ms p99={st['p99']:.1f}ms | "
        f"dlag_p95={dl['p95']:.1f}ms | "
        f"reported_hit={hit_pct:.2f}% true_hit={true_pct:.2f}% | "
        f"new_prefill={prefill_tok:,.0f} | wall={wall_s:.1f}s{flag}"
    )
    return summary


async def run_calibrate(args: argparse.Namespace) -> None:
    """Sweep baseline (0% audit) arrival rates to measure 100% sustainable capacity."""
    out_dir = Path(args.out_dir) / "calibration"
    rps_grid = [float(x) for x in args.rps_grid.split(",")]
    results = []
    for rps in rps_grid:
        res = await run_single(
            base_url=args.base_url,
            model=args.model,
            rps=rps,
            audit_frac=0.0,
            arrival_seed=args.seed,
            audit_mode="uniform",
            audit_logprobs_k=args.audit_logprobs_k,
            num_sessions=args.num_sessions,
            num_turns=args.num_turns,
            num_streaming_sessions=args.num_streaming_sessions,
            init_tokens=args.init_tokens,
            turn_tokens=args.turn_tokens,
            output_tokens=args.output_tokens,
            out_dir=out_dir,
            run_name=f"cal_rps{rps:.2f}_s{args.seed}",
        )
        results.append(res)

    # Identify baseline capacity: highest RPS with zero errors, p95 TTFT <= threshold,
    # and achieved_rps >= 0.90 * target_rps (no severe queue divergence).
    sustainable = []
    for r in results:
        p95 = r["streaming_unscored"]["ttft_ms"]["p95"]
        ach = r["timing"]["achieved_rps"]
        tgt = r["config"]["rps"]
        errs = r["counts"]["errors"]
        if errs == 0 and p95 is not None and p95 <= args.slo_p95_ms and ach >= 0.90 * tgt:
            sustainable.append(tgt)
    cap_rps = max(sustainable) if sustainable else rps_grid[0]
    cal_summary = {
        "slo_p95_ms": args.slo_p95_ms,
        "rps_grid": rps_grid,
        "measured_baseline_capacity_rps": cap_rps,
        "recommended_75pct_rps": round(0.75 * cap_rps, 3),
        "runs": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "calibration_summary.json").open("w") as f:
        json.dump(cal_summary, f, indent=2)
    print(
        f"\nCalibration complete: baseline capacity = {cap_rps:.2f} req/s "
        f"(SLO p95 <= {args.slo_p95_ms:.0f} ms) -> 75% target = {0.75 * cap_rps:.2f} req/s"
    )


async def run_sweep(args: argparse.Namespace) -> None:
    """Run main audit-fraction sweep (0%, 1%, 5%, 10%) across 3 arrival seeds."""
    out_dir = Path(args.out_dir) / "main_sweep"
    fracs = [float(x) for x in args.audit_fracs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    all_runs = []
    for seed in seeds:
        for frac in fracs:
            tag = f"audit{int(round(frac * 100)):02d}pct_rps{args.rps:.2f}_s{seed}"
            res = await run_single(
                base_url=args.base_url,
                model=args.model,
                rps=args.rps,
                audit_frac=frac,
                arrival_seed=seed,
                audit_mode="uniform",
                audit_logprobs_k=args.audit_logprobs_k,
                num_sessions=args.num_sessions,
                num_turns=args.num_turns,
                num_streaming_sessions=args.num_streaming_sessions,
                init_tokens=args.init_tokens,
                turn_tokens=args.turn_tokens,
                output_tokens=args.output_tokens,
                out_dir=out_dir,
                run_name=tag,
            )
            all_runs.append(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sweep_manifest.json").open("w") as f:
        json.dump({"rps": args.rps, "audit_fracs": fracs, "seeds": seeds, "runs": all_runs}, f, indent=2)


async def run_hunts(args: argparse.Namespace) -> None:
    """Run the three follow-up hunts: early-vs-late, serialization-vs-queue, and load boundary."""
    base_out = Path(args.out_dir) / "hunts"
    seeds = [int(x) for x in args.seeds.split(",")]
    which = set(args.which.split(","))

    common = dict(
        base_url=args.base_url,
        model=args.model,
        num_sessions=args.num_sessions,
        num_turns=args.num_turns,
        num_streaming_sessions=args.num_streaming_sessions,
        init_tokens=args.init_tokens,
        turn_tokens=args.turn_tokens,
        output_tokens=args.output_tokens,
    )

    if "early_late" in which:
        out_dir = base_out / "hunt1_early_vs_late"
        runs = []
        configs = [
            ("baseline_0", "uniform", 0),
            ("early_24", "early", 24),
            ("early_36", "early", 36),
            ("late_5", "late", 5),
            ("late_10", "late", 10),
        ]
        for seed in seeds:
            for label, mode, cnt in configs:
                runs.append(
                    await run_single(
                        **common,
                        rps=args.rps_75pct,
                        audit_frac=0.0,
                        arrival_seed=seed,
                        audit_mode=mode,
                        audit_count=cnt,
                        audit_logprobs_k=args.audit_logprobs_k,
                        out_dir=out_dir,
                        run_name=f"{label}_s{seed}",
                    )
                )
        with (out_dir / "manifest.json").open("w") as f:
            json.dump({"rps": args.rps_75pct, "runs": runs}, f, indent=2)

    if "serialization" in which:
        out_dir = base_out / "hunt2_serialization"
        runs = []
        for seed in seeds:
            runs.append(
                await run_single(
                    **common,
                    rps=args.rps_75pct,
                    audit_frac=0.0,
                    arrival_seed=seed,
                    audit_mode="uniform",
                    audit_action="logprobs",
                    audit_logprobs_k=1,
                    out_dir=out_dir,
                    run_name=f"baseline_0pct_s{seed}",
                )
            )
            # 5% cache-miss-only control: identical audited turns with token 0 mutated,
            # prompt_logprobs unset -> isolates pure prefix-cache bypass from lm_head/log_softmax/serialization.
            runs.append(
                await run_single(
                    **common,
                    rps=args.rps_75pct,
                    audit_frac=0.05,
                    arrival_seed=seed,
                    audit_mode="uniform",
                    audit_action="cache_miss",
                    audit_logprobs_k=1,
                    out_dir=out_dir,
                    run_name=f"cache_miss_5pct_s{seed}",
                )
            )
            for k in (0, 1, 5):
                runs.append(
                    await run_single(
                        **common,
                        rps=args.rps_75pct,
                        audit_frac=0.05,
                        arrival_seed=seed,
                        audit_mode="uniform",
                        audit_action="logprobs",
                        audit_logprobs_k=k,
                        out_dir=out_dir,
                        run_name=f"audit5pct_k{k}_s{seed}",
                    )
                )
        with (out_dir / "manifest.json").open("w") as f:
            json.dump({"rps": args.rps_75pct, "runs": runs}, f, indent=2)

    if "load_boundary" in which:
        out_dir = base_out / "hunt3_load_boundary"
        cap_rps = args.rps_75pct / 0.75
        load_fracs = [float(x) for x in args.load_fracs.split(",")]
        runs = []
        for lf in load_fracs:
            rps = round(cap_rps * lf, 3)
            for seed in seeds:
                for af in (0.0, 0.05):
                    runs.append(
                        await run_single(
                            **common,
                            rps=rps,
                            audit_frac=af,
                            arrival_seed=seed,
                            audit_mode="uniform",
                            audit_logprobs_k=args.audit_logprobs_k,
                            out_dir=out_dir,
                            run_name=f"load{int(round(lf*100)):02d}pct_audit{int(round(af*100)):02d}pct_s{seed}",
                        )
                    )
        with (out_dir / "manifest.json").open("w") as f:
            json.dump({"capacity_rps": cap_rps, "load_fracs": load_fracs, "runs": runs}, f, indent=2)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--num-sessions", type=int, default=24)
    p.add_argument("--num-turns", type=int, default=32)
    p.add_argument("--num-streaming-sessions", type=int, default=16)
    p.add_argument("--init-tokens", type=int, default=4096)
    p.add_argument("--turn-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--audit-logprobs-k", type=int, default=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM v0.34.0 prompt_logprobs audit spillover benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Run a single condition")
    add_common_args(p_single)
    p_single.add_argument("--rps", type=float, required=True)
    p_single.add_argument("--audit-frac", type=float, default=0.0)
    p_single.add_argument("--audit-mode", choices=["uniform", "early", "late"], default="uniform")
    p_single.add_argument("--audit-action", choices=["logprobs", "cache_miss"], default="logprobs")
    p_single.add_argument("--audit-count", type=int, default=None)
    p_single.add_argument("--seed", type=int, default=101)
    p_single.add_argument("--audit-seed", type=int, default=None)
    p_single.add_argument("--run-name", default=None)

    p_cal = sub.add_parser("calibrate", help="Measure 100% baseline capacity at 0% auditing")
    add_common_args(p_cal)
    p_cal.add_argument("--rps-grid", default="4,8,12,16,20,24,28,32,36,40")
    p_cal.add_argument("--slo-p95-ms", type=float, default=500.0)
    p_cal.add_argument("--seed", type=int, default=101)

    p_sweep = sub.add_parser("sweep", help="Main 0/1/5/10% audit sweep x 3 arrival seeds")
    add_common_args(p_sweep)
    p_sweep.add_argument("--rps", type=float, required=True, help="Target RPS (75% of baseline capacity)")
    p_sweep.add_argument("--audit-fracs", default="0.0,0.01,0.05,0.10")
    p_sweep.add_argument("--seeds", default="101,202,303")

    p_hunts = sub.add_parser("hunts", help="Run the 3 hunt ablations")
    add_common_args(p_hunts)
    p_hunts.add_argument("--rps-75pct", type=float, required=True)
    p_hunts.add_argument("--seeds", default="101,202,303")
    p_hunts.add_argument("--load-fracs", default="0.35,0.50,0.75,0.90")
    p_hunts.add_argument("--which", default="early_late,serialization,load_boundary")

    args = parser.parse_args()
    if args.cmd == "single":
        name = args.run_name or f"single_{args.audit_mode}_{args.audit_action}_a{int(round(args.audit_frac*100))}_rps{args.rps:.2f}_s{args.seed}"
        asyncio.run(
            run_single(
                base_url=args.base_url,
                model=args.model,
                rps=args.rps,
                audit_frac=args.audit_frac,
                arrival_seed=args.seed,
                audit_mode=args.audit_mode,
                audit_action=args.audit_action,
                audit_count=args.audit_count,
                audit_logprobs_k=args.audit_logprobs_k,
                audit_seed=args.audit_seed,
                num_sessions=args.num_sessions,
                num_turns=args.num_turns,
                num_streaming_sessions=args.num_streaming_sessions,
                init_tokens=args.init_tokens,
                turn_tokens=args.turn_tokens,
                output_tokens=args.output_tokens,
                out_dir=Path(args.out_dir) / "single",
                run_name=name,
            )
        )
    elif args.cmd == "calibrate":
        asyncio.run(run_calibrate(args))
    elif args.cmd == "sweep":
        asyncio.run(run_sweep(args))
    elif args.cmd == "hunts":
        asyncio.run(run_hunts(args))


if __name__ == "__main__":
    main()
