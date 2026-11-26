#!/usr/bin/env python3
"""
Generate the Money Chart and Hunt diagnostic plots from benchmark results.

Money Chart:
  Audit fraction (0%, 1%, 5%, 10%) vs p95 TTFT (ms) for requests that were
  NEVER audited (separate streaming cohort), with each point labeled by:
    - Reported prefix-cache hit ratio (%)
    - Newly computed prefill tokens (K / M)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def fmt_tokens(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{n:.0f}"


def plot_money_chart(results_dir: Path, out_dir: Path) -> None:
    manifest_path = results_dir / "main_sweep" / "sweep_manifest.json"
    if not manifest_path.exists():
        print(f"[skip] {manifest_path} not found")
        return
    manifest = load_json(manifest_path)
    runs = manifest["runs"]
    rps = manifest["rps"]

    by_frac: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_frac[float(r["config"]["audit_frac"])].append(r)

    fracs = sorted(by_frac.keys())
    x_pct = [f * 100.0 for f in fracs]

    p95_means, p95_stds, p95_per_seed = [], [], []
    hit_means, true_hit_means, prefill_means = [], [], []

    for f in fracs:
        group = by_frac[f]
        p95s = [g["streaming_unscored"]["ttft_ms"]["p95"] for g in group]
        hits = [
            g["server_metrics"]["reported_cache_hit_ratio"] * 100.0
            for g in group
            if g["server_metrics"]["reported_cache_hit_ratio"] is not None
        ]
        thits = [
            g["server_metrics"]["true_effective_cache_hit_ratio"] * 100.0
            for g in group
            if g["server_metrics"]["true_effective_cache_hit_ratio"] is not None
        ]
        prefills = [g["server_metrics"]["newly_computed_prefill_tokens"] or 0.0 for g in group]
        p95_per_seed.append(p95s)
        p95_means.append(float(np.mean(p95s)))
        p95_stds.append(float(np.std(p95s, ddof=1)) if len(p95s) > 1 else 0.0)
        hit_means.append(float(np.mean(hits)) if hits else float("nan"))
        true_hit_means.append(float(np.mean(thits)) if thits else float("nan"))
        prefill_means.append(float(np.mean(prefills)))

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    # Plot individual seeds faintly + mean curve with error bars.
    for xi, ys in zip(x_pct, p95_per_seed):
        ax.scatter([xi] * len(ys), ys, color="#1f77b4", alpha=0.45, s=36, zorder=3)

    ax.errorbar(
        x_pct,
        p95_means,
        yerr=p95_stds,
        fmt="-o",
        color="#0b4f8a",
        linewidth=2.4,
        markersize=8,
        capsize=5,
        zorder=4,
        label="Unscored cohort p95 TTFT (mean ± SD across seeds)",
    )

    # Hypothesis +20% reference line relative to 0% baseline.
    base_p95 = p95_means[0]
    thresh_p95 = base_p95 * 1.20
    ax.axhline(
        thresh_p95,
        color="#c0392b",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
        label=f"+20% hypothesis threshold ({thresh_p95:.1f} ms)",
    )

    y_span = max(p95_means) - min(p95_means)
    y_pad = max(y_span * 0.14, base_p95 * 0.08, 12.0)

    for idx, (xi, yi, hit_r, true_r, ptok) in enumerate(
        zip(x_pct, p95_means, hit_means, true_hit_means, prefill_means)
    ):
        rel = ((yi / base_p95) - 1.0) * 100.0 if base_p95 > 0 else 0.0
        sign = "+" if rel >= 0 else ""
        label_txt = (
            f"{yi:.0f} ms ({sign}{rel:.0f}%)\n"
            f"Reported hit: {hit_r:.1f}%\n"
            f"New prefill: {fmt_tokens(ptok)} tok"
        )
        y_offset = y_pad if idx % 2 == 0 else y_pad * 1.15
        va = "bottom"
        ax.annotate(
            label_txt,
            xy=(xi, yi),
            xytext=(xi, yi + y_offset),
            ha="center",
            va=va,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#adb5bd", lw=0.9),
            arrowprops=dict(arrowstyle="->", color="#6c757d", lw=0.9),
            zorder=5,
        )

    ax.set_xlabel("Audited Turn Fraction (% of all turns with prompt_logprobs)", fontsize=11)
    ax.set_ylabel("Unscored Cohort p95 TTFT (ms)", fontsize=11)
    ax.set_title(
        f"Audit Spillover in vLLM v0.34.0 (Qwen2.5-3B-Instruct BF16, 75% Load = {rps:.2f} req/s)\n"
        "Reported Prefix-Cache Hit Rate Stays Flat While GPU Prefill Work & Unscored Tail Latency Surge",
        fontsize=11.5,
        pad=14,
    )
    ax.set_xticks(x_pct)
    ax.set_xticklabels([f"{x:.0f}%" for x in x_pct])
    ax.set_xlim(-0.8, max(x_pct) + 1.2)
    ax.set_ylim(0, max(p95_means) + y_pad * 3.1)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "money_chart.png", dpi=200)
    fig.savefig(out_dir / "money_chart.pdf")
    plt.close(fig)
    print(f"[saved] {out_dir / 'money_chart.png'}")

    # Print concise hypothesis check table.
    print("\n=== Money Chart Summary ===")
    print(f"{'Audit%':>7} | {'p95 TTFT (ms)':>14} | {'vs 0%':>8} | {'Rep Hit%':>9} | {'True Hit%':>10} | {'New Prefill':>12}")
    print("-" * 74)
    for xi, yi, sd, hr, tr, pt in zip(x_pct, p95_means, p95_stds, hit_means, true_hit_means, prefill_means):
        rel = ((yi / base_p95) - 1.0) * 100.0 if base_p95 > 0 else 0.0
        print(f"{xi:6.1f}% | {yi:7.1f} ± {sd:4.1f} | {rel:+7.1f}% | {hr:8.2f}% | {tr:9.2f}% | {pt:12,.0f}")


def plot_hunt1(results_dir: Path, out_dir: Path) -> None:
    p = results_dir / "hunts" / "hunt1_early_vs_late" / "manifest.json"
    if not p.exists():
        return
    runs = load_json(p)["runs"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        label = r["run_name"].rsplit("_s", 1)[0]
        groups[label].append(r)

    order = [k for k in ["baseline_0", "early_24", "early_36", "late_5", "late_10"] if k in groups]
    means = [float(np.mean([g["streaming_unscored"]["ttft_ms"]["p95"] for g in groups[k]])) for k in order]
    stds = [
        float(np.std([g["streaming_unscored"]["ttft_ms"]["p95"] for g in groups[k]], ddof=1))
        if len(groups[k]) > 1
        else 0.0
        for k in order
    ]
    prefills = [float(np.mean([g["server_metrics"]["newly_computed_prefill_tokens"] or 0.0 for g in groups[k]])) for k in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    colors = ["#7f8c8d", "#2980b9", "#1f618d", "#d35400", "#a04000"][: len(order)]
    bars = ax.bar(order, means, yerr=stds, capsize=5, color=colors, alpha=0.9)
    for b, m, pt in zip(bars, means, prefills):
        ax.text(
            b.get_x() + b.get_width() / 2,
            m + max(means) * 0.03,
            f"{m:.0f} ms\n({fmt_tokens(pt)})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylabel("Unscored Cohort p95 TTFT (ms)")
    ax.set_title("Hunt 1: Early-Turn vs Late-Turn Audits (at 75% Load)")
    ax.set_ylim(0, max(means) * 1.28)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "hunt1_early_vs_late.png", dpi=200)
    plt.close(fig)
    print(f"[saved] {out_dir / 'hunt1_early_vs_late.png'}")


def plot_hunt2(results_dir: Path, out_dir: Path) -> None:
    p = results_dir / "hunts" / "hunt2_serialization" / "manifest.json"
    if not p.exists():
        return
    runs = load_json(p)["runs"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        label = r["run_name"].rsplit("_s", 1)[0]
        groups[label].append(r)

    order = [k for k in ["baseline_0pct", "audit5pct_k0", "audit5pct_k1", "audit5pct_k5"] if k in groups]
    client_p95 = [float(np.mean([g["streaming_unscored"]["ttft_ms"]["p95"] for g in groups[k]])) for k in order]
    srv_ttft_p95 = [
        float(np.mean([(g["server_metrics"]["server_ttft_s"]["p95"] or 0.0) * 1000.0 for g in groups[k]]))
        for k in order
    ]
    srv_q_p95 = [
        float(np.mean([(g["server_metrics"]["server_queue_s"]["p95"] or 0.0) * 1000.0 for g in groups[k]]))
        for k in order
    ]
    payload_mb = [
        float(np.mean([(g["nonstreaming_audited"]["response_bytes"]["mean"] or 0.0) / (1024 * 1024) for g in groups[k]]))
        for k in order
    ]

    x = np.arange(len(order))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=180)
    ax.bar(x - w, client_p95, w, label="Client Unscored p95 TTFT (ms)", color="#0b4f8a")
    ax.bar(x, srv_ttft_p95, w, label="Server Prometheus p95 TTFT (ms)", color="#27ae60")
    ax.bar(x + w, srv_q_p95, w, label="Server Prometheus p95 Queue (ms)", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(audit payload: {mb:.2f} MB)" for k, mb in zip(order, payload_mb)], fontsize=8.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Hunt 2: Logprob Payload Serialization Overhead vs Server Queue/TTFT")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "hunt2_serialization_gap.png", dpi=200)
    plt.close(fig)
    print(f"[saved] {out_dir / 'hunt2_serialization_gap.png'}")


def plot_hunt3(results_dir: Path, out_dir: Path) -> None:
    p = results_dir / "hunts" / "hunt3_load_boundary" / "manifest.json"
    if not p.exists():
        return
    data = load_json(p)
    runs = data["runs"]
    cap_rps = data["capacity_rps"]

    by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in runs:
        lf = int(round((r["config"]["rps"] / cap_rps) * 100.0))
        af = int(round(r["config"]["audit_frac"] * 100.0))
        by_key[(lf, af)].append(r["streaming_unscored"]["ttft_ms"]["p95"])

    loads = sorted({lf for lf, _ in by_key.keys()})
    y0 = [float(np.mean(by_key[(lf, 0)])) for lf in loads]
    y5 = [float(np.mean(by_key[(lf, 5)])) for lf in loads]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax.plot(loads, y0, "-o", color="#2c3e50", lw=2.2, label="0% Auditing (Baseline)")
    ax.plot(loads, y5, "-s", color="#c0392b", lw=2.2, label="5% Auditing (prompt_logprobs)")
    for lf, v0, v5 in zip(loads, y0, y5):
        pct = ((v5 / v0) - 1.0) * 100.0 if v0 > 0 else 0.0
        ax.annotate(f"{pct:+.0f}%", xy=(lf, v5), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Load (% of Measured Baseline Capacity)")
    ax.set_ylabel("Unscored Cohort p95 TTFT (ms)")
    ax.set_title(f"Hunt 3: Load Boundary for 5% Audit Spillover (100% Capacity = {cap_rps:.2f} req/s)")
    ax.set_xticks(loads)
    ax.set_xticklabels([f"{l}%" for l in loads])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "hunt3_load_boundary.png", dpi=200)
    plt.close(fig)
    print(f"[saved] {out_dir / 'hunt3_load_boundary.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark money chart and hunt diagnostics")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_money_chart(results_dir, out_dir)
    plot_hunt1(results_dir, out_dir)
    plot_hunt2(results_dir, out_dir)
    plot_hunt3(results_dir, out_dir)


if __name__ == "__main__":
    main()
