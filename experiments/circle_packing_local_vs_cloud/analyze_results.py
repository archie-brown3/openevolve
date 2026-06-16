#!/usr/bin/env python3
"""
Analyze OpenEvolve results for the circle packing local-model experiment.

Reads:
  - <output>/checkpoints/checkpoint_N/best_program_info.json  (fitness curve)
  - <output>/evolution_trace.jsonl                             (mutation quality)
  - <output>/logs/*.log                                        (parse failure count)

Writes a new section to experiments/EXPERIMENTS.md.

Usage:
  python3 analyze_results.py \
      --local  experiments/circle_packing_local_vs_cloud/results/local \
      --output experiments/EXPERIMENTS.md \
      --wall-time 1800 \
      --model  "Qwen2.5-0.5B" \
      --iterations 30
"""

import argparse
import difflib
import json
import sys
from datetime import datetime
from pathlib import Path


TARGET_SUM_RADII = 2.635
INITIAL_SUM_RADII = 0.959  # initial program baseline


# ── Data loading ──────────────────────────────────────────────────────────────

def read_checkpoints(output_dir: Path) -> list[dict]:
    """Return sorted list of {checkpoint, metrics, generation, code} dicts."""
    cp_root = output_dir / "checkpoints"
    if not cp_root.exists():
        return []

    results = []
    for cp_dir in sorted(
        cp_root.iterdir(),
        key=lambda d: int(d.name.split("_")[-1]) if d.name.startswith("checkpoint_") and d.name.split("_")[-1].isdigit() else 0,
    ):
        info_file = cp_dir / "best_program_info.json"
        if not info_file.exists():
            continue
        with open(info_file) as f:
            info = json.load(f)

        code = ""
        prog_file = cp_dir / "best_program.py"
        if prog_file.exists():
            code = prog_file.read_text()

        cp_num = int(cp_dir.name.split("_")[-1]) if cp_dir.name.split("_")[-1].isdigit() else 0
        results.append(
            {
                "checkpoint": cp_num,
                "metrics": info.get("metrics", {}),
                "generation": info.get("generation", 0),
                "code": code,
            }
        )
    return results


def read_evolution_trace(output_dir: Path) -> list[dict]:
    """Return list of trace entries from evolution_trace.jsonl."""
    trace_file = output_dir / "evolution_trace.jsonl"
    if not trace_file.exists():
        return []

    entries = []
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def read_logs(output_dir: Path) -> dict:
    """Count invalid-parse log lines and return stats dict."""
    logs_dir = output_dir / "logs"
    if not logs_dir.exists():
        return {"invalid_parse": 0}

    invalid_parse = 0
    for log_file in sorted(logs_dir.glob("*.log")):
        with open(log_file) as f:
            for line in f:
                if "No valid code found in response" in line or "No valid diffs found in response" in line:
                    invalid_parse += 1

    return {"invalid_parse": invalid_parse}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_diversity(checkpoints: list[dict]) -> list[dict]:
    """Character-level similarity between consecutive best programs."""
    scores = []
    for i in range(1, len(checkpoints)):
        a = checkpoints[i - 1].get("code", "")
        b = checkpoints[i].get("code", "")
        if a and b:
            sim = difflib.SequenceMatcher(None, a, b).ratio() * 100
            scores.append(
                {
                    "from_cp": checkpoints[i - 1]["checkpoint"],
                    "to_cp": checkpoints[i]["checkpoint"],
                    "similarity_pct": sim,
                    "diversity_pct": 100 - sim,
                }
            )
    return scores


def detect_plateau(checkpoints: list[dict], threshold: float = 0.001, window: int = 2) -> int | None:
    """Return checkpoint number where fitness stabilises, or None."""
    if len(checkpoints) < window + 1:
        return None
    scores = [(cp["checkpoint"], cp["metrics"].get("combined_score", 0.0)) for cp in checkpoints]
    for i in range(window, len(scores)):
        deltas = [abs(scores[j][1] - scores[j - 1][1]) for j in range(i - window + 1, i + 1)]
        if all(d < threshold for d in deltas):
            return scores[i - window][0]
    return None


def verdict(final_ratio: float, acceptance_rate: float, improvement_rate: float) -> str:
    """viable | borderline | useless"""
    initial_ratio = INITIAL_SUM_RADII / TARGET_SUM_RADII
    if final_ratio > 0.70 and acceptance_rate >= 0.50:
        return "viable"
    if final_ratio > initial_ratio * 1.5 and acceptance_rate >= 0.25:
        return "borderline"
    return "useless"


VERDICT_COPY = {
    "viable": (
        "The local Qwen2.5-0.5B model shows meaningful fitness progression and produces "
        "parseable, improving mutations at an acceptable rate. Worth using as a low-cost "
        "provider for initial exploration phases before switching to a cloud model for exploitation."
    ),
    "borderline": (
        "The local model produces some valid mutations and limited fitness improvement, but with "
        "high noise and a low acceptance rate. Potentially useful for diversity-seeding on very "
        "simple sub-problems, but not reliable as a primary evolution driver."
    ),
    "useless": (
        "The local model fails to make meaningful progress: too many invalid outputs and/or no "
        "fitness improvement beyond the baseline. Not viable for autonomous evolution on this problem. "
        "Consider using it only as a low-weight secondary ensemble member under a capable cloud model."
    ),
}


# ── Report generation ─────────────────────────────────────────────────────────

def build_report(
    model_name: str,
    iterations: int,
    wall_time: int,
    checkpoints: list[dict],
    traces: list[dict],
    log_stats: dict,
) -> str:
    final_cp = checkpoints[-1] if checkpoints else {}
    final_m = final_cp.get("metrics", {})
    final_sum = final_m.get("sum_radii", 0.0)
    final_ratio = final_m.get("target_ratio", 0.0)

    avg_iter_s = wall_time / iterations if iterations else 0
    mins, secs = divmod(wall_time, 60)

    total_valid = len(traces)
    improved = sum(
        1
        for t in traces
        if (t.get("improvement_delta") or {}).get("combined_score", 0.0) > 0
    )
    improvement_rate = improved / total_valid if total_valid else 0.0

    invalid_parses = log_stats.get("invalid_parse", 0)
    acceptance_rate = (iterations - invalid_parses) / iterations if iterations else 0.0

    diversity_scores = compute_diversity(checkpoints)
    avg_div = (
        sum(d["diversity_pct"] for d in diversity_scores) / len(diversity_scores)
        if diversity_scores
        else 0.0
    )

    plateau_at = detect_plateau(checkpoints)
    v = verdict(final_ratio, acceptance_rate, improvement_rate)
    date_str = datetime.now().strftime("%Y-%m-%d")

    lines: list[str] = []

    lines += [
        f"\n## Experiment: {model_name} (local llama.cpp) — Circle Packing n=26",
        f"",
        f"**Date:** {date_str}  ",
        f"**Model:** {model_name} via llama.cpp at `localhost:8080`  ",
        f"**Config:** `experiments/circle_packing_local_vs_cloud/config_local.yaml`  ",
        f"**Iterations:** {iterations}  ",
        f"**Wall time:** {mins}m {secs}s  ",
        f"**Avg time / iteration:** {avg_iter_s:.1f}s  ",
        f"",
        f"### Fitness Curve",
        f"",
        f"Target: 2.635 (AlphaEvolve). Baseline: ~0.959 (ratio ≈ 0.364).",
        f"",
        f"| Checkpoint | Best sum\\_radii | Target ratio | Generation |",
        f"|:----------:|:--------------:|:------------:|:---------:|",
    ]

    for cp in checkpoints:
        m = cp.get("metrics", {})
        lines.append(
            f"| {cp['checkpoint']:>10} "
            f"| {m.get('sum_radii', 0.0):>14.4f} "
            f"| {m.get('target_ratio', 0.0):>12.4f} "
            f"| {cp.get('generation', 0):>9} |"
        )

    lines += [
        f"",
        f"**Final best:** {final_sum:.4f} sum\\_radii "
        f"({final_ratio * 100:.2f}% of target 2.635)",
        f"",
        f"### Mutation Acceptance & Quality",
        f"",
        f"| Metric | Value |",
        f"|:-------|------:|",
        f"| Total iterations scheduled | {iterations} |",
        f"| Invalid LLM outputs (parse failed) | {invalid_parses} ({(1 - acceptance_rate) * 100:.1f}%) |",
        f"| Valid programs produced (in trace) | {total_valid} |",
        f"| Mutations improving parent score | {improved} / {total_valid} ({improvement_rate * 100:.1f}%) |",
        f"",
        f"### Convergence",
        f"",
    ]

    if plateau_at is not None:
        lines.append(
            f"Fitness plateau detected at checkpoint **{plateau_at}** "
            f"(score stable within ±0.001 over two consecutive checkpoints)."
        )
    else:
        lines.append(
            f"No clear plateau within {iterations} iterations — "
            f"the run may still have headroom to improve."
        )

    lines += [
        f"",
        f"### Mutation Diversity",
        f"",
        f"Code similarity between consecutive best programs (lower similarity = more exploration):",
        f"",
        f"| Span | Similarity | Diversity |",
        f"|:-----|----------:|----------:|",
    ]

    for d in diversity_scores:
        lines.append(
            f"| cp{d['from_cp']} → cp{d['to_cp']} "
            f"| {d['similarity_pct']:.1f}% "
            f"| {d['diversity_pct']:.1f}% |"
        )

    if diversity_scores:
        lines.append(
            f"| **Average** "
            f"| **{100 - avg_div:.1f}%** "
            f"| **{avg_div:.1f}%** |"
        )

    lines += [
        f"",
        f"### Throughput",
        f"",
        f"- Total wall time: {mins}m {secs}s for {iterations} iterations",
        f"- Average time per iteration: {avg_iter_s:.1f}s",
        f"- Token counts not directly available; use `llama-server --metrics` "
        f"and `GET /metrics` for per-request token data.",
        f"",
        f"### Verdict: **{v.upper()}**",
        f"",
        VERDICT_COPY[v],
        f"",
        f"---",
    ]

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze circle packing experiment results")
    p.add_argument("--local", required=True, help="Path to OpenEvolve output dir for local run")
    p.add_argument("--output", required=True, help="Path to EXPERIMENTS.md to append results to")
    p.add_argument("--wall-time", type=int, default=0, help="Total wall time in seconds")
    p.add_argument("--model", default="Qwen2.5-0.5B", help="Model display name")
    p.add_argument("--iterations", type=int, default=30, help="Total iterations configured")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.local)

    print(f"Reading checkpoints from:    {output_dir / 'checkpoints'}")
    print(f"Reading evolution trace from: {output_dir / 'evolution_trace.jsonl'}")
    print(f"Reading logs from:            {output_dir / 'logs'}")

    checkpoints = read_checkpoints(output_dir)
    traces = read_evolution_trace(output_dir)
    log_stats = read_logs(output_dir)

    if not checkpoints:
        print("WARNING: No checkpoint data found — the run may not have completed.")

    print(f"\nCheckpoints found: {len(checkpoints)}")
    print(f"Trace entries:     {len(traces)}")
    print(f"Invalid parses:    {log_stats['invalid_parse']}")

    report = build_report(
        model_name=args.model,
        iterations=args.iterations,
        wall_time=args.wall_time,
        checkpoints=checkpoints,
        traces=traces,
        log_stats=log_stats,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(report + "\n")

    print(f"\nReport appended to: {out}")

    if checkpoints:
        final_m = checkpoints[-1].get("metrics", {})
        v = verdict(
            final_m.get("target_ratio", 0.0),
            (args.iterations - log_stats["invalid_parse"]) / args.iterations,
            len([t for t in traces if (t.get("improvement_delta") or {}).get("combined_score", 0) > 0]) / max(len(traces), 1),
        )
        print(f"\nVerdict: {v.upper()}")
        print(f"Final sum_radii: {final_m.get('sum_radii', 0):.4f}  "
              f"({final_m.get('target_ratio', 0) * 100:.2f}% of target)")


if __name__ == "__main__":
    main()
