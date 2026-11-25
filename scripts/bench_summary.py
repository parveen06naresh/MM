"""Run the C++ benchmark over several seeds and keep the median of each metric.
    python scripts/bench_summary.py > results/bench_summary.json
"""
import json
import statistics
import subprocess
import sys

SEEDS = [1, 2, 3, 4, 5]
runs = [json.loads(subprocess.check_output(["build/bench_book", "5000000", str(s)])) for s in SEEDS]
med = lambda xs: statistics.median(xs)  # noqa: E731
out = {
    "seeds": SEEDS,
    "messages_per_run": runs[0]["messages"],
    "throughput_msgs_per_s_median": round(med([r["throughput_msgs_per_s"] for r in runs])),
    "throughput_msgs_per_s_range": [min(r["throughput_msgs_per_s"] for r in runs), max(r["throughput_msgs_per_s"] for r in runs)],
    "timer_overhead_p50_ns": med([r["timer_overhead_p50_ns"] for r in runs]),
}
for op in ("all", "add", "cancel", "market", "reduce"):
    out[op] = {k: med([r[op][k] for r in runs]) for k in ("p50_ns", "p90_ns", "p99_ns", "p999_ns")}
json.dump(out, sys.stdout, indent=2)
print()
