#!/usr/bin/env python3
"""Run an llm_demo command repeatedly and summarize its performance output.

Example:
    tools/script/llm_demo_repeat.py --runs 10 --warmup 1 -- \
        ./build/llm_demo ../MNN-ownia/model_quant/config.json ../MNN-test/output.wav

The command must emit llm_demo's standard summary, including ``prefill speed``,
``decode speed``, and ``audio RTF``. Each measurement run must emit all three.
"""

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time


NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
METRIC_PATTERNS = {
    "prefill_speed": re.compile(r"^\s*prefill speed\s*=\s*" + NUMBER + r"\s+tok/s\s*$", re.MULTILINE),
    "decode_speed": re.compile(r"^\s*decode speed\s*=\s*" + NUMBER + r"\s+tok/s\s*$", re.MULTILINE),
    "audio_rtf": re.compile(r"^\s*audio RTF\s*=\s*" + NUMBER + r"\s*$", re.MULTILINE),
    "audio_e2e_rtf": re.compile(r"^\s*audio E2E RTF\s*=\s*" + NUMBER + r"\s*$", re.MULTILINE),
}
REQUIRED_METRICS = ("prefill_speed", "decode_speed", "audio_rtf")
METRIC_LABELS = {
    "prefill_speed": ("Prefill speed", "tok/s"),
    "decode_speed": ("Decode speed", "tok/s"),
    "audio_rtf": ("Audio RTF", ""),
    "audio_e2e_rtf": ("Audio E2E RTF", ""),
}


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def non_negative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def parse_metrics(output):
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(output)
        if not matches:
            continue
        value = float(matches[-1])
        if math.isfinite(value):
            metrics[name] = value
    return metrics


def run_once(command, timeout):
    started_at = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return None, output, time.perf_counter() - started_at, "timed out"
    except OSError as error:
        return None, "", time.perf_counter() - started_at, str(error)

    error = None
    if result.returncode != 0:
        error = f"exited with status {result.returncode}"
    return result.returncode, result.stdout, time.perf_counter() - started_at, error


def save_log(log_dir, phase, index, output):
    if log_dir is None:
        return
    path = os.path.join(log_dir, f"{phase}-{index:02d}.log")
    with open(path, "w", encoding="utf-8") as log_file:
        log_file.write(output)


def format_run(metrics):
    values = []
    for name in REQUIRED_METRICS:
        label, unit = METRIC_LABELS[name]
        value = metrics.get(name)
        if value is None:
            values.append(f"{label}=missing")
        elif unit:
            values.append(f"{label}={value:.2f} {unit}")
        else:
            values.append(f"{label}={value:.3f}")
    return ", ".join(values)


def print_summary(measurements, failures):
    print("\nllm_demo repeat summary")
    print(f"Valid measurement runs: {len(measurements)}")
    print(f"Failed runs:            {failures}")
    print()

    for name, (label, unit) in METRIC_LABELS.items():
        values = [metrics[name] for metrics in measurements if name in metrics]
        if not values:
            continue
        average = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        suffix = f" {unit}" if unit else ""
        print(f"Average {label:<14} {average:9.3f}{suffix}  stddev {deviation:.3f}  (n={len(values)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=positive_int, default=5, help="Measurement runs (default: 5)")
    parser.add_argument("--warmup", type=non_negative_int, default=1, help="Warmup runs excluded from averages")
    parser.add_argument(
        "--timeout",
        type=non_negative_int,
        default=0,
        help="Per-run timeout in seconds; 0 disables the timeout (default: 0)",
    )
    parser.add_argument("--show-output", action="store_true", help="Print complete llm_demo output for every run")
    parser.add_argument("--log-dir", help="Write each run's combined stdout and stderr to this directory")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run; place it after --")
    args = parser.parse_args()

    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide the llm_demo command after --")

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    timeout = args.timeout or None
    measurements = []
    failures = 0
    total_runs = args.warmup + args.runs

    for run_number in range(1, total_runs + 1):
        is_warmup = run_number <= args.warmup
        phase = "warmup" if is_warmup else "measure"
        phase_index = run_number if is_warmup else run_number - args.warmup
        print(f"{phase} {phase_index}/{args.warmup if is_warmup else args.runs}...", end=" ", flush=True)
        _, output, wall_time, error = run_once(command, timeout)
        save_log(args.log_dir, phase, phase_index, output)
        if args.show_output:
            print()
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                print()

        metrics = parse_metrics(output)
        missing_metrics = [name for name in REQUIRED_METRICS if name not in metrics]
        if error is not None or missing_metrics:
            failures += 1
            reason = error or f"missing {', '.join(missing_metrics)}"
            print(f"FAILED ({reason}; wall={wall_time:.2f}s)")
            if not args.show_output and output:
                print(f"Last output lines:\n{output[-2000:]}", file=sys.stderr)
            continue

        print(f"{format_run(metrics)}; wall={wall_time:.2f}s")
        if not is_warmup:
            measurements.append(metrics)

    if not measurements:
        print("No valid measurement runs; cannot calculate averages.", file=sys.stderr)
        return 1

    print_summary(measurements, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())