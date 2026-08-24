#!/usr/bin/env python3
"""Summarize MNN OpenCL MNN_GPU_TIME_PROFILE output.

Examples:
    tools/script/opencl_profile_summary.py gpu_profile.log
    tools/script/opencl_profile_summary.py optimized.log --compare baseline.log
    ./build_profile/llm_demo model_quant/config.json prompt.txt 50 2>&1 | \
        tools/script/opencl_profile_summary.py -

The names in this report are MNN GPU-profile event labels. They are not
necessarily the underlying OpenCL source kernel names.
"""

import argparse
import re
import sys
from collections import defaultdict


EVENT_PATTERN = re.compile(r"^\s*kernel time =\s*(\d+(?:\.\d+)?)\s+us\s+(.+?)\s*$")
BATCH_PATTERN = re.compile(r"^\s*total kernel time =")


def parse_profile(lines):
    totals = defaultdict(lambda: [0, 0.0])
    batch_count = 0
    event_count = 0

    for line in lines:
        if BATCH_PATTERN.match(line):
            batch_count += 1

        match = EVENT_PATTERN.match(line)
        if match is None:
            continue

        duration_us = float(match.group(1))
        event_name = match.group(2)
        totals[event_name][0] += 1
        totals[event_name][1] += duration_us
        event_count += 1

    return totals, batch_count, event_count


def format_us(value):
    if value >= 1000.0:
        return f"{value / 1000.0:,.3f} ms"
    return f"{value:,.0f} us"


def print_summary(totals, batch_count, event_count, top):
    total_us = sum(total for _, total in totals.values())
    if event_count == 0:
        print("No MNN GPU_PROFILE event lines found.", file=sys.stderr)
        return 1

    rows = sorted(totals.items(), key=lambda item: (-item[1][1], item[0]))
    if top > 0:
        rows = rows[:top]

    name_width = max(len("Event"), *(len(name) for name, _ in rows))
    print("OpenCL GPU_PROFILE summary")
    print(f"Profile batches: {batch_count}")
    print(f"Events:          {event_count}")
    print(f"Event time:      {format_us(total_us)}")
    print()
    print(f"{'Event':<{name_width}}  {'Calls':>7}  {'Total':>12}  {'Average':>12}  {'Share':>7}")
    print(f"{'-' * name_width}  {'-' * 7}  {'-' * 12}  {'-' * 12}  {'-' * 7}")
    for name, (calls, duration_us) in rows:
        share = 100.0 * duration_us / total_us if total_us > 0.0 else 0.0
        print(
            f"{name:<{name_width}}  {calls:7d}  {format_us(duration_us):>12}  "
            f"{format_us(duration_us / calls):>12}  {share:6.2f}%"
        )
    return 0


def format_delta_us(value):
    if value >= 1000.0 or value <= -1000.0:
        return f"{value / 1000.0:+,.3f} ms"
    return f"{value:+,.0f} us"


def format_delta_percent(baseline, candidate):
    if baseline <= 0.0:
        return "n/a"
    return f"{100.0 * (candidate - baseline) / baseline:+.2f}%"


def print_comparison(baseline_profile, candidate_profile, baseline_path, candidate_path, top):
    baseline_totals, baseline_batches, baseline_events = baseline_profile
    candidate_totals, candidate_batches, candidate_events = candidate_profile
    if baseline_events == 0:
        print(f"No MNN GPU_PROFILE event lines found in baseline: {baseline_path}", file=sys.stderr)
        return 1
    if candidate_events == 0:
        print(f"No MNN GPU_PROFILE event lines found in candidate: {candidate_path}", file=sys.stderr)
        return 1

    rows = []
    for event_name in set(baseline_totals) | set(candidate_totals):
        baseline_calls, baseline_total = baseline_totals.get(event_name, (0, 0.0))
        candidate_calls, candidate_total = candidate_totals.get(event_name, (0, 0.0))
        baseline_average = baseline_total / baseline_calls if baseline_calls else None
        candidate_average = candidate_total / candidate_calls if candidate_calls else None
        if baseline_average is None or candidate_average is None:
            sort_key = max(baseline_average or 0.0, candidate_average or 0.0)
        else:
            sort_key = abs(candidate_average - baseline_average)
        rows.append((event_name, baseline_calls, candidate_calls, baseline_average, candidate_average, sort_key))

    rows.sort(key=lambda row: (-row[5], row[0]))
    if top > 0:
        rows = rows[:top]

    baseline_total_us = sum(total for _, total in baseline_totals.values())
    candidate_total_us = sum(total for _, total in candidate_totals.values())
    name_width = max(len("Event"), *(len(row[0]) for row in rows))
    calls_width = max(len("Calls"), *(len(f"{row[1]}/{row[2]}") for row in rows))
    print("OpenCL GPU_PROFILE comparison")
    print(f"Baseline:          {baseline_path}")
    print(f"Candidate:         {candidate_path}")
    print(f"Baseline batches:  {baseline_batches}")
    print(f"Candidate batches: {candidate_batches}")
    print(f"Baseline events:   {baseline_events}")
    print(f"Candidate events:  {candidate_events}")
    print(f"Baseline time:     {format_us(baseline_total_us)}")
    print(f"Candidate time:    {format_us(candidate_total_us)}")
    print(f"Total change:      {format_delta_us(candidate_total_us - baseline_total_us)} "
          f"({format_delta_percent(baseline_total_us, candidate_total_us)})")
    print("Per-event deltas compare average time per call; total time also reflects call-count changes.")
    print()
    print(
        f"{'Event':<{name_width}}  {'Calls':>{calls_width}}  {'Baseline avg':>14}  "
        f"{'Candidate avg':>14}  {'Delta':>12}  {'Delta %':>9}"
    )
    print(
        f"{'-' * name_width}  {'-' * calls_width}  {'-' * 14}  {'-' * 14}  "
        f"{'-' * 12}  {'-' * 9}"
    )
    for event_name, baseline_calls, candidate_calls, baseline_average, candidate_average, _ in rows:
        calls = f"{baseline_calls}/{candidate_calls}"
        if baseline_average is None or candidate_average is None:
            baseline_text = format_us(baseline_average) if baseline_average is not None else "n/a"
            candidate_text = format_us(candidate_average) if candidate_average is not None else "n/a"
            delta_text = "n/a"
            delta_percent = "n/a"
        else:
            baseline_text = format_us(baseline_average)
            candidate_text = format_us(candidate_average)
            delta_text = format_delta_us(candidate_average - baseline_average)
            delta_percent = format_delta_percent(baseline_average, candidate_average)
        print(
            f"{event_name:<{name_width}}  {calls:>{calls_width}}  {baseline_text:>14}  "
            f"{candidate_text:>14}  {delta_text:>12}  {delta_percent:>9}"
        )
    return 0


def read_profile(path):
    if path == "-":
        return parse_profile(sys.stdin)
    with open(path, encoding="utf-8", errors="replace") as log_file:
        return parse_profile(log_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="GPU_PROFILE output file, or - to read stdin")
    parser.add_argument("--compare", metavar="BASELINE_LOG", help="Compare LOG against a baseline profile log")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only the top N events by total time, or absolute average change with --compare; 0 shows all",
    )
    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must be non-negative")
    if args.log == "-" and args.compare == "-":
        parser.error("LOG and --compare cannot both read from stdin")

    try:
        candidate_profile = read_profile(args.log)
        if args.compare is not None:
            baseline_profile = read_profile(args.compare)
            return print_comparison(baseline_profile, candidate_profile, args.compare, args.log, args.top)
        return print_summary(*candidate_profile, args.top)
    except OSError as error:
        print(f"Cannot read {error.filename or args.log}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
