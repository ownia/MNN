#!/usr/bin/env python3
"""Summarize MNN OpenCL MNN_GPU_TIME_PROFILE output.

Examples:
    tools/script/opencl_profile_summary.py gpu_profile.log
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="GPU_PROFILE output file, or - to read stdin")
    parser.add_argument("--top", type=int, default=0, help="Show only the top N events by total time; 0 shows all")
    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must be non-negative")

    if args.log == "-":
        return print_summary(*parse_profile(sys.stdin), args.top)

    try:
        with open(args.log, encoding="utf-8", errors="replace") as log_file:
            return print_summary(*parse_profile(log_file), args.top)
    except OSError as error:
        print(f"Cannot read {args.log}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
