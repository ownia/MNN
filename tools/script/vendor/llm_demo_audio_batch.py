#!/usr/bin/env python3
"""Run llm_demo over WAV files in a directory and summarize performance.

Example:
    python3 tools/script/llm_demo_audio_batch.py --audio-dir ./audio -- \
        ./build/llm_demo /path/to/config.json {audio}

    python3 tools/script/llm_demo_audio_batch.py --read-summary llm_demo_audio_summary.json --max-rtf 1.0
    python3 tools/script/llm_demo_audio_batch.py --read-summary llm_demo_audio_summary.json --max-rtf-file

The command must contain ``{audio}``, which is replaced with the absolute path
of each WAV file. Batch runs always store unfiltered measurements in
``llm_demo_audio_summary.json`` by default; use ``--summary-json`` to choose
another path. With ``--read-summary``, ``--max-rtf`` selects the audio files
used for every average by their Audio E2E RTF.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
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
METRIC_LABELS = {
    "prefill_speed": ("Prefill speed", "tok/s"),
    "decode_speed": ("Decode speed", "tok/s"),
    "audio_rtf": ("Audio RTF", ""),
    "audio_e2e_rtf": ("Audio E2E RTF", ""),
}
RTF_METRICS = ("audio_rtf", "audio_e2e_rtf")
RTF_FILTER_METRIC = "audio_e2e_rtf"
RTF_FILTER_LABEL = "Audio E2E RTF"
SUMMARY_FORMAT_VERSION = 1
DEFAULT_SUMMARY_JSON = Path("llm_demo_audio_summary.json")


def non_negative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
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


def metric_value(record, name):
    value = record.get("metrics", {}).get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def build_summary(records, max_rtf):
    metrics_summary = {}
    valid_records = [record for record in records if record.get("status") == "ok"]
    failed_records = [record for record in records if record.get("status") == "failed"]
    included_records = valid_records
    if max_rtf is not None:
        included_records = []
        for record in valid_records:
            e2e_rtf = metric_value(record, RTF_FILTER_METRIC)
            if e2e_rtf is not None and e2e_rtf <= max_rtf:
                included_records.append(record)

    for name in METRIC_LABELS:
        samples = []
        for record in valid_records:
            value = metric_value(record, name)
            if value is not None:
                samples.append((record, value))

        included_samples = []
        for record in included_records:
            value = metric_value(record, name)
            if value is not None:
                included_samples.append((record, value))

        values = [value for _, value in included_samples]
        maximum = None
        if name in RTF_METRICS and samples:
            max_record, max_value = max(samples, key=lambda sample: sample[1])
            maximum = {
                "value": max_value,
                "audio_file": max_record["audio_file"],
                "relative_path": max_record["relative_path"],
            }

        metrics_summary[name] = {
            "available_count": len(samples),
            "count": len(values),
            "excluded": len(samples) - len(values),
            "average": statistics.mean(values) if values else None,
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "maximum": maximum,
        }

    return {
        "total_audio_files": len(records),
        "valid_audio_files": len(valid_records),
        "failed_audio_files": len(failed_records),
        "max_rtf_filter": max_rtf,
        "rtf_filter_counts": {
            "included": len(included_records),
            "excluded": len(valid_records) - len(included_records),
            "total": len(valid_records),
        },
        "metrics": metrics_summary,
    }


def find_audio_files(audio_dir, recursive):
    paths = audio_dir.rglob("*") if recursive else audio_dir.iterdir()
    return sorted(
        (path for path in paths if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: str(path).lower(),
    )


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
        return output, time.perf_counter() - started_at, "timed out"
    except OSError as error:
        return "", time.perf_counter() - started_at, str(error)

    error = None
    if result.returncode != 0:
        error = f"exited with status {result.returncode}"
    return result.stdout, time.perf_counter() - started_at, error


def format_metrics(metrics):
    values = []
    for name, (label, unit) in METRIC_LABELS.items():
        value = metrics.get(name)
        if value is None:
            continue
        suffix = f" {unit}" if unit else ""
        precision = ".2f" if unit else ".3f"
        values.append(f"{label}={value:{precision}}{suffix}")
    return ", ".join(values)


def print_summary(summary):
    print("\nllm_demo audio batch summary")
    print(f"Valid audio files: {summary['valid_audio_files']}")
    print(f"Failed audio files: {summary['failed_audio_files']}")
    if summary["max_rtf_filter"] is not None:
        counts = summary["rtf_filter_counts"]
        print(f"All averages use {RTF_FILTER_LABEL} <= {summary['max_rtf_filter']:.3f}.")
        print(f"RTF filter: included={counts['included']}, excluded={counts['excluded']}, total={counts['total']}")
    print()

    for name, (label, unit) in METRIC_LABELS.items():
        metric_summary = summary["metrics"][name]
        suffix = f" {unit}" if unit else ""
        count_note = (
            f"(included={metric_summary['count']}, excluded={metric_summary['excluded']}, "
            f"total={metric_summary['available_count']})"
            if summary["max_rtf_filter"] is not None
            else f"(n={metric_summary['count']})"
        )
        if metric_summary["average"] is None:
            print(f"Average {label:<14} no values {count_note}")
            continue

        print(
            f"Average {label:<14} {metric_summary['average']:9.3f}{suffix}"
            f"  stddev {metric_summary['stddev']:.3f}  {count_note}"
        )

    for name in RTF_METRICS:
        label, _ = METRIC_LABELS[name]
        maximum = summary["metrics"][name]["maximum"]
        if maximum is not None:
            print(f"Maximum {label:<14} {maximum['value']:.3f}  ({maximum['relative_path']})")


def write_summary(path, report):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as summary_file:
            json.dump(report, summary_file, indent=2)
            summary_file.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Cannot write summary JSON {path}: {error}") from error


def load_summary(path):
    try:
        with path.open(encoding="utf-8") as summary_file:
            report = json.load(summary_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read summary JSON {path}: {error}") from error

    if not isinstance(report, dict) or report.get("format_version") != SUMMARY_FORMAT_VERSION:
        raise RuntimeError(f"{path} is not a supported llm_demo audio summary")
    records = report.get("audio_files")
    if not isinstance(records, list):
        raise RuntimeError(f"{path} does not contain audio file records")
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"{path} contains an invalid audio file record")
        if not isinstance(record.get("audio_file"), str) or not isinstance(record.get("relative_path"), str):
            raise RuntimeError(f"{path} contains an audio file record without a path")
        if record.get("status") not in ("ok", "failed") or not isinstance(record.get("metrics"), dict):
            raise RuntimeError(f"{path} contains an invalid audio file result")
    return report


def print_max_rtf_file(summary):
    maximum = summary["metrics"]["audio_rtf"]["maximum"]
    if maximum is None:
        print("No valid Audio RTF measurement is available.", file=sys.stderr)
        return 1
    print(maximum["audio_file"])
    return 0


def make_report(audio_dir, command, recursive, timeout, summary, records):
    return {
        "format_version": SUMMARY_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_dir": str(audio_dir),
        "command": command,
        "options": {
            "recursive": recursive,
            "timeout_seconds": timeout,
        },
        "summary": summary,
        "audio_files": records,
    }


def read_summary_mode(args):
    summary_path = args.read_summary.expanduser()
    try:
        report = load_summary(summary_path)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    summary = build_summary(report["audio_files"], args.max_rtf)
    if args.max_rtf_file:
        return print_max_rtf_file(summary)

    print(f"Summary JSON: {summary_path.resolve()}")
    if isinstance(report.get("audio_dir"), str):
        print(f"Audio directory: {report['audio_dir']}")
    print_summary(summary)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio-dir", type=Path, help="Directory containing WAV files")
    input_group.add_argument("--read-summary", type=Path, help="Read a JSON report from an earlier batch run")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=f"Write the batch report here (default: {DEFAULT_SUMMARY_JSON})",
    )
    parser.add_argument("--recursive", action="store_true", help="Also process WAV files in subdirectories")
    parser.add_argument(
        "--max-rtf",
        "--rtf-max",
        dest="max_rtf",
        type=non_negative_float,
        help="With --read-summary, use only files at or below this Audio E2E RTF for every average",
    )
    parser.add_argument(
        "--timeout",
        type=non_negative_float,
        default=0.0,
        help="Per-file timeout in seconds; 0 disables the timeout (default: 0)",
    )
    parser.add_argument("--show-output", action="store_true", help="Print complete llm_demo output for every file")
    parser.add_argument(
        "--max-rtf-file",
        action="store_true",
        help="Print the absolute path of the WAV with the highest Audio RTF and exit",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run; place it after --")
    args = parser.parse_args()

    if args.read_summary is not None:
        if args.command:
            parser.error("a command cannot be used with --read-summary")
        return read_summary_mode(args)

    if args.max_rtf is not None:
        parser.error("--max-rtf is only available with --read-summary")

    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide the llm_demo command after --")
    if not any("{audio}" in argument for argument in command):
        parser.error("the command must contain an {audio} placeholder")

    audio_dir = args.audio_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        parser.error(f"audio directory does not exist: {audio_dir}")
    audio_files = find_audio_files(audio_dir, args.recursive)
    if not audio_files:
        parser.error(f"no WAV files found in: {audio_dir}")

    timeout = args.timeout or None
    records = []

    for index, audio_file in enumerate(audio_files, start=1):
        relative_path = audio_file.relative_to(audio_dir)
        audio_path = str(audio_file.resolve())
        audio_command = [argument.replace("{audio}", audio_path) for argument in command]
        print(f"[{index}/{len(audio_files)}] {relative_path}...", end=" ", flush=True)
        output, wall_time, error = run_once(audio_command, timeout)
        if args.show_output:
            print()
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                print()

        metrics = parse_metrics(output)
        record = {
            "index": index,
            "audio_file": audio_path,
            "relative_path": str(relative_path),
            "wall_time_seconds": wall_time,
            "metrics": metrics,
            "status": "ok",
            "error": None,
        }
        if error is not None or "audio_rtf" not in metrics:
            reason = error or "missing audio RTF"
            record["status"] = "failed"
            record["error"] = reason
            records.append(record)
            print(f"FAILED ({reason}; wall={wall_time:.2f}s)")
            if not args.show_output and output:
                print(f"Last output lines:\n{output[-2000:]}", file=sys.stderr)
            continue

        print(f"{format_metrics(metrics)}; wall={wall_time:.2f}s")
        records.append(record)

    summary = build_summary(records, None)
    report = make_report(audio_dir, command, args.recursive, args.timeout, summary, records)
    summary_path = args.summary_json.expanduser()
    try:
        write_summary(summary_path, report)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if args.max_rtf_file:
        max_status = print_max_rtf_file(summary)
        if max_status != 0:
            return max_status
        return 1 if summary["failed_audio_files"] else 0

    print_summary(summary)
    print(f"Summary JSON: {summary_path.resolve()}")
    if not summary["valid_audio_files"]:
        print("No valid audio measurements; cannot calculate averages.", file=sys.stderr)
        return 1
    return 1 if summary["failed_audio_files"] else 0


if __name__ == "__main__":
    sys.exit(main())