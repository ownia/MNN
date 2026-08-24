#!/usr/bin/env python3
"""Summarize OpenCL tuning records stored in an MNN cache file.

Examples:
    .venv/bin/python tools/script/opencl_cache_summary.py tmp_xxx/mnn_cachefile.bin
    .venv/bin/python tools/script/opencl_cache_summary.py --device Mali-G720 --top 20 mnn_cachefile.bin
    .venv/bin/python tools/script/opencl_cache_summary.py --format json mnn_cachefile.bin

Generic Autotuning records contain the selected global/local work sizes and
their measured OpenCL event time in microseconds. Xgemm GemmInfo records store
the selected shape and parameter tile, but the cache schema does not store the
measured event time for them.
"""

import argparse
import json
import struct
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OPENCL_SCHEMA_PATH = REPOSITORY_ROOT / "source" / "backend" / "opencl" / "schema"
GEMM_SHAPE_FIELDS = (
    "M",
    "N",
    "K",
    "layout_precision",
    "batch",
    "bias_group",
    "precision_type",
)
GEMM_PARAM_FIELDS = (
    "KWG",
    "KWI",
    "MDIMA",
    "MDIMC",
    "MWG",
    "NDIMB",
    "NDIMC",
    "NWG",
    "SA",
    "SB",
    "STRM",
    "STRN",
    "VWM",
    "VWN",
)


def load_cache_binding():
    sys.path.insert(0, str(OPENCL_SCHEMA_PATH))
    try:
        import flatbuffers  # noqa: F401
        from CLCache import Cache
    except ModuleNotFoundError as error:
        if error.name == "flatbuffers":
            raise RuntimeError(
                "Missing Python package 'flatbuffers'. Run with the repository virtual environment "
                "(.venv/bin/python) or install it with: python3 -m pip install flatbuffers"
            ) from error
        raise RuntimeError(f"Cannot load OpenCL cache bindings: {error}") from error
    return Cache


def text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def vector(table, value_at, length):
    return [value_at(index) for index in range(length())]


def named_values(names, values):
    return {name: values[index] if index < len(values) else None for index, name in enumerate(names)}


def read_cache(path):
    cache_binding = load_cache_binding()
    try:
        data = bytearray(path.read_bytes())
    except OSError as error:
        raise RuntimeError(f"Cannot read {path}: {error}") from error
    if len(data) < 4:
        raise RuntimeError(f"{path} is too small to be an MNN OpenCL cache")

    try:
        cache = cache_binding.Cache.GetRootAs(data, 0)
        backends = []
        for backend_index in range(cache.BackendsLength()):
            backend = cache.Backends(backend_index)
            autotuning = []
            for tuning_index in range(backend.TuningsLength()):
                tuning = backend.Tunings(tuning_index)
                autotuning.append(
                    {
                        "key": text(tuning.Key()),
                        "program": text(tuning.Name()),
                        "global_size": vector(tuning, tuning.GloablSize, tuning.GloablSizeLength),
                        "local_size": vector(tuning, tuning.LocalSize, tuning.LocalSizeLength),
                        "time_cost_us": tuning.TimeCost(),
                        "md5": text(tuning.Md5()),
                    }
                )

            gemm = []
            for gemm_index in range(backend.GemmLength()):
                entry = backend.Gemm(gemm_index)
                shape = vector(entry, entry.GemmSize, entry.GemmSizeLength)
                params = vector(entry, entry.ParamInfo, entry.ParamInfoLength)
                gemm.append(
                    {
                        "shape": named_values(GEMM_SHAPE_FIELDS, shape),
                        "params": named_values(GEMM_PARAM_FIELDS, params),
                        "time_cost_us": None,
                        "md5": text(entry.Md5()),
                    }
                )

            backends.append(
                {
                    "device": text(backend.DeviceName()),
                    "autotuning": autotuning,
                    "gemm": gemm,
                }
            )
    except (IndexError, TypeError, ValueError, struct.error) as error:
        raise RuntimeError(f"{path} is not a readable MNN OpenCL cache: {error}") from error

    return {
        "cache": str(path),
        "size_bytes": len(data),
        "tuned_op_records": cache.TunedLength(),
        "backends": backends,
    }


def size_text(values):
    return "x".join(str(value) for value in values) if values else "-"


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_text_summary(summary, device_filter, top):
    selected_backends = [
        backend for backend in summary["backends"] if not device_filter or device_filter.lower() in backend["device"].lower()
    ]
    print("OpenCL MNN cache summary")
    print(f"Cache:            {summary['cache']}")
    print(f"Cache size:       {summary['size_bytes']} bytes")
    print(f"Backend records:  {len(summary['backends'])}")
    print(f"Tuned op records: {summary['tuned_op_records']}")

    if not selected_backends:
        print(f"\nNo backend device matches: {device_filter}")
        return 1

    for backend in selected_backends:
        autotuning = sorted(backend["autotuning"], key=lambda record: record["time_cost_us"], reverse=True)
        gemm = sorted(
            backend["gemm"],
            key=lambda record: (
                record["shape"]["M"] or 0,
                record["shape"]["N"] or 0,
                record["shape"]["K"] or 0,
            ),
        )
        if top:
            autotuning = autotuning[:top]
            gemm = gemm[:top]

        print(f"\nDevice: {backend['device'] or '<unnamed>'}")
        print(f"Autotuning records: {len(backend['autotuning'])}")
        print(f"Xgemm records:      {len(backend['gemm'])}")

        if autotuning:
            print("\nGeneric autotuning event times")
            rows = [
                (
                    record["key"],
                    record["program"],
                    size_text(record["global_size"]),
                    size_text(record["local_size"]),
                    str(record["time_cost_us"]),
                )
                for record in autotuning
            ]
            print_table(("Key", "Program", "Global size", "Local size", "Time (us)"), rows)

        if gemm:
            print("\nXgemm cached shapes and tiles")
            rows = []
            for record in gemm:
                shape = record["shape"]
                params = record["params"]
                rows.append(
                    (
                        "x".join(str(shape[name]) for name in ("M", "N", "K")),
                        str(shape["batch"]),
                        str(shape["layout_precision"]),
                        str(shape["bias_group"]),
                        size_text([params[name] for name in GEMM_PARAM_FIELDS]),
                        "not stored",
                    )
                )
            print_table(("M x N x K", "Batch", "Layout", "Bias/group", "Xgemm parameters", "Time"), rows)
            print("Xgemm GemmInfo stores the selected tile, not its measured event time.")

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", type=Path, help="MNN OpenCL mnn_cachefile.bin")
    parser.add_argument("--device", help="Show only backends whose device name contains this text")
    parser.add_argument("--top", type=int, default=0, help="Show only the top N generic timings and first N Xgemm shapes")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format (default: text)")
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be non-negative")

    try:
        summary = read_cache(args.cache)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if args.format == "json":
        if args.device:
            summary["backends"] = [
                backend for backend in summary["backends"] if args.device.lower() in backend["device"].lower()
            ]
        print(json.dumps(summary, indent=2))
        return 0 if summary["backends"] else 1
    return print_text_summary(summary, args.device, args.top)


if __name__ == "__main__":
    sys.exit(main())
