"""Reproduce the baseline on validation without changing the supplied trainer/scorer."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_bytes():
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure(args):
    # Run the unmodified scorer in this fresh process; query OS lifetime peak RSS.
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    sys.argv = [str(ROOT / "evaluate.py"), "--checkpoint", str(checkpoint),
                "--device", "cpu", "--precision", "fp32", "--threads", "4",
                "--split", args.split, "--output", str(output)]
    runpy.run_path(str(ROOT / "evaluate.py"), run_name="__main__")
    resources = {"peak_process_rss_bytes": peak_rss_bytes(),
                 "checkpoint_bytes": checkpoint.stat().st_size,
                 "checkpoint_sha256": digest(checkpoint),
                 "scope": "Fresh scorer process, including imports, loading and evaluation; checkpoint size is not total inference asset size."}
    output.with_suffix(".resources.json").write_text(json.dumps(resources, indent=2) + "\n")


def reproduce(args):
    import torch
    import numpy
    import tokenizers
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protected = ["common.py", "evaluate.py", "model.py", "configs/baseline.json"]
    source_files = protected + ["train.py", "student.py", "tests/test_contract.py", "baseline_pipeline.py"]
    manifest = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "environment": {"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "processor": platform.processor(),
                                "logical_cpus": os.cpu_count(), "torch": str(torch.__version__),
                                "numpy": numpy.__version__, "tokenizers": tokenizers.__version__},
                "source_sha256": {name: digest(ROOT / name) for name in source_files},
                "selection_split": "validation", "parent_checkpoints": [], "commands": []}
    manifest_path = output / "pipeline.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(label, arguments):
        command = [sys.executable, "-u", *map(str, arguments)]
        row = {"label": label, "argv": command}
        manifest["commands"].append(row)
        save()
        started = time.perf_counter()
        with (output / f"{label}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode, wall_seconds=time.perf_counter() - started)
        save()
        print(f"{label}: exit={result.returncode}, seconds={row['wall_seconds']:.1f}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {output / (label + '.log')}")

    save()
    try:
        run("contract", ["-m", "unittest", "discover", "-s", "tests", "-v"])
        for label, steps, interval in (("smoke", 10, 0), ("baseline", 1200, 300)):
            run(label + "_train", ["train.py", "--implementation", "model", "--device", "cpu",
                "--precision", "fp32", "--threads", "4", "--seed", "17", "--steps", steps,
                "--eval-every", interval, "--run-dir", output / label])
            run(label + "_reload", ["baseline_pipeline.py", "measure", "--checkpoint",
                output / label / "checkpoint.pt", "--output", output / label / "validation_cpu_fp32.json"])
            trained = json.loads((output / label / "metrics.json").read_text())
            reloaded = json.loads((output / label / "validation_cpu_fp32.json").read_text())
            if abs(trained["validation"]["bpb"] - reloaded["bpb"]) > 1e-6:
                raise RuntimeError(f"{label}: reloaded validation BPB differs")
        if any(digest(ROOT / name) != manifest["source_sha256"][name] for name in protected):
            raise RuntimeError("Protected baseline/benchmark files changed")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("run")
    baseline.add_argument("--run-dir", type=Path, required=True)
    measurement = commands.add_parser("measure")
    measurement.add_argument("--checkpoint", type=Path, required=True)
    measurement.add_argument("--output", type=Path, required=True)
    measurement.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    (reproduce if args.command == "run" else measure)(args)
