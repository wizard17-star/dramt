"""Wait for the sharded sweep to finish, aggregate it, then run the full chain.

Launched in the background so the GPU is never idle between the sweep ending
and the results chain starting.

Safety: the chain is only started if the sweep actually COMPLETED. If the
shard processes exit with fewer trials than the grid requires (a crash, an
OOM, a manual kill), this stops and says so rather than selecting a "best"
config from a partial grid and silently producing results on top of it.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

POLL_SECONDS = 120
RUNS = Path("runs/sweep")


def expected_trials() -> int:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    n_cfg = len(list(itertools.product(
        cfg["windowing"]["T_sweep"],
        cfg["model"]["d_model_sweep"],
        cfg["model"]["n_layers_sweep"],
        cfg["loss"]["lambda1_sweep"],
        cfg["loss"]["lambda2_sweep"],
        cfg["training"]["lr_sweep"],
    )))
    return n_cfg * cfg["splits"]["n_folds"]


def completed_trials() -> int:
    n = 0
    for p in list(RUNS.glob("trials.csv")) + list(RUNS.glob("trials_shard*.csv")):
        with open(p, encoding="utf-8") as f:
            n += max(sum(1 for _ in f) - 1, 0)      # minus header
    return n


def sweep_processes_running() -> int:
    """Count live `src.sweep` python processes (Windows, via WMI through PS).

    Note: on Windows each worker shows up TWICE -- `.venv\\Scripts\\python.exe`
    is a launcher stub that spawns the base interpreter as a child, and both
    carry the same command line. So 4 shards report 8 processes. Only the
    zero/non-zero distinction is used here, and the pair dies together, so the
    doubling is harmless; do not read this number as a worker count.
    """
    ps = (
        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*src.sweep*' } | "
        "Measure-Object).Count"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip() or 0)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[after_sweep] process check failed ({exc}); assuming still running",
              flush=True)
        return 1


def main() -> None:
    want = expected_trials()
    print(f"[after_sweep] waiting for {want} sweep trials", flush=True)

    last = -1
    while True:
        have = completed_trials()
        alive = sweep_processes_running()
        if have != last:
            print(f"[after_sweep] {have}/{want} trials, {alive} worker(s) alive",
                  flush=True)
            last = have
        if have >= want:
            print("[after_sweep] sweep complete", flush=True)
            break
        if alive == 0:
            print(f"[after_sweep] ABORT: all workers exited with only {have}/{want} "
                  f"trials done. Not selecting a config from a partial grid. "
                  f"Inspect runs/sweep/sweep_shard*.log, then relaunch the shards "
                  f"(they resume) before running scripts/gpu_chain.py.", flush=True)
            sys.exit(1)
        time.sleep(POLL_SECONDS)

    # aggregate: no --no-aggregate, every trial already done -> writes best.yaml
    print("[after_sweep] aggregating -> results.csv / best.yaml", flush=True)
    agg = subprocess.run([sys.executable, "-m", "src.sweep", "--folds", "all",
                          "--suffix", ""], capture_output=True, text=True)
    print("\n".join(agg.stdout.splitlines()[-5:] + agg.stderr.splitlines()[-5:]),
          flush=True)
    if agg.returncode != 0 or not (RUNS / "best.yaml").exists():
        sys.exit("[after_sweep] aggregation failed; chain not started")

    # 4 concurrent training jobs: a single job leaves this GPU ~20% utilised
    # (launch-bound at this model size). Verified safe -- every job writes to
    # its own runs/<name> directory, and the shared GARCH/HAR vol cache is
    # published atomically, tested with 3 runs racing on the same cache files.
    parallel = os.environ.get("DRAMT_CHAIN_PARALLEL", "4")
    print(f"[after_sweep] starting gpu_chain (parallel={parallel})", flush=True)
    chain = subprocess.run(
        [sys.executable, "-m", "scripts.gpu_chain", "--parallel", parallel], text=True)
    sys.exit(chain.returncode)


if __name__ == "__main__":
    main()
