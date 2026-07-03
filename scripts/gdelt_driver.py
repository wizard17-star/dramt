"""GDELT prioritized pull: case-study daily windows first (all tickers, primary
keyword only), then monthly history. Detached; log -> runs/gdelt_pull.log."""
from __future__ import annotations

import subprocess
import sys
import time


def run(args: list[str]) -> None:
    t0 = time.time()
    print(f"[gdelt] START {' '.join(args)}", flush=True)
    proc = subprocess.run([sys.executable, "download_data.py", *args])
    print(f"[gdelt] rc={proc.returncode} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    common = ["--only-gdelt", "--primary-kw"]
    run([*common, "--gdelt-phase", "case"])      # thesis case study first
    run([*common, "--gdelt-phase", "monthly"])   # then full monthly history
    print("[gdelt] driver complete", flush=True)


if __name__ == "__main__":
    main()
