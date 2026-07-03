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


def _case_files() -> int:
    from pathlib import Path
    return len(list(Path("data/raw/gdelt").glob("*/*_daily.csv")))


def main() -> None:
    common = ["--only-gdelt", "--primary-kw"]
    # Case phase with retries: windows that exhaust their retry budget are not
    # cached, so re-running the phase retries ONLY the failed windows (cached
    # ones are skipped instantly). Loop until a pass adds nothing new.
    for attempt in range(8):
        before = _case_files()
        run([*common, "--gdelt-phase", "case"])
        after = _case_files()
        print(f"[gdelt] case pass {attempt}: {before} -> {after} files", flush=True)
        if after == before:
            break
    run([*common, "--gdelt-phase", "monthly"])   # then full monthly history
    print("[gdelt] driver complete", flush=True)


if __name__ == "__main__":
    main()
