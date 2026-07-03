"""M6 driver: run all sentiment-independent baselines on all folds, then the
DRAM-T hyperparameter sweep on fold 0. Designed to run detached; progress goes
to stdout (redirected to runs/m6_driver.log).

sentiment_lstm and the final DRAM-T all-fold run are intentionally EXCLUDED:
they wait for the real GDELT sentiment features.
"""
from __future__ import annotations

import subprocess
import sys
import time

BASELINES = [
    "arima", "garch", "garch_midas", "dcc_garch",
    "lstm", "gru", "cnn_bilstm", "cnn_bilstm_attn", "transformer",
]


def run(args: list[str]) -> None:
    t0 = time.time()
    print(f"[m6] START {' '.join(args)}", flush=True)
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    tail = "\n".join(proc.stdout.splitlines()[-3:] + proc.stderr.splitlines()[-3:])
    status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    print(f"[m6] {status} ({time.time() - t0:.0f}s) {' '.join(args)}\n{tail}", flush=True)


def main() -> None:
    for model in BASELINES:
        run(["src.baselines", "--model", model, "--suffix", "_nosent"])
    run(["src.sweep", "--suffix", "_nosent"])
    print("[m6] driver complete", flush=True)


if __name__ == "__main__":
    main()
