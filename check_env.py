"""M0 environment check: reports Python/torch versions and GPU/CUDA availability."""
from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("No CUDA GPU detected -> training will run on CPU.")


if __name__ == "__main__":
    main()
