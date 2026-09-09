"""Preprocessing entry point: reads raw Data/, writes processed/cache/.

Run this ONCE before training:
    python scripts/preprocess.py --config configs/default.yaml

Or in Colab after mounting drive:
    !python scripts/preprocess.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data.preprocess import run_preprocessing


def main():
    parser = argparse.ArgumentParser(description="ERA5 preprocessing pipeline")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override config values, e.g. --set domain.lat_min=21.0")
    parser.add_argument("--out", default=None, help="Override output cache directory path")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    out_path = run_preprocessing(cfg, out_path=args.out, verbose=True)
    print(f"\n[preprocess] Done. Cache written to: {out_path}")
    print("[preprocess] Next steps:")
    print("  1. python scripts/colab/train_colab.py  (or use Colab GPU)")
    print("  2. python scripts/optimize_threshold.py")
    print("  3. python scripts/evaluate.py")
    print("  4. python -m src.api.run_server")


if __name__ == "__main__":
    main()

