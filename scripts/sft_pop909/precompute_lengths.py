#!/usr/bin/env python
"""Precompute length cache through the existing RT tokenizer path."""

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from get_length import precompute_dataset_lengths, verify_cache  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute old-compatible dataset length cache.")
    parser.add_argument("--data-dir", required=True, help="Directory containing old-compatible NPZ files.")
    parser.add_argument("--patch-h", type=int, default=1)
    parser.add_argument("--patch-w", type=int, default=4)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true", help="Remove existing .lengths_cache.pkl first.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing cache.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    cache_path = os.path.join(data_dir, ".lengths_cache.pkl")
    if args.verify_only:
        ok = verify_cache(data_dir)
        raise SystemExit(0 if ok else 1)

    if args.overwrite and os.path.exists(cache_path):
        os.remove(cache_path)

    precompute_dataset_lengths(
        data_dir=data_dir,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        max_workers=args.workers,
    )
    verify_cache(data_dir)


if __name__ == "__main__":
    main()
