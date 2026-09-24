"""Slice a 30-minute time window (default: hour 5:00 -> 5:30) out of each full
dataset and write it to the `samples/` folder for testing.

Frame mapping (data is 30 fps, fid starts at 1):
    t seconds  -> fid = t_seconds * 30 + 1
    hour 5     -> fid = 5 * 3600 * 30 + 1  = 540001
    hour 5:30  -> fid = 5.5 * 3600 * 30    = 594000  (exclusive end)
i.e. 54000 frames == 30 minutes. Original fid values are preserved so that query
intervals still refer to the same absolute time.

Usage:
    python sample_30min.py                 # hour 5 -> 5:30
    python sample_30min.py --start 5 --end 5.5
"""

import argparse
import json
import os

import pandas as pd

FPS = 30
DATA_DIR = "/home/cw/DataSet/Dataset/youtube/10h/interp_pd"
SAMPLE_DIR = os.path.join(DATA_DIR, "samples")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "query_configs", "dataset.json")


def hour_to_fid(hour: float) -> int:
    """Start fid (inclusive) corresponding to a decimal hour."""
    return int(round(hour * 3600 * FPS)) + 1


def main(start_hour: float, end_hour: float) -> None:
    fid_start = hour_to_fid(start_hour)
    fid_end = hour_to_fid(end_hour) - 1  # inclusive end
    n_frames = fid_end - fid_start + 1

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    with open(CONFIG_PATH, "r") as f:
        data_m = json.load(f)

    print(f"window {start_hour}h->{end_hour}h : fid [{fid_start}, {fid_end}] "
          f"({n_frames} frames = {n_frames / FPS / 60:.1f} min)\n")

    for name, cfg in data_m.items():
        ext = cfg.get("extension", "parquet")
        src = os.path.join(DATA_DIR, f"{name}10h.{ext}")
        dst = os.path.join(SAMPLE_DIR, f"{name}10h.{ext}")
        if not os.path.exists(src):
            print(f"[skip] {name}: file not found at {src}")
            continue

        # predicate pushdown keeps only the requested fid range in memory
        sub = pd.read_parquet(src, filters=[("fid", ">=", fid_start),
                                            ("fid", "<=", fid_end)])
        sub.to_parquet(dst, index=False)
        print(f"{name:12s} {len(sub):>9d} rows -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=5.0, help="start hour")
    parser.add_argument("--end", type=float, default=5.5, help="end hour")
    args = parser.parse_args()
    main(args.start, args.end)
