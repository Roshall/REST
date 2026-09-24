"""Compute the top-9 most frequent `cls` (category) values for every dataset
and fill them into `experiment/query_configs/dataset.json` as numeric values.

Usage:
    python fill_top9_cat.py            # update dataset.json in place
    python fill_top9_cat.py --top N    # use a different top-N (default 9)
"""

import argparse
import json
import os

import pandas as pd

# Where the interpolated parquet files live.
DATA_DIR = "/home/cw/DataSet/Dataset/youtube/10h/interp_pd"
# Path to the dataset config that we will update.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "query_configs", "dataset.json")


def main(top_n: int) -> None:
    with open(CONFIG_PATH, "r") as f:
        data_m = json.load(f)

    for name, cfg in data_m.items():
        ext = cfg.get("extension", "parquet")
        path = os.path.join(DATA_DIR, f"{name}10h.{ext}")
        if not os.path.exists(path):
            print(f"[skip] {name}: file not found at {path}")
            continue

        df = pd.read_parquet(path, columns=["cls"])
        # Most frequent category values, as plain Python ints (not numpy/str).
        top_cats = [int(v) for v in df["cls"].value_counts().head(top_n).index.tolist()]

        cfg["top9_cat"] = top_cats
        print(f"{name:12s} top{top_n}: {top_cats}")

    with open(CONFIG_PATH, "w") as f:
        json.dump(data_m, f, indent=2)
        f.write("\n")

    print(f"\nUpdated {CONFIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=9, help="number of top categories")
    args = parser.parse_args()
    main(args.top)
