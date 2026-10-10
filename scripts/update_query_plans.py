"""Update experiment/query_configs/query_plans.json from cached density stats.

Regions: for each dataset and each area-fraction scale, place the top-density
regions (time-aggregated points) as a group of ``--regions-per-scale`` boxes in
``region.bboxes[<dataset>]``. If a scale is too large to fit that many
non-overlapping boxes, the remainder is filled by density, allowing overlap.

Intervals: derived from the (dataset-averaged) temporal density CDF. For each
clip length, endpoints are the times where the cumulative activity crosses
``k/K``; the list is padded or cut to exactly ``K`` endpoints, keeping integer
minutes.

Example::

    python scripts/update_query_plans.py --dry-run
    python scripts/update_query_plans.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import density  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATS_DIR = os.path.join(ROOT, "meta", "density")
DEFAULT_PLANS = os.path.join(ROOT, "experiment", "query_configs", "query_plans.json")
DEFAULT_SCALES = "0.02,0.05,0.10,0.20,0.40"
# clip key -> clip length in minutes
CLIP_MINUTES = {"10h": 600, "5h": 300, "3h": 180}


def density_interval_ends(profile, bucket_sec, clip_minutes, k):
    """K integer-minute endpoints where the activity CDF crosses k/K."""
    nb = min(len(profile), max(1, clip_minutes * 60 // bucket_sec))
    c = np.asarray(profile[:nb], dtype=np.float64)
    if c.sum() <= 0:
        return list(range(1, clip_minutes + 1, max(1, clip_minutes // k)))[:k]
    cdf = c.cumsum() / c.sum()
    raw = []
    for i in range(1, k + 1):
        idx = int(np.searchsorted(cdf, i / k, side="left"))
        idx = min(idx, nb - 1)
        raw.append(int(np.ceil((idx + 1) * bucket_sec / 60.0)))
    ends = []
    for v in raw:
        v = max(1, min(v, clip_minutes))
        if ends and v <= ends[-1]:
            v = ends[-1] + 1
        if v > clip_minutes:
            v = clip_minutes
        ends.append(v)
    pts = sorted(set(ends))
    while len(pts) < k:  # pad into the widest gaps
        arr = [0] + pts
        best = None
        for i, lo in enumerate(arr):
            hi = arr[i + 1] if i + 1 < len(arr) else clip_minutes
            if hi - lo >= 2 and (best is None or hi - lo > best[1] - best[0]):
                best = (lo, hi)
        if best is None:
            break
        pts = sorted(set(pts) | {(best[0] + best[1]) // 2})
    return pts[:k]  # cut if too many


def _any_overlap(regions):
    """True if any two ``[xmin,xmax,ymin,ymax]`` boxes geometrically overlap."""
    for i in range(len(regions)):
        ax0, ax1, ay0, ay1 = regions[i]
        for j in range(i + 1, len(regions)):
            bx0, bx1, by0, by1 = regions[j]
            if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                return True
    return False


def aggregate_profile(stats_dir, datasets):
    """Dataset-averaged temporal profile (each dataset normalized to sum 1)."""
    prof = None
    for name in datasets:
        with open(os.path.join(stats_dir, f"{name}10h.stats.json")) as f:
            man = json.load(f)
        arr = density.load_stats(os.path.join(stats_dir, f"{name}10h.stats.npz"))
        p = arr["temporal"].sum(axis=0).astype(np.float64)
        if p.sum() <= 0:
            continue
        p /= p.sum()
        if prof is None:
            prof = np.zeros(len(p))
        if len(p) > len(prof):
            prof = np.pad(prof, (0, len(p) - len(prof)))
        prof[:len(p)] += p
    if prof is None:
        raise SystemExit("no usable temporal stats")
    return prof / len(datasets), int(man["params"]["t_bucket_sec"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    p.add_argument("--plans", default=DEFAULT_PLANS)
    p.add_argument("--out", default=None, help="output path (default: overwrite --plans)")
    p.add_argument("--scales", default=DEFAULT_SCALES,
                   help="comma list of area fractions, small -> large")
    p.add_argument("--regions-per-scale", type=int, default=3)
    p.add_argument("--interval-counts", default=None,
                   help="override e.g. '10h:15,5h:9,3h:9'")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    scales = [float(x) for x in args.scales.split(",")]
    with open(args.plans) as f:
        plans = json.load(f)

    datasets = sorted(fn[: -len("10h.stats.json")] for fn in os.listdir(args.stats_dir)
                      if fn.endswith("10h.stats.json"))
    bboxes = {}
    for name in datasets:
        base = os.path.join(args.stats_dir, f"{name}10h.stats")
        with open(base + ".json") as f:
            man = json.load(f)
        arr = density.load_stats(base + ".npz")
        width, height = int(man["width"]), int(man["height"])
        groups = []
        for frac in scales:
            w_px, h_px = density.area_frac_size(width, height, frac)
            regs = density.select_top_regions(
                arr["joint"], arr["jx_edges"], arr["jy_edges"], w_px, h_px,
                args.regions_per_scale)
            if len(regs) < args.regions_per_scale:
                print(f"[warn] {name}: scale {frac:.2f} only placed "
                      f"{len(regs)}/{args.regions_per_scale} regions")
            elif _any_overlap([r["region"] for r in regs]):
                print(f"[warn] {name}: scale {frac:.2f} regions overlap "
                      f"(too large to place {args.regions_per_scale} disjoint)")
            groups.append([r["region"] for r in regs])
        bboxes[name] = groups
        print(f"[{name}] scales={[f'{s:.2f}' for s in scales]} "
              f"regions/scale={[len(g) for g in groups]}")

    plans["region"]["scale"] = scales
    plans["region"]["num"] = args.regions_per_scale
    plans["region"]["bboxes"] = bboxes

    if args.interval_counts:
        counts = {}
        for tok in args.interval_counts.split(","):
            key, val = tok.split(":")
            counts[key] = int(val)
    else:
        counts = {k: len(v) for k, v in plans["interval"].items()}

    profile, bucket_sec = aggregate_profile(args.stats_dir, datasets)
    for key, clip_min in CLIP_MINUTES.items():
        if key in counts:
            ends = density_interval_ends(profile, bucket_sec, clip_min, counts[key])
            plans["interval"][key] = ends
            print(f"[interval] {key}: K={counts[key]} -> {ends}")

    if args.dry_run:
        print("dry-run: not writing")
        return
    out = args.out or args.plans
    with open(out, "w") as f:
        json.dump(plans, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
