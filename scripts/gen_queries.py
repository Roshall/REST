"""Generate query (region, interval) pairs from cached density statistics.

Loads the ``meta/density/<name>10h.stats.*`` cache produced by
``scripts/density_stats.py`` and emits ``N`` joint (region, interval) queries per
dataset, chosen for contained object density. Two modes:

- ``greedy``   : highest-density first, regions non-overlapping (deterministic).
- ``weighted`` : density-weighted seeded random ordering (reproducible).

Region size is given either in pixels (``--region-wh W H``) or as a fraction of
the frame area (``--region-area-frac F``). Regions are ``[x_min, x_max, y_min,
y_max]`` (inclusive pixels); intervals are ``[start_sec, end_sec)`` (seconds).

Example::

    python scripts/gen_queries.py --region-wh 512 288 --interval-sec 600 \
        --n 20 --mode greedy
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import density  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATS_DIR = os.path.join(ROOT, "meta", "density")
DEFAULT_OUT = os.path.join(ROOT, "experiment", "query_configs",
                           "generated_queries.json")


def available_datasets(stats_dir):
    names = []
    for fn in sorted(os.listdir(stats_dir)):
        if fn.endswith("10h.stats.json"):
            names.append(fn[: -len("10h.stats.json")])
    return names


def resolve_datasets(opt, names):
    if opt == "all":
        return list(names)
    out = []
    for tok in opt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in names:
            out.append(tok)
            continue
        cand = [n for n in names if n.startswith(tok)]
        if len(cand) == 1:
            out.append(cand[0])
        else:
            raise KeyError(f"dataset {tok!r} ambiguous/unknown; available: {names}")
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def region_size_px(args, width, height):
    """Resolve the requested region to ``(w_px, h_px)`` for one dataset frame."""
    if args.region_wh is not None:
        return int(args.region_wh[0]), int(args.region_wh[1])
    if args.aspect == ["frame"]:
        aspect = None
    elif len(args.aspect) == 2:
        aspect = (float(args.aspect[0]), float(args.aspect[1]))
    else:
        raise SystemExit("--aspect takes two numbers (e.g. 16 9) or 'frame'")
    return density.area_frac_size(width, height, float(args.region_area_frac),
                                  aspect)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    p.add_argument("--datasets", default="all")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--mode", choices=("greedy", "weighted"), default="greedy")
    p.add_argument("--score", choices=("density", "count"), default="density")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--region-wh", type=int, nargs=2, metavar=("W", "H"))
    g.add_argument("--region-area-frac", type=float, metavar="F")
    p.add_argument("--aspect", nargs="+", default=["16", "9"],
                   help="aspect ratio R W (default 16 9), or the literal 'frame'")
    p.add_argument("--region-stride-cells", type=int, default=1)
    p.add_argument("--interval-sec", type=float, default=300.0)
    p.add_argument("--interval-stride-sec", type=float, default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--strict-interval", action="store_true",
                   help="error if interval-sec is not a whole number of joint buckets")
    p.add_argument("--labels", default=None,
                   help="comma list of class ids to restrict density to (default all)")
    return p.parse_args()


def main():
    args = parse_args()
    names_all = available_datasets(args.stats_dir)
    if not names_all:
        raise SystemExit(f"no stats found in {args.stats_dir}; run density_stats.py first")
    names = resolve_datasets(args.datasets, names_all)
    labels = None if args.labels is None else [int(x) for x in args.labels.split(",")]

    out = {
        "meta": {
            "stats_dir": os.path.abspath(args.stats_dir),
            "mode": args.mode,
            "score": args.score,
            "region_size": {
                "form": "pixels" if args.region_wh is not None else "area_frac",
                "wh": list(args.region_wh) if args.region_wh is not None else None,
                "area_frac": args.region_area_frac,
                "aspect": args.aspect,
            },
            "region_stride_cells": args.region_stride_cells,
            "interval_length_sec": args.interval_sec,
            "interval_stride_sec": args.interval_stride_sec,
            "n_requested": args.n,
            "seed": args.seed,
            "allow_overlap": args.allow_overlap,
            "min_score": args.min_score,
            "labels": labels,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "datasets": {},
    }
    total = 0
    for name in names:
        base = os.path.join(args.stats_dir, f"{name}10h.stats")
        with open(base + ".json") as f:
            manifest = json.load(f)
        arrays = density.load_stats(base + ".npz")
        width, height = int(manifest["width"]), int(manifest["height"])
        jbs = int(manifest["params"]["joint_bucket_sec"])

        if args.strict_interval and abs(args.interval_sec / jbs
                                        - round(args.interval_sec / jbs)) > 1e-9:
            raise SystemExit(f"[{name}] interval-sec={args.interval_sec} is not a "
                             f"multiple of joint bucket {jbs}s (--strict-interval)")

        w_px, h_px = region_size_px(args, width, height)
        queries = density.build_queries(
            arrays, manifest, w_px, h_px, args.interval_sec, args.n,
            mode=args.mode, score=args.score,
            region_stride=args.region_stride_cells,
            interval_stride_sec=args.interval_stride_sec, seed=args.seed,
            allow_overlap=args.allow_overlap, min_score=args.min_score,
            labels=labels)
        total += len(queries)
        real_len = queries[0]["interval_sec"] if queries else 0.0
        real_reg = queries[0]["region_px"] if queries else [w_px, h_px]
        if queries and abs(real_len - args.interval_sec) > 1e-6:
            print(f"[warn] {name}: interval {args.interval_sec}s snapped to "
                  f"{real_len}s (joint bucket {jbs}s)")
        if len(queries) < args.n:
            print(f"[warn] {name}: generated {len(queries)}/{args.n} "
                  f"(non-overlapping region capacity reached; use a smaller "
                  f"region, --mode weighted --allow-overlap, or lower --n)")
        out["datasets"][name] = {
            "classes": manifest["classes"],
            "stats": manifest["npz"],
            "region_px": real_reg,
            "region_px_requested": [w_px, h_px],
            "interval_sec": real_len,
            "n_generated": len(queries),
            "queries": queries,
        }
        print(f"[{name}] {len(queries)}/{args.n} queries "
              f"region={real_reg[0]}x{real_reg[1]} interval={real_len}s")

    out["meta"]["n_generated"] = total
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {total} queries -> {args.out}")


if __name__ == "__main__":
    main()
