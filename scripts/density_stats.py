"""Compute and cache object-point density statistics for the 7 datasets.

Streams each full 10h parquet once (never loads it whole) and writes, per
dataset, a compressed ``.npz`` of numpy arrays plus a self-describing ``.json``
manifest under ``meta/density/``. See ``utilities/density.py`` for the schema.

Examples::

    python scripts/density_stats.py --datasets florida
    python scripts/density_stats.py --data-dir testdata_nosnappy --datasets f
    python scripts/density_stats.py                       # all 7, full 10h data
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import density  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = "/home/cw/DataSet/Dataset/youtube/10h/interp_pd"
DEFAULT_CONFIG = os.path.join(ROOT, "experiment", "query_configs", "dataset.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "meta", "density")

# Authoritative video resolutions (ffprobe of the source clips). Six datasets
# are 1080p; timesquare is 1440p.
RESOLUTIONS = {
    "florida": (1920, 1080),
    "ireland": (1920, 1080),
    "jacksontown": (1920, 1080),
    "beachbar": (1920, 1080),
    "osaka": (1920, 1080),
    "shinjuku": (1920, 1080),
    "timesquare": (2560, 1440),
}


def resolve_datasets(opt, registry):
    if opt == "all":
        return list(registry.keys())
    out = []
    for tok in opt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in registry:
            out.append(tok)
            continue
        cand = [n for n in registry if n.startswith(tok)]
        if len(cand) == 1:
            out.append(cand[0])
        else:
            raise KeyError(f"dataset {tok!r} is ambiguous or unknown; "
                           f"registry keys: {list(registry)}")
    # de-dup, preserve order
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _process_one(spec):
    name = spec["name"]
    stats = density.compute_stats(
        spec["parquet"], spec["classes"], spec["width"], spec["height"],
        cell_px=spec["cell_px"], t_bucket_sec=spec["t_bucket_sec"],
        joint_cell_px=spec["joint_cell_px"],
        joint_bucket_sec=spec["joint_bucket_sec"], fps=spec["fps"],
        batch_size=spec["batch_size"], count_objects=spec["count_objects"])
    os.makedirs(spec["out_dir"], exist_ok=True)
    npz_path = os.path.join(spec["out_dir"], f"{name}10h.stats.npz")
    json_path = os.path.join(spec["out_dir"], f"{name}10h.stats.json")
    density.save_stats(stats, npz_path)

    classes = [int(c) for c in stats["classes"]]
    manifest = {
        "dataset": name,
        "parquet": f"{name}10h.parquet",
        "source": os.path.abspath(spec["parquet"]),
        "fps": int(stats["fps"]),
        "width": int(stats["width"]), "height": int(stats["height"]),
        "fid_min": int(stats["fid_min"]), "fid_max": int(stats["fid_max"]),
        "n_frames": int(stats["fid_max"] - stats["fid_min"] + 1),
        "total_points": int(stats["total_points"]),
        "classes": classes,
        "class_points": {str(c): int(p) for c, p in zip(classes, stats["class_points"])},
        "params": {
            "cell_px": int(stats["cell_px"]),
            "t_bucket_sec": int(stats["t_bucket_sec"]),
            "joint_cell_px": int(stats["joint_cell_px"]),
            "joint_bucket_sec": int(stats["joint_bucket_sec"]),
        },
        "shapes": {
            "spatial": list(stats["spatial"].shape),
            "temporal": list(stats["temporal"].shape),
            "joint": list(stats["joint"].shape),
        },
        "npz": os.path.basename(npz_path),
    }
    if "total_objects" in stats:
        manifest["total_objects"] = int(stats["total_objects"])
        manifest["class_objects"] = {
            str(c): int(o) for c, o in zip(classes, stats["class_objects"])}
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--datasets", default="all",
                   help="'all', or a comma list of names / unique-prefix abbreviations")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cell-px", type=int, default=32)
    p.add_argument("--t-bucket-sec", type=int, default=60)
    p.add_argument("--joint-cell-px", type=int, default=64)
    p.add_argument("--joint-bucket-sec", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1 << 20)
    p.add_argument("--width", type=int, default=None, help="override frame width")
    p.add_argument("--height", type=int, default=None, help="override frame height")
    p.add_argument("--count-objects", action="store_true",
                   help="also count unique oids per class (memory-heavy on timesquare)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        registry = json.load(f)
    names = resolve_datasets(args.datasets, registry)

    specs = []
    for name in names:
        parquet = os.path.join(args.data_dir, f"{name}10h.parquet")
        if not os.path.exists(parquet):
            print(f"[skip] {name}: missing {parquet}")
            continue
        npz_path = os.path.join(args.out_dir, f"{name}10h.stats.npz")
        if os.path.exists(npz_path) and not args.overwrite:
            print(f"[skip] {name}: {npz_path} exists (use --overwrite)")
            continue
        w, h = RESOLUTIONS.get(name, (1920, 1080))
        if args.width:
            w = args.width
        if args.height:
            h = args.height
        specs.append({
            "name": name, "parquet": parquet,
            "classes": registry[name]["top9_cat"], "width": w, "height": h,
            "cell_px": args.cell_px, "t_bucket_sec": args.t_bucket_sec,
            "joint_cell_px": args.joint_cell_px,
            "joint_bucket_sec": args.joint_bucket_sec, "fps": args.fps,
            "batch_size": args.batch_size, "count_objects": args.count_objects,
            "out_dir": args.out_dir,
        })

    if not specs:
        print("nothing to do")
        return

    for spec in specs:
        print(f"[{spec['name']}] {spec['parquet']}  {spec['width']}x{spec['height']}")

    t0 = time.time()
    if args.jobs > 1 and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for m in ex.map(_process_one, specs):
                print(f"[done] {m['dataset']}: total_points={m['total_points']} "
                      f"joint={m['shapes']['joint']}")
    else:
        for spec in specs:
            s = time.time()
            m = _process_one(spec)
            print(f"[done] {m['dataset']}: total_points={m['total_points']} "
                  f"joint={m['shapes']['joint']} ({time.time() - s:.1f}s)")
    print(f"all done in {time.time() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
