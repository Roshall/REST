"""Object-point density statistics and (region, interval) query generation.

Stats are computed over the *full* video frame (the "video visual region"), per
class, restricted to a dataset's top-9 classes:

- ``spatial``  ``(C, Hc, Wc)``      point counts on a fine ``cell_px`` pixel grid
- ``temporal`` ``(C, T)``           point counts in ``t_bucket_sec`` time buckets
- ``joint``    ``(C, Tj, Hj, Wj)``  point counts on a coarser cell x time grid,
                                    used for O(1) joint (region, interval) scoring

All stored time values are frame indices (``fid``); intervals are authored in
seconds and converted with ``fps``. Query regions follow the repo convention
``[x_min, x_max, y_min, y_max]`` with **inclusive** bounds; intervals are
half-open ``[start_sec, end_sec)``.
"""

import numpy as np

# LUT sentinel for classes outside the dataset's top-9 (cls is uint8, 0..79).
NON_CLASS = 255

# Arrays persisted to the ``.npz`` stats cache.
NPZ_KEYS = ("spatial", "temporal", "joint", "classes", "class_points",
            "x_edges", "y_edges", "jx_edges", "jy_edges", "t_edges", "jt_edges")
NPZ_OPTIONAL_KEYS = ("class_objects", "total_objects")


def _ceil_div(a, b):
    return -(-int(a) // int(b))


def make_class_lut(classes, table_size=256):
    """Map raw ``cls`` values to a compact axis index, others to ``NON_CLASS``."""
    lut = np.full(table_size, NON_CLASS, dtype=np.uint8)
    lut[np.asarray(classes, dtype=np.int64)] = np.arange(len(classes), dtype=np.uint8)
    return lut


# ---------------------------------------------------------------------------
# stats computation
# ---------------------------------------------------------------------------
def _fid_bounds(pf):
    """Return ``(fid_min, fid_max)`` from row-group statistics, else a scan."""
    md = pf.metadata
    col_idx = None
    first = md.row_group(0)
    for i in range(first.num_columns):
        if first.column(i).path_in_schema == "fid":
            col_idx = i
            break
    lo = hi = None
    if col_idx is not None:
        for r in range(md.num_row_groups):
            st = md.row_group(r).column(col_idx).statistics
            if st is None or not st.has_min_max:
                lo = hi = None
                break
            lo = st.min if lo is None else min(lo, st.min)
            hi = st.max if hi is None else max(hi, st.max)
    if lo is None:
        for b in pf.iter_batches(batch_size=1 << 20, columns=["fid"]):
            f = b.column("fid").to_numpy(zero_copy_only=False)
            if f.size:
                a, z = int(f.min()), int(f.max())
                lo = a if lo is None else min(lo, a)
                hi = z if hi is None else max(hi, z)
    if lo is None:
        raise ValueError("no rows / no fid statistics in parquet")
    return int(lo), int(hi)


def build_edges(width, height, fid_min, fid_max, cell_px, t_bucket_sec,
                joint_cell_px, joint_bucket_sec, fps):
    """Cell / bucket pixel & frame edges (last edge clipped to the real extent)."""
    n_frames = fid_max - fid_min + 1
    Wc, Hc = _ceil_div(width, cell_px), _ceil_div(height, cell_px)
    T = _ceil_div(n_frames, t_bucket_sec * fps)
    Wj, Hj = _ceil_div(width, joint_cell_px), _ceil_div(height, joint_cell_px)
    Tj = _ceil_div(n_frames, joint_bucket_sec * fps)

    x_edges = np.arange(Wc + 1, dtype=np.int64) * cell_px
    x_edges[-1] = width
    y_edges = np.arange(Hc + 1, dtype=np.int64) * cell_px
    y_edges[-1] = height
    jx_edges = np.arange(Wj + 1, dtype=np.int64) * joint_cell_px
    jx_edges[-1] = width
    jy_edges = np.arange(Hj + 1, dtype=np.int64) * joint_cell_px
    jy_edges[-1] = height
    t_edges = fid_min + np.arange(T + 1, dtype=np.int64) * (t_bucket_sec * fps)
    t_edges[-1] = fid_max + 1
    jt_edges = fid_min + np.arange(Tj + 1, dtype=np.int64) * (joint_bucket_sec * fps)
    jt_edges[-1] = fid_max + 1
    return {
        "x_edges": x_edges, "y_edges": y_edges,
        "jx_edges": jx_edges, "jy_edges": jy_edges,
        "t_edges": t_edges, "jt_edges": jt_edges,
    }


def compute_stats(parquet_path, classes, width, height, cell_px=32,
                  t_bucket_sec=60, joint_cell_px=64, joint_bucket_sec=300,
                  fps=30, batch_size=1 << 20, count_objects=False,
                  fid_min=None, fid_max=None):
    """Stream a parquet and accumulate per-class spatial/temporal/joint counts.

    Returns a dict of numpy arrays and scalars (see the module docstring).
    """
    import pyarrow.parquet as pq

    classes = [int(c) for c in classes]
    C = len(classes)
    pf = pq.ParquetFile(parquet_path)
    if fid_min is None or fid_max is None:
        fid_min, fid_max = _fid_bounds(pf)
    edges = build_edges(width, height, fid_min, fid_max, cell_px, t_bucket_sec,
                        joint_cell_px, joint_bucket_sec, fps)
    x_edges, y_edges = edges["x_edges"], edges["y_edges"]
    jx_edges, jy_edges = edges["jx_edges"], edges["jy_edges"]
    Wc, Hc = len(x_edges) - 1, len(y_edges) - 1
    Wj, Hj = len(jx_edges) - 1, len(jy_edges) - 1
    T = len(edges["t_edges"]) - 1
    Tj = len(edges["jt_edges"]) - 1
    bf, jbf = t_bucket_sec * fps, joint_bucket_sec * fps

    lut = make_class_lut(classes)
    spatial = np.zeros((C, Hc, Wc), dtype=np.int64)
    temporal = np.zeros((C, T), dtype=np.int64)
    joint = np.zeros((C, Tj, Hj, Wj), dtype=np.int64)
    class_points = np.zeros(C, dtype=np.int64)
    total_points = 0
    oid_sets = [set() for _ in range(C)] if count_objects else None

    columns = ["cls", "fid", "x", "y"] + (["oid"] if count_objects else [])
    for b in pf.iter_batches(batch_size=batch_size, columns=columns):
        total_points += b.num_rows
        cls = b.column("cls").to_numpy(zero_copy_only=False)
        ci = lut[cls].astype(np.int64)
        m = ci != NON_CLASS
        if not m.any():
            continue
        ci = ci[m]
        fid = b.column("fid").to_numpy(zero_copy_only=False).astype(np.int64)[m]
        x = b.column("x").to_numpy(zero_copy_only=False).astype(np.int64)[m]
        y = b.column("y").to_numpy(zero_copy_only=False).astype(np.int64)[m]
        if oid_sets is not None:
            oid = b.column("oid").to_numpy(zero_copy_only=False)[m]
            for c in range(C):
                sel = ci == c
                if sel.any():
                    oid_sets[c].update(oid[sel].tolist())

        class_points += np.bincount(ci, minlength=C)

        fx = np.minimum(x // cell_px, Wc - 1)
        fy = np.minimum(y // cell_px, Hc - 1)
        flat = (ci * Hc + fy) * Wc + fx
        spatial += np.bincount(flat, minlength=C * Hc * Wc).reshape(C, Hc, Wc)

        tb = np.clip((fid - fid_min) // bf, 0, T - 1)
        temporal += np.bincount(ci * T + tb, minlength=C * T).reshape(C, T)

        jx = np.minimum(x // joint_cell_px, Wj - 1)
        jy = np.minimum(y // joint_cell_px, Hj - 1)
        jt = np.clip((fid - fid_min) // jbf, 0, Tj - 1)
        fJ = ((ci * Tj + jt) * Hj + jy) * Wj + jx
        joint += np.bincount(fJ, minlength=C * Tj * Hj * Wj).reshape(C, Tj, Hj, Wj)

    out = {
        "spatial": spatial, "temporal": temporal, "joint": joint,
        "classes": np.asarray(classes, dtype=np.int32),
        "class_points": class_points,
        "x_edges": x_edges, "y_edges": y_edges,
        "jx_edges": jx_edges, "jy_edges": jy_edges,
        "t_edges": edges["t_edges"], "jt_edges": edges["jt_edges"],
        "total_points": np.int64(total_points),
        "fid_min": np.int64(fid_min), "fid_max": np.int64(fid_max),
        "width": np.int64(width), "height": np.int64(height), "fps": np.int64(fps),
        "cell_px": np.int64(cell_px), "t_bucket_sec": np.int64(t_bucket_sec),
        "joint_cell_px": np.int64(joint_cell_px),
        "joint_bucket_sec": np.int64(joint_bucket_sec),
    }
    if oid_sets is not None:
        obj = np.array([len(s) for s in oid_sets], dtype=np.int64)
        out["class_objects"] = obj
        out["total_objects"] = np.int64(int(obj.sum()))
    return out


def save_stats(stats, npz_path):
    """Persist the numpy arrays of a stats dict to a compressed ``.npz``."""
    arrays = {k: np.asarray(stats[k]) for k in NPZ_KEYS}
    for k in NPZ_OPTIONAL_KEYS:
        if k in stats:
            arrays[k] = np.asarray(stats[k])
    np.savez_compressed(npz_path, **arrays)


def load_stats(npz_path):
    """Load a stats ``.npz`` into a plain dict of arrays."""
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# prefix-sum scoring
# ---------------------------------------------------------------------------
def prefix3d(t):
    """Cumulative sum over the last 3 axes, padded with a zero face."""
    p = np.zeros((t.shape[0] + 1, t.shape[1] + 1, t.shape[2] + 1), dtype=np.int64)
    p[1:, 1:, 1:] = t.cumsum(0).cumsum(1).cumsum(2)
    return p


def box_sum(P, t0, t1, y0, y1, x0, x1):
    """Half-open box sum on a 3D prefix. Indices may be scalars or arrays."""
    return (P[t1, y1, x1] - P[t0, y1, x1] - P[t1, y0, x1] + P[t0, y0, x1]
            - P[t1, y1, x0] + P[t0, y1, x0] + P[t1, y0, x0] - P[t0, y0, x0])


# ---------------------------------------------------------------------------
# candidate region / interval enumeration
# ---------------------------------------------------------------------------
def region_cells(edge, size_px):
    """Number of cells of ``edge`` needed to span ``size_px`` pixels (>=1)."""
    k = int(np.searchsorted(edge, size_px, side="left"))
    return max(1, k)


def area_frac_size(width, height, frac, aspect=None):
    """Pixel ``(w, h)`` of a region covering ``frac`` of the frame area.

    ``aspect`` is ``(rw, rh)``; ``None`` means use the frame aspect.
    """
    rw, rh = (width, height) if aspect is None else aspect
    area = frac * width * height
    w = int(round((area * rw / rh) ** 0.5))
    h = int(round(w * rh / rw))
    return max(1, w), max(1, h)


def select_top_regions(joint, jx_edges, jy_edges, w_px, h_px, n, stride=1,
                       min_score=0.0):
    """Top-``n`` regions by time/class-aggregated density.

    Regions are non-overlapping when possible; if fewer than ``n`` fit, the
    remainder is filled by score, allowing overlap (so ``n`` is always met when
    the candidate set is large enough). Returns a list of dicts with
    ``region`` ``[x_min, x_max, y_min, y_max]`` (inclusive), ``count`` and
    ``cells`` ``[cw, ch]``.
    """
    Jsp = joint.sum(axis=1).sum(axis=0)  # (Hj, Wj): aggregate over time & class
    Hj, Wj = Jsp.shape
    P = np.zeros((Hj + 1, Wj + 1), dtype=np.int64)
    P[1:, 1:] = Jsp.cumsum(0).cumsum(1)
    cw = min(Wj, region_cells(jx_edges, w_px))
    ch = min(Hj, region_cells(jy_edges, h_px))
    xs = np.arange(0, Wj - cw + 1, stride)
    ys = np.arange(0, Hj - ch + 1, stride)
    ix, iy = np.meshgrid(xs, ys, indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    scores = (P[iy + ch, ix + cw] - P[iy, ix + cw]
              - P[iy + ch, ix] + P[iy, ix]).astype(np.float64)
    order = np.lexsort((iy, ix, -scores))
    occ = np.zeros((Hj, Wj), dtype=bool)
    chosen = []
    for j in order:
        if len(chosen) >= n or scores[j] < min_score:
            break
        x, y = int(ix[j]), int(iy[j])
        if occ[y:y + ch, x:x + cw].any():
            continue
        occ[y:y + ch, x:x + cw] = True
        chosen.append(int(j))
    if len(chosen) < n:  # not enough spatial capacity: fill by score, overlapping
        taken = set(chosen)
        for j in order:
            if len(chosen) >= n:
                break
            if scores[j] < min_score:
                break
            if int(j) in taken:
                continue
            chosen.append(int(j))
            taken.add(int(j))
    out = []
    for j in chosen:
        x, y = int(ix[j]), int(iy[j])
        out.append({
            "region": [int(jx_edges[x]), int(jx_edges[x + cw]) - 1,
                       int(jy_edges[y]), int(jy_edges[y + ch]) - 1],
            "count": int(scores[j]),
            "cells": [cw, ch],
        })
    return out


def enumerate_candidates(Wj, Hj, Tj, cw, ch, nt, region_stride, t_stride):
    xs = np.arange(0, Wj - cw + 1, region_stride, dtype=np.int64)
    ys = np.arange(0, Hj - ch + 1, region_stride, dtype=np.int64)
    ts = np.arange(0, Tj - nt + 1, t_stride, dtype=np.int64)
    if xs.size == 0 or ys.size == 0 or ts.size == 0:
        raise ValueError("empty candidate set (region/interval too large)")
    ix, iy, t0 = np.meshgrid(xs, ys, ts, indexing="ij")
    return ix.ravel(), iy.ravel(), t0.ravel()


def score_candidates(P, ix, iy, t0, cw, ch, nt):
    return box_sum(P, t0, t0 + nt, iy, iy + ch, ix, ix + cw).astype(np.float64)


def select_greedy(scores, ix, iy, t0, cw, ch, Wj, Hj, n, min_score=0.0):
    """Top-score-first, non-overlapping regions. Returns chosen candidate idx."""
    order = np.lexsort((t0, iy, ix, -scores))
    occ = np.zeros((Hj, Wj), dtype=bool)
    chosen = []
    for j in order:
        if len(chosen) >= n:
            break
        if scores[j] < min_score:
            break  # order is score-descending
        x, y = int(ix[j]), int(iy[j])
        if occ[y:y + ch, x:x + cw].any():
            continue
        occ[y:y + ch, x:x + cw] = True
        chosen.append(int(j))
    return np.asarray(chosen, dtype=np.int64)


def select_weighted(scores, ix, iy, cw, ch, Wj, Hj, n, rng,
                    allow_overlap=False, min_score=0.0):
    """Seeded density-weighted ordering; non-overlapping unless ``allow_overlap``."""
    w = np.where(scores >= min_score, scores, 0.0)
    if not np.any(w > 0):
        w = np.ones_like(w)
    pos = w > 0
    g = rng.gumbel(size=int(pos.sum()))
    order_pos = np.nonzero(pos)[0][np.argsort(-(np.log(w[pos]) + g))]
    neg = np.nonzero(~pos)[0]
    rng.shuffle(neg)
    order = np.concatenate([order_pos, neg])
    chosen = []
    if allow_overlap:
        for j in order:
            if len(chosen) >= n:
                break
            chosen.append(int(j))
    else:
        occ = np.zeros((Hj, Wj), dtype=bool)
        for j in order:
            if len(chosen) >= n:
                break
            x, y = int(ix[j]), int(iy[j])
            if occ[y:y + ch, x:x + cw].any():
                continue
            occ[y:y + ch, x:x + cw] = True
            chosen.append(int(j))
    return np.asarray(chosen, dtype=np.int64)


# ---------------------------------------------------------------------------
# high-level query generation
# ---------------------------------------------------------------------------
def build_queries(arrays, manifest, region_w_px, region_h_px, interval_sec,
                  n, mode="greedy", score="density", region_stride=1,
                  interval_stride_sec=None, seed=1234, allow_overlap=False,
                  min_score=0.0, labels=None):
    """Return a list of joint ``(region, interval)`` query dicts for one dataset.

    ``arrays``/``manifest`` come from a saved stats file. Regions are in pixels
    ``[x_min, x_max, y_min, y_max]`` (inclusive); intervals ``[start_sec,end_sec)``.
    """
    joint = arrays["joint"]
    classes = np.asarray(arrays["classes"]).tolist()
    jx_edges = arrays["jx_edges"].astype(np.int64)
    jy_edges = arrays["jy_edges"].astype(np.int64)
    jt_edges = arrays["jt_edges"].astype(np.int64)
    fps = int(manifest["fps"])
    jbs = int(manifest["params"]["joint_bucket_sec"])
    C, Tj, Hj, Wj = joint.shape

    if labels is None:
        sel = list(range(C))
    else:
        want = set(int(x) for x in labels)
        sel = [i for i, c in enumerate(classes) if c in want]
        if not sel:
            raise ValueError(f"none of the labels {sorted(want)} in {classes}")
    Jsel = joint[sel].sum(axis=0)
    P = prefix3d(Jsel)
    Pc = np.stack([prefix3d(joint[c]) for c in sel])  # for per-class breakdown

    cw = min(Wj, region_cells(jx_edges, region_w_px))
    ch = min(Hj, region_cells(jy_edges, region_h_px))
    nt = max(1, int(round(interval_sec / jbs)))
    nt = min(nt, Tj)
    if interval_stride_sec is None:
        interval_stride_sec = interval_sec
    t_stride = max(1, int(round(interval_stride_sec / jbs)))

    ix, iy, t0 = enumerate_candidates(Wj, Hj, Tj, cw, ch, nt,
                                      region_stride, t_stride)
    scores = score_candidates(P, ix, iy, t0, cw, ch, nt)
    if score == "density":
        norm = float(cw * ch * nt)
        metrics = scores / norm
    else:
        metrics = scores

    if mode == "greedy":
        chosen = select_greedy(metrics, ix, iy, t0, cw, ch, Wj, Hj, n, min_score)
    elif mode == "weighted":
        rng = np.random.default_rng(seed)
        chosen = select_weighted(metrics, ix, iy, cw, ch, Wj, Hj, n, rng,
                                 allow_overlap, min_score)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    queries = []
    for qid, j in enumerate(chosen):
        x0, y0 = int(ix[j]), int(iy[j])
        xl, xr = int(jx_edges[x0]), int(jx_edges[x0 + cw])
        yt, yb = int(jy_edges[y0]), int(jy_edges[y0 + ch])
        tlo, thi = int(jt_edges[t0[j]]), int(jt_edges[t0[j] + nt])
        per_class = {
            int(classes[c]): int(box_sum(Pc[i], t0[j], t0[j] + nt, y0, y0 + ch, x0, x0 + cw))
            for i, c in enumerate(sel)
        }
        queries.append({
            "id": qid,
            "region": [xl, xr - 1, yt, yb - 1],
            "interval": [tlo / fps, thi / fps],
            "score": float(metrics[j]),
            "count": int(scores[j]),
            "region_px": [xr - xl, yb - yt],
            "interval_sec": (thi - tlo) / fps,
            "cell_origin": [x0, y0, int(t0[j])],
            "per_class": per_class,
        })
    return queries
