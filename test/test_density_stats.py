"""Tests for the density-stats cache and (region, interval) query generator."""

import os

import numpy as np
import pandas as pd
import pytest

from utilities import density
from utilities.time_unit import seconds_to_frames

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "testdata_nosnappy", "florida10h.parquet")
FLORIDA_TOP9 = [2, 0, 10, 58, 74, 1, 7, 25, 73]


# ---------------------------------------------------------------------------
# synthetic binning
# ---------------------------------------------------------------------------
def _write_synth(path):
    df = pd.DataFrame({
        "oid": np.array([10, 11, 12, 13, 14], dtype="int32"),
        "cls": np.array([1, 1, 1, 2, 3], dtype="uint8"),
        "fid": np.array([1, 10, 20, 5, 5], dtype="int32"),
        "x": np.array([5, 5, 40, 40, 5], dtype="uint16"),
        "y": np.array([5, 5, 40, 5, 5], dtype="uint16"),
    })
    df.to_parquet(path, index=False)


def test_synthetic_binning(tmp_path):
    p = str(tmp_path / "synth.parquet")
    _write_synth(p)
    s = density.compute_stats(p, [1, 2], width=64, height=64, cell_px=32,
                              t_bucket_sec=1, joint_cell_px=32,
                              joint_bucket_sec=2, fps=10)
    # 4 top rows; the cls=3 row is excluded from every accumulator but counted
    assert int(s["total_points"]) == 5
    assert int(s["class_points"].sum()) == 4
    assert s["spatial"].sum() == s["temporal"].sum() == s["joint"].sum() == 4

    sp = s["spatial"]  # (C=2, Hc=2, Wc=2), class order [1, 2]
    assert int(sp[0, 0, 0]) == 2       # cls1: (5,5) twice
    assert int(sp[0, 1, 1]) == 1       # cls1: (40,40)
    assert int(sp[1, 0, 1]) == 1       # cls2: (40,5)
    assert int(sp[1].sum()) == 1

    te = s["temporal"]  # buckets: [1,11) -> 0, [11,21) -> 1
    assert list(map(int, te[0])) == [2, 1]
    assert list(map(int, te[1])) == [1, 0]

    jo = s["joint"]  # single joint bucket
    assert jo.shape == (2, 1, 2, 2)
    assert int(jo[0, 0, 0, 0]) == 2 and int(jo[0, 0, 1, 1]) == 1

    assert list(map(int, s["x_edges"])) == [0, 32, 64]
    assert list(map(int, s["t_edges"])) == [1, 11, 21]
    assert list(map(int, s["jt_edges"])) == [1, 21]


def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "synth.parquet")
    _write_synth(p)
    s = density.compute_stats(p, [1, 2], 64, 64, cell_px=32, t_bucket_sec=1,
                              joint_cell_px=32, joint_bucket_sec=2, fps=10)
    npz = str(tmp_path / "s.npz")
    density.save_stats(s, npz)
    d = density.load_stats(npz)
    for k in density.NPZ_KEYS:
        assert np.array_equal(d[k], np.asarray(s[k]))


# ---------------------------------------------------------------------------
# brute-force parity on the in-repo 30-min sample
# ---------------------------------------------------------------------------
pytestmark_sample = pytest.mark.skipif(not os.path.exists(SAMPLE),
                                       reason="testdata_nosnappy sample missing")


@pytestmark_sample
def test_bruteforce_parity():
    cell, tbucket, jcell, jbucket = 32, 60, 64, 300
    s = density.compute_stats(SAMPLE, FLORIDA_TOP9, 1920, 1080, cell_px=cell,
                              t_bucket_sec=tbucket, joint_cell_px=jcell,
                              joint_bucket_sec=jbucket, fps=30)
    df = pd.read_parquet(SAMPLE, columns=["cls", "fid", "x", "y"])
    df = df[df.cls.isin(FLORIDA_TOP9)]
    fid_min = int(df.fid.min())
    Wc = (1920 + cell - 1) // cell
    Hc = (1080 + cell - 1) // cell
    Wj = (1920 + jcell - 1) // jcell
    Hj = (1080 + jcell - 1) // jcell
    T = (int(df.fid.max()) - fid_min + 1 + tbucket * 30 - 1) // (tbucket * 30)
    Tj = (int(df.fid.max()) - fid_min + 1 + jbucket * 30 - 1) // (jbucket * 30)

    for i, c in enumerate(FLORIDA_TOP9):
        sub = df[df.cls == c]
        xi = np.minimum(sub.x.to_numpy() // cell, Wc - 1)
        yi = np.minimum(sub.y.to_numpy() // cell, Hc - 1)
        brut = np.zeros((Hc, Wc), dtype=np.int64)
        np.add.at(brut, (yi, xi), 1)
        assert np.array_equal(s["spatial"][i], brut), f"spatial class {c}"

        tbi = np.clip((sub.fid.to_numpy() - fid_min) // (tbucket * 30), 0, T - 1)
        brut_t = np.bincount(tbi, minlength=T)
        assert np.array_equal(s["temporal"][i], brut_t), f"temporal class {c}"

        jxi = np.minimum(sub.x.to_numpy() // jcell, Wj - 1)
        jyi = np.minimum(sub.y.to_numpy() // jcell, Hj - 1)
        jti = np.clip((sub.fid.to_numpy() - fid_min) // (jbucket * 30), 0, Tj - 1)
        brut_j = np.zeros((Tj, Hj, Wj), dtype=np.int64)
        np.add.at(brut_j, (jti, jyi, jxi), 1)
        assert np.array_equal(s["joint"][i], brut_j), f"joint class {c}"


@pytestmark_sample
def test_projections_consistent():
    s = density.compute_stats(SAMPLE, FLORIDA_TOP9, 1920, 1080, cell_px=32,
                              t_bucket_sec=60, joint_cell_px=64,
                              joint_bucket_sec=300, fps=30)
    C, Hc, Wc = s["spatial"].shape
    _, Tj, Hj, Wj = s["joint"].shape
    assert Hc % Hj == 0 and Wc % Wj == 0
    # fine spatial -> coarse, compare to joint summed over time
    blk = s["spatial"].reshape(C, Hj, Hc // Hj, Wj, Wc // Wj)
    assert np.array_equal(blk.sum(axis=(2, 4)), s["joint"].sum(axis=1))
    # temporal bucket 60s -> joint bucket 300s (5x)
    _, T = s["temporal"].shape
    assert T % Tj == 0
    tsum = s["temporal"].reshape(C, Tj, T // Tj).sum(axis=2)
    assert np.array_equal(tsum, s["joint"].sum(axis=(2, 3)))


# ---------------------------------------------------------------------------
# prefix-sum scoring
# ---------------------------------------------------------------------------
def test_prefix_box_sum_matches_bruteforce():
    rng = np.random.default_rng(0)
    joint = rng.integers(0, 5, size=(2, 6, 5, 7))
    P = density.prefix3d(joint.sum(axis=0))
    for _ in range(20):
        t0, nt = sorted(rng.integers(0, 7, size=2))
        y0, ch = sorted(rng.integers(0, 6, size=2))
        x0, cw = sorted(rng.integers(0, 8, size=2))
        if t0 == nt or y0 == ch or x0 == cw:
            continue
        got = density.box_sum(P, t0, nt, y0, ch, x0, cw)
        want = joint.sum(axis=0)[t0:nt, y0:ch, x0:cw].sum()
        assert int(got) == int(want)


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def _fake_stats(spike=(1, 2, 1), val=1000, shape=(1, 4, 4, 4), fps=30, jbs=300):
    C, Tj, Hj, Wj = shape
    joint = np.zeros(shape, dtype=np.int64)
    if spike is not None:
        joint[0, spike[2], spike[1], spike[0]] = val
    edges = np.arange(Wj + 1, dtype=np.int64) * 32
    jedges = np.arange(Hj + 1, dtype=np.int64) * 32
    jt = np.arange(Tj + 1, dtype=np.int64) * (jbs * fps)
    arrays = {
        "joint": joint, "classes": np.arange(C, dtype=np.int32),
        "jx_edges": edges, "jy_edges": jedges, "jt_edges": jt,
    }
    manifest = {"fps": fps, "params": {"joint_bucket_sec": jbs},
                "width": int(edges[-1]), "height": int(jedges[-1])}
    return arrays, manifest


def test_greedy_picks_densest_first():
    arrays, man = _fake_stats(spike=(1, 2, 1))
    q = density.build_queries(arrays, man, 32, 32, 300, 1, mode="greedy")
    assert len(q) == 1
    assert q[0]["region"] == [32, 63, 64, 95]
    assert q[0]["interval"] == [300.0, 600.0]
    assert q[0]["count"] == 1000


def test_greedy_regions_non_overlapping():
    rng = np.random.default_rng(1)
    arrays, man = _fake_stats(spike=None)
    arrays["joint"] = rng.integers(1, 10, size=arrays["joint"].shape).astype(np.int64)
    # 64px region -> 2x2 joint cells; a 4x4 grid fits at most 4 non-overlapping
    q = density.build_queries(arrays, man, 64, 64, 300, 4, mode="greedy")
    assert 1 <= len(q) <= 4
    boxes = [tuple(x["cell_origin"]) for x in q]
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            ax, ay, _ = boxes[i]
            bx, by, _ = boxes[j]
            assert not (ax < bx + 2 and bx < ax + 2 and ay < by + 2 and by < ay + 2)


def test_greedy_caps_at_max_non_overlapping():
    arrays, man = _fake_stats(spike=None)
    arrays["joint"][:] = 5
    # region spanning the whole grid -> only one can fit
    q = density.build_queries(arrays, man, 32 * 4, 32 * 4, 300, 5, mode="greedy")
    assert len(q) == 1


def test_weighted_reproducible_same_seed():
    rng = np.random.default_rng(2)
    arrays, man = _fake_stats(spike=None)
    arrays["joint"] = rng.integers(1, 10, size=arrays["joint"].shape).astype(np.int64)
    q1 = density.build_queries(arrays, man, 32, 32, 300, 4, mode="weighted", seed=7)
    q2 = density.build_queries(arrays, man, 32, 32, 300, 4, mode="weighted", seed=7)
    q3 = density.build_queries(arrays, man, 32, 32, 300, 4, mode="weighted", seed=8)
    assert q1 == q2
    assert q1 != q3


def test_min_score_and_conventions():
    arrays, man = _fake_stats(spike=(1, 2, 1))
    # with min_score above the zero cells, only the spike qualifies
    q = density.build_queries(arrays, man, 32, 32, 300, 5, mode="greedy",
                              min_score=1.0)
    assert len(q) == 1
    for qq in q:
        xl, xr, yt, yb = qq["region"]
        assert 0 <= xl <= xr < man["width"] and 0 <= yt <= yb < man["height"]
        lo, hi = qq["interval"]
        assert seconds_to_frames(hi, fps=30) > seconds_to_frames(lo, fps=30)


# ---------------------------------------------------------------------------
# region sizing / top-region selection / interval endpoints
# ---------------------------------------------------------------------------
def test_area_frac_size_area_and_aspect():
    w, h = density.area_frac_size(1920, 1080, 0.05)  # frame aspect
    assert abs(w * h - 0.05 * 1920 * 1080) / (0.05 * 1920 * 1080) < 0.02
    assert abs(w / h - 16 / 9) < 0.05
    w2, h2 = density.area_frac_size(1920, 1080, 0.05, aspect=(1, 1))
    assert abs(w2 - h2) <= 1


def test_select_top_regions_finds_spike_and_fills_n():
    arrays, man = _fake_stats(spike=(1, 2, 1), val=1000)
    regs = density.select_top_regions(arrays["joint"], arrays["jx_edges"],
                                      arrays["jy_edges"], 32, 32, 3)
    assert len(regs) == 3  # overlap fallback guarantees n
    assert regs[0]["region"] == [32, 63, 64, 95]
    assert regs[0]["count"] == 1000
    # regions stay in frame
    for r in regs:
        xl, xr, yt, yb = r["region"]
        assert 0 <= xl <= xr < man["width"] and 0 <= yt <= yb < man["height"]


def test_density_interval_ends_monotonic_and_bounded():
    from scripts.update_query_plans import density_interval_ends
    profile = np.zeros(600)
    profile[100:200] = 5.0   # activity concentrated in minutes 100-200
    ends = density_interval_ends(profile, 60, 600, 15)
    assert len(ends) == 15
    assert all(1 <= v <= 600 for v in ends)
    assert ends == sorted(ends) and len(set(ends)) == 15
    # most endpoints should land inside/near the active window
    assert sum(90 <= v <= 210 for v in ends) >= 8

