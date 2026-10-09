from collections.abc import Sequence
from typing import Union

from search.index_based.base import base_enumerate
from search.index_based.max_dur import max_dur_enumerate
from search.index_based.max_obj import MaxObjNumEnumerator
from search.rest import (
    coalesce,
    one_pass_merge,
    stride_merge,
    stride_merge_windowed,
    vanilla_merge,
)
from utilities.time_unit import interval_to_frames, seconds_to_frames


def _parse_method_str(method: str):
    """Parse a method string into ``(enu_mtd, merge_mtd, win_args)``.

    Supports the same token form as the C++ backend, e.g.
    ``"max_obj_win_m1500_e2"`` -> ``(("max", "obj"), "win", {"minima": 1500,
    "expend": 2})``. Without a ``win`` token the behaviour matches the legacy
    string form (enum method only, ``multi`` merge).
    """
    parts = method.split("_")
    enum_sub = None
    merge_mtd = "multi"
    win_args = {"minima": 3000, "expend": 2}
    if "win" in parts:
        for p in parts:
            if p in ("dur", "obj"):
                enum_sub = p
            elif p == "win":
                merge_mtd = "win"
            elif p.startswith("m") and p[1:].isdigit():
                win_args["minima"] = int(p[1:])
            elif p.startswith("e") and p[1:].isdigit():
                win_args["expend"] = int(p[1:])
        return ("max", enum_sub or "obj"), merge_mtd, win_args
    # Legacy string form: a bare enum method name (e.g. "base", "max").
    return (method,), "multi", win_args


def index_based_framework(
    data_pack,
    region,
    labels,
    duration: Sequence[int],
    interval: Sequence[int],
    *,
    method: Union[str, tuple] = ("max", "dur", "multi"),
    fps=None,
):
    """
    Framework for index-based trajectory pattern mining.

    Args:
        data_pack: A tuple of (rest_idx, trajs) where rest_idx is the index
                   and trajs is the trajectory data.
        region: The spatial region to query.
        labels: A mapping of label to required count.
        duration: A sequence of durations in seconds. The first element is used.
        interval: A sequence of [start, end] seconds.
        method: The search method configuration. Can be:
                - A tuple of 3 elements: (enum_method, sub_method, merge_method)
                  e.g., ('max', 'dur', 'multi') or ('max', 'obj', 'one')
                - A tuple of 2 elements: (enum_method, sub_method), defaults to 'multi' merge
                  e.g., ('max', 'dur')
                - A string: enum_method only, defaults to 'multi' merge and no sub-method
                  e.g., 'base'
        fps: Frames per second used to convert the query to frames; defaults to
             ``cfg.DATA.FPS``. Pass 1 for frame-valued input.

    Returns:
        A list of co-movement patterns found in the specified region and time interval.
    """
    rest_idx, trajs = data_pack

    if not duration:
        raise ValueError("duration must be a non-empty sequence")

    dur = seconds_to_frames(duration[0], fps, minimum=1)
    interval = interval_to_frames(interval, fps)

    for c in labels:
        if c not in rest_idx:
            return []

    # Parse method configuration
    if isinstance(method, str):
        enu_mtd, merge_mtd, win_args = _parse_method_str(method)
    elif len(method) == 3:
        enu_mtd = tuple(method[:2])
        merge_mtd = method[2]
        win_args = {}
    elif len(method) == 2:
        enu_mtd = tuple(method)
        merge_mtd = "multi"
        win_args = {}
    elif len(method) in (4, 5) and method[2] == "win":
        # ("max", "obj", "win", minima=, expend=) -- window tuning
        enu_mtd = tuple(method[:2])
        merge_mtd = "win"
        win_args = {
            "minima": method[3] if len(method) >= 4 else 3000,
            "expend": method[4] if len(method) == 5 else 2,
        }
    else:
        raise ValueError(
            f"Invalid method format: {method}. Expected string, 2-tuple, or 3-tuple"
        )

    match merge_mtd:  # merge method
        case "multi":
            trajs = vanilla_merge(rest_idx, trajs, region, labels, dur, interval)
        case "one":
            match enu_mtd:
                case ("max", "dur"):
                    trajs = one_pass_merge(rest_idx, trajs, region, labels, dur, interval)
                case _:
                    trajs = stride_merge(rest_idx, trajs, region, labels, dur, interval)
        case "win":
            # Bounded-memory windowed streaming merge. It cuts a trajectory
            # that crosses a window boundary into a head plus a touching tail,
            # and MaxObjNumEnumerator re-joins them, so the pattern set equals
            # `stride_merge`'s while memory stays bounded by the window.
            # Materialised here for the same list contract the other merge
            # strategies hand to the enumerator.
            trajs = list(
                stride_merge_windowed(
                    rest_idx, trajs, region, labels, dur, interval,
                    expend=win_args.get("expend", 2),
                    minima=win_args.get("minima", 3000),
                )
            )
        case _:
            raise ValueError(f"Unknown merge method {merge_mtd} in {method}")

    match enu_mtd:
        case ("base",):
            return base_enumerate(trajs, labels, (dur,), interval)
        case ("max", sub_mtd):
            match sub_mtd:
                case "obj":
                    return MaxObjNumEnumerator(trajs, labels, dur, interval)
                case "dur":
                    return max_dur_enumerate(
                        trajs, labels, dur, interval, mtd=merge_mtd
                    )
                case _:
                    raise ValueError(
                        f"Unknown sub-method '{sub_mtd}' for 'max' enumeration in {method}"
                    )
        case _:
            raise ValueError(f"Unknown enumeration method {enu_mtd} in {method}")
