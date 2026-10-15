from collections.abc import Sequence
from typing import Union

from search.index_based.base import base_enumerate
from search.index_based.max_dur import max_dur_enumerate
from search.index_based.max_obj import MaxObjNumEnumerator
from search.rest import (
    coalesce,
    one_pass_merge,
    stride_merge,
    stride_merge_streaming,
    stride_merge_windowed,
    vanilla_merge,
)
from utilities.time_unit import interval_to_frames, seconds_to_frames


# Merge groups. "stored" is the materializing baseline (vanilla_merge for every
# enumerator); "streaming" is bounded-memory and resolves per enumerator:
# max_dur reads stride_merge_streaming's lazily-coalesced stream (one segment
# per object run, released through a pending heap, so nothing is cut and the
# multi-pass enumerator consumes it unchanged) while max_obj needs per-object
# coalescing plus the window-cut re-join, which is stride_merge_windowed.
# Every individual merge is selectable too so any (enumerator x merge)
# combination can be measured.
_MERGE_ALIASES = {
    "stored": "vanilla",
    "materialized": "vanilla",
    "vanilla": "vanilla",
    "multi": "vanilla",  # legacy name of the stored group
    "streaming": "streaming",
    "stream": "streaming",
    "one": "streaming",  # legacy name of the streaming group
    "one_pass": "one_pass",
    "onepass": "one_pass",
    "stride": "stride",
    "stride_stream": "stride_stream",
    "ss": "stride_stream",
    "stride_win": "stride_win",
    "win": "stride_win",
}


def _streaming_merge(enu_mtd):
    """The streaming merge an enumerator is built for."""
    return "stride_stream" if enu_mtd == ("max", "dur") else "stride_win"


_TWO_TOKEN_MERGES = {"one_pass", "stride_stream", "stride_win"}


def _parse_method_str(method: str):
    """Parse a method string into ``(enu_mtd, merge_mtd, win_args)``.

    Supports the same token form as the C++ backend, e.g.
    ``"max_obj_stride_win_m1500_e2"`` -> ``(("max", "obj"), "stride_win",
    {"minima": 1500, "expend": 2})``. Without a merge token the behaviour
    matches the legacy string form (enum method only, ``stored`` merge).
    """
    tokens = method.split("_")
    # Join the two-token merge names first, so a bare "one" is never mistaken
    # for the streaming group ("onepass" is also accepted as one token).
    parts = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and "_".join(tokens[i : i + 2]) in _TWO_TOKEN_MERGES:
            parts.append("_".join(tokens[i : i + 2]))
            i += 2
        else:
            parts.append(tokens[i])
            i += 1

    enum_sub = None
    merge_mtd = "stored"
    win_args = {"minima": 3000, "expend": 2}
    for p in parts:
        if p in ("dur", "obj"):
            enum_sub = p
        elif p == "old":
            return (("max", "obj_old"), "stored", win_args)
        elif p in _MERGE_ALIASES:
            merge_mtd = _MERGE_ALIASES[p]
        elif p.startswith("m") and p[1:].isdigit():
            win_args["minima"] = int(p[1:])
        elif p.startswith("e") and p[1:].isdigit():
            win_args["expend"] = int(p[1:])
    if merge_mtd == "streaming":
        merge_mtd = _streaming_merge(("max", enum_sub or "obj"))
    return ("max", enum_sub or "obj"), merge_mtd, win_args


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
        merge_mtd = _MERGE_ALIASES.get(method[2], method[2])
        win_args = {}
    elif len(method) == 2:
        enu_mtd = tuple(method)
        merge_mtd = "stored"
        win_args = {}
    elif len(method) in (4, 5) and method[2] == "win":
        # ("max", "obj", "win", minima=, expend=) -- window tuning
        enu_mtd = tuple(method[:2])
        merge_mtd = "stride_win"
        win_args = {
            "minima": method[3] if len(method) >= 4 else 3000,
            "expend": method[4] if len(method) == 5 else 2,
        }
    else:
        raise ValueError(
            f"Invalid method format: {method}. Expected string, 2-tuple, or 3-tuple"
        )

    # The streaming group is a per-enumerator default, exactly like the C++
    # framework's ResolveMerge().
    if merge_mtd == "streaming":
        merge_mtd = _streaming_merge(enu_mtd)

    match merge_mtd:  # merge method
        case "vanilla" | "multi":
            # Stored: materialized, per-object coalesced, begin-sorted. The
            # baseline every other strategy is validated against.
            trajs = vanilla_merge(rest_idx, trajs, region, labels, dur, interval)
        case "one_pass":
            # Streaming raw per-cell stream: one object arrives as several
            # pieces. max_dur unions them on the fly; max_obj counts each piece
            # as a separate presence, so this pairing is not equivalent.
            trajs = one_pass_merge(rest_idx, trajs, region, labels, dur, interval)
        case "stride":
            trajs = stride_merge(rest_idx, trajs, region, labels, dur, interval)
        case "stride_stream":
            # Lazily-coalescing streaming merge: one segment per object run and
            # no window cuts, released as soon as no still-open trajectory began
            # earlier. Same multiset as stride_merge, but the enumerator can
            # start before the last input segment is read. Materialised here for
            # the same list contract the other strategies hand over.
            trajs = list(
                stride_merge_streaming(
                    rest_idx, trajs, region, labels, dur, interval,
                    expend=win_args.get("expend", 2),
                    minima=win_args.get("minima", 3000),
                )
            )
        case "stride_win":
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
                case "obj_old":
                    from search.index_based.max_obj_v2 import MaxObjNumEnumeratorOld
                    return MaxObjNumEnumeratorOld(trajs, labels, dur, interval)
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
