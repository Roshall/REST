from collections.abc import Sequence
from typing import Union

from search.index_based.base import base_enumerate
from search.index_based.max_dur import max_dur_enumerate
from search.index_based.max_obj import MaxObjNumEnumerator
from search.rest import coalesce, one_pass_merge, stride_merge, vanilla_merge
from utilities.time_unit import interval_to_frames, seconds_to_frames


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
        enu_mtd = (method,)
        merge_mtd = "multi"
    elif len(method) == 3:
        enu_mtd = tuple(method[:2])
        merge_mtd = method[2]
    elif len(method) == 2:
        enu_mtd = tuple(method)
        merge_mtd = "multi"
    else:
        raise ValueError(
            f"Invalid method format: {method}. Expected string, 2-tuple, or 3-tuple"
        )

    match merge_mtd:  # merge method
        case "multi":
            trajs = vanilla_merge(rest_idx, trajs, region, labels, dur, interval)
        case "one":
            match enu_mtd:
                case ("max", "obj"):
                    # one_pass_merge streams per-cell segments without
                    # coalescing them per object, so the enumerator would see a
                    # single object as several pieces. Absorb first so all merge
                    # strategies hand the enumerator the same segment multiset.
                    trajs = coalesce(
                        one_pass_merge(rest_idx, trajs, region, labels, dur, interval),
                        dur,
                    )
                case _:
                    trajs = stride_merge(rest_idx, trajs, region, labels, dur, interval)
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
