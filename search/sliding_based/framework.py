from functools import partial

from search.verifier import obj_verify, df_filter
from search.sliding_based.naive import naive_slider
from search.sliding_based.state import state_slider
from utilities.data_preprocessing import group_by_frame
from utilities.time_unit import interval_to_frames, seconds_to_frames


def sliding_framework(df, region, labels, duration, interval, *, method='naive', fps=None):
    """
    Args:
        duration: A sequence of durations in seconds. The first element is used.
        interval: A sequence of [start, end] seconds. The interval is treated as
            half-open ``[start, end)``: a frame at ``end`` is excluded. This
            matches the canonical co-movement contract shared with the index
            path and the reference oracle.
        fps: Frames per second used to convert the query to frames; defaults to
             ``cfg.DATA.FPS``. Pass 1 for frame-valued input.
    """
    dur = seconds_to_frames(duration[0], fps, minimum=1)
    interval = interval_to_frames(interval, fps)
    frames = group_by_frame(df, interval)
    obj_verifier = partial(obj_verify, labels)
    dfilter = partial(df_filter, reg_verifier=region.enclose, target_label=labels.keys())
    match method:
        case 'naive' | 'base':
            # 'base' is the C++ spelling of the naive slider; accept both so the
            # two implementations expose the same method names.
            slider = naive_slider
        case 'state':
            slider = state_slider
        case _:
            raise NotImplementedError(f'Unknown method: {method}')

    return slider(frames, dur, obj_verifier, dfilter)
