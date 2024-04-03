from functools import partial

from search.rest import df_filter
from search.verifier import obj_verify
from search.baseline.naive import base_slider
from search.baseline.state import state_slider
from utilities.data_preprocessing import group_by_frame


def sliding_framework(df, region, labels, duration, interval, *, method='base'):
    dur = duration[0]
    frames = group_by_frame(df, interval)
    obj_verifier = partial(obj_verify, labels)
    dfilter = partial(df_filter, reg_verifier=region.enclose, target_label=labels.keys())
    match method:
        case 'base':
            slider = base_slider
        case 'state':
            slider = state_slider
        case _:
            raise NotImplementedError(f'Unknown method: {method}')

    return slider(frames, dur, obj_verifier, dfilter)
