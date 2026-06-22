from search.index_based.base import base_enumerate
from search.index_based.max_dur import max_dur_enumerate
from search.index_based.max_obj import MaxObjNumEnumerator
from search.rest import vanilla_merge, one_pass_merge, stride_merge


def index_based_framework(data_pack, region, labels, duration, interval, *, method=('max', 'dur', 'multi')):
    rest_idx, trajs = data_pack
    dur = duration[0]
    for c in labels:
        if c not in rest_idx:
            return []

    if len(method) == 3:
        *enu_mtd, merge_mtd = method
    else:
        enu_mtd, merge_mtd = method, 'multi'

    match merge_mtd:  # merge method
        case 'multi':
            trajs = vanilla_merge(rest_idx, trajs, region, labels, dur, interval)
        case 'one':
            match enu_mtd:
                case ['max', 'obj']:
                    trajs = one_pass_merge(rest_idx, trajs, region, labels, dur, interval)
                case _:
                    trajs = stride_merge(rest_idx, trajs, region, labels, dur, interval)
        case _:
            raise ValueError(f'Unknown merge method {merge_mtd} in {method}')

    match enu_mtd:
        case ['base']:
            return base_enumerate(trajs, labels, duration, interval)
        case ['max', sub_mtd]:
            match sub_mtd:
                case 'obj':
                    return MaxObjNumEnumerator(trajs, labels, dur, interval)
                case 'dur':
                    return max_dur_enumerate(trajs, labels, dur, interval, mtd=merge_mtd)
                case _:
                    raise ValueError(f'Unknown search method {method}')
        case _:
            raise ValueError(f'Unknown search method {method}')
