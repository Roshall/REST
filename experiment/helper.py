from utilities.box2D import Box2D as _Box


def abbr_map(data):
    return {name[0]: name for name in data}


def full_name(data_m, desideratum):
    return list(f'{name}10h.{data_m[name]['extension']}' for name in
                desideratum)


def gather_all(opt, data_m):
    if opt == 'all':
        data_m = data_m.copy()
        data_m.pop('default', None)
        return list(data_m.values())
    elif opt == 'default':
        return [opt]
    else:
        cand = set(opt)
        if not cand.issubset(data_m.keys()):
            raise KeyError(f'{cand - data_m.keys()} of ({opt}) is not found in {data_m.keys()}')
        return list(data_m[abbr] for abbr in cand)


# =========== for searching methods ==========
#
#               |-both(b)-|
#               /          \
#     sliding(s)             index(i)
#    /      \               /    |    \
# naive(v) state(t) max_dur(d) base(a) max_obj(j)
#                    /    \
#           one_pass(o)  multipass(u)
_mth_tree = {
    'b': ('', ['i', 's']), 's': ('sliding', ['t', 'v']), 'i': ('index', ['d', 'j', 'a']), 'v': ("naive", None, ['s']),
    't': ("state", None, ['s']), 'j': ("max_obj", None, ['i']), 'a': ("base", None, ['i']),
    'd': ("max_dur", ['u', 'o']), 'o': ("one", None, ['i', 'j']), 'u': ("multi", None, ['i', 'j'])
}


def _traverse_node(node, visited):
    if (children := node[1]) is None:
        name = [_mth_tree[abbr][0] for abbr in node[2]]
        name.append(node[0])
        visited.append('_'.join(name))
    else:
        for child in children:
            _traverse_node(_mth_tree[child], visited)


def which_mth(opt):
    visited = []
    for m in opt:
        _traverse_node(_mth_tree[m], visited)
    return visited


# =========== for query plans ==========
def default_plan(default):
    return {'region': default['bbox_scale_idx'],  # wait for dataset name
            'pattern': dict(default['pattern']),
            'duration': [default['duration_left'], 100000],
            'interval': [0, default['interval_right_idx']]
            }  # wait for
    # dataset name


def refine_interval(plan, plan_meta):
    idx = plan['interval'][1]
    plan['interval'][1] = plan_meta['interval']['10h'][idx] * 30 * 60


def refine_region(plan, reg_meta):
    idx = plan['region']
    plan['region'] = reg_meta[idx]


def reset_region(plan, reg, conf):
    plan['region'] = _Box(reg, conf)
