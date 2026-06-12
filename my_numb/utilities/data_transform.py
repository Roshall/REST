from numba import njit, jit_module


def unpack_i32(packed):
    return packed >> 32, packed & 0xFFFFFFFF


def pack_i16(point):
    return point[1] << 16 | point[0]


def pack_i32(high, low):
    return high << 32 | low


jit_module(nopython=True, cache=True)