"""Conversion between the query time unit (seconds) and internal frames.

Queries are authored in seconds: ``duration`` and both ``interval`` endpoints.
Everything below the framework boundary (merges, enumerators, the C++ engine)
works in frames, so the conversion happens exactly once, at the boundary.

The frame rate comes from ``configs.cfg.DATA.FPS``. Pass ``fps`` explicitly to
override it, which is what frame-level tests do (``fps=1``).
"""


def seconds_to_frames(value, fps=None, minimum=None):
    """Convert a time value in seconds to frames.

    :param value: seconds
    :param fps: frames per second; defaults to ``cfg.DATA.FPS``
    :param minimum: optional lower clamp, e.g. ``1`` for a duration
    :return: frame count as ``int``
    """
    if fps is None:
        from configs import cfg  # lazy so cfg overrides are honoured

        fps = cfg.DATA.FPS
    frames = int(round(value * fps))
    if minimum is not None:
        frames = max(frames, minimum)
    return frames


def interval_to_frames(interval, fps=None):
    """Convert a ``[start, end]`` interval in seconds to frames.

    :param interval: two-element sequence in seconds
    :param fps: frames per second; defaults to ``cfg.DATA.FPS``
    :return: ``[start, end]`` in frames
    """
    if fps is None:
        from configs import cfg  # lazy so cfg overrides are honoured

        fps = cfg.DATA.FPS
    lo = seconds_to_frames(interval[0], fps)
    hi = seconds_to_frames(interval[1], fps)
    if lo >= hi:
        raise ValueError(
            f"interval {tuple(interval)} (seconds) spans less than one frame "
            f"at {fps} fps"
        )
    return [lo, hi]
