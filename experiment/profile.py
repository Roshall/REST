import contextlib
from time import perf_counter as now


class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0):
        self.t = t

    def __enter__(self):
        self.start = now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dt = now() - self.start
        self.t += self.dt
