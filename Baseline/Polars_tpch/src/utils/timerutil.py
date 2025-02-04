import time
from functools import wraps


class TPCHTimer:
    """Custom class used for timing various stages of
    the TPC-H benchmarks
    """

    times = {}

    def __init__(self, name: str, logging=False):
        self.start_time = time.perf_counter()
        self.name = name
        self.logging = logging

    def __enter__(self):
        """Start the timer within a context manager

        Returns:
            TPCHTimer: The timer created
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Stop the timer within a context manager"""
        time_elapsed = time.perf_counter() - self.start_time

        if self.name in TPCHTimer.times:
            TPCHTimer.times[self.name] += time_elapsed
        else:
            TPCHTimer.times[self.name] = time_elapsed

        if self.logging:
            print(f"{self.name} ran for {time_elapsed:.8f}s")

    def __call__(self, func: callable):
        """Timer function wrapper

        Args:
            func (callable): the function to wrap
        """

        @wraps(func)
        def timed_func(*args, **kwargs):
            self.start_time = time.perf_counter()

            func(*args, **kwargs)

            time_elapsed = time.perf_counter() - self.start_time

            if self.name in TPCHTimer.times:
                TPCHTimer.times[self.name] += time_elapsed
            else:
                TPCHTimer.times[self.name] = time_elapsed

            if self.logging:
                print(f"{self.name} ran for {time_elapsed:.8f}s")

        return timed_func
