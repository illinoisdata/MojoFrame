import time


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

    def __exit__(self):
        """Stop the timer within a context manager"""
        time_elapsed = time.perf_counter() - self.start_time

        if self.name in TPCHTimer.times:
            TPCHTimer.times[self.name] += time_elapsed
        else:
            TPCHTimer.times[self.name] = time_elapsed

        if self.logging:
            print(time_elapsed)

    def time(self, func: callable, name: str):
        """Timer function wrapper

        Args:
            func (callable): the function to wrap
            name (str): the process name to save
        """
        pass
