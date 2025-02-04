import time


class TPCHTimer:
    """Custom class used for timing various stages of
    the TPC-H benchmarks
    """

    times = {}

    def __init__(self, name: str):
        self.start_time = time.perf_counter()
        self.name = name

    def __enter__(self, name: str):
        """Start the timer within a context manager

        Args:
            name (str): the process name that you are
            timing
        """
        self.start_time = time.perf_counter()
        self.name = name

    def __exit__(self):
        """Stop the timer within a context manager"""
        time_elapsed = time.perf_counter() - self.start_time

        if self.name in TPCHTimer.times:
            TPCHTimer.times[self.name] += time_elapsed
        else:
            TPCHTimer.times[self.name] = time_elapsed

    def time(self, func: callable, name: str):
        """Timer function wrapper

        Args:
            func (callable): the function to wrap
            name (str): the process name to save
        """
        pass
