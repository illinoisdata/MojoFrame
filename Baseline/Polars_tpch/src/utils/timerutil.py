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
        pass

    def __exit__(self):
        """Stop the timer within a context manager"""
        pass

    def time(self, func: callable):
        """Timer function decorator

        Args:
            func (callable): the function to wrap
        """
        pass
