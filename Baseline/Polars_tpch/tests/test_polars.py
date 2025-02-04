import time

from src.utils.timerutil import TPCHTimer


def test_timer_context(capsys):
    with capsys.disabled():
        for i in range(10):
            with TPCHTimer(f"test_{i}", logging=True):
                time.sleep(0.1)


def test_timer_decorator(capsys):
    with capsys.disabled():

        @TPCHTimer("test_function", logging=True)
        def test_function():
            time.sleep(0.1)

        for i in range(100):
            test_function()
