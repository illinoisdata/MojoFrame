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

        for i in range(10):
            test_function()
            time.sleep(0.5)


def test_timer_multi(capsys):
    with capsys.disabled():

        @TPCHTimer("test_function_0.1", logging=True)
        def test_function():
            time.sleep(0.1)

        @TPCHTimer("test_function_0.3", logging=True)
        def test_function_two():
            time.sleep(0.3)

        for i in range(3):
            test_function()
            test_function_two()
            with TPCHTimer("context_manager", logging=True):
                time.sleep(0.5)

        print(TPCHTimer.times)
