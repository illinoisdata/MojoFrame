import time

from src.utils.timerutil import TPCHTimer


def test_timer_context(capsys):
    with capsys.disabled():
        for i in range(10):
            with TPCHTimer(f"test_{i}", logging=True):
                time.sleep(0.1)
