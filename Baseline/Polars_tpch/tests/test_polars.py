from src.utils.timerutil import TPCHTimer


def test_timer_context():
    with TPCHTimer("test", True):
        for i in range(1000):
            continue
