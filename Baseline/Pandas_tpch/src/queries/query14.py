import tracemalloc

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 14


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    VAR1 = 809913600.0
    VAR2 = 812505600.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        line_item_ds = utils.get_line_item_ds()
        part_ds = utils.get_part_ds()

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        ...

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak
