import tracemalloc

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 16


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        part_ds = utils.get_part_ds()
        part_supp_ds = utils.get_part_supp_ds()
        supp_ds = utils.get_supplier_ds()

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
