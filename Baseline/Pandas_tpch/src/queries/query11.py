import tracemalloc

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 11


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    var_n_name = 52342.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        partsupp_ds = utils.get_part_supp_ds()
        supplier_ds = utils.get_supplier_ds()
        nation_ds = utils.get_nation_ds()

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
