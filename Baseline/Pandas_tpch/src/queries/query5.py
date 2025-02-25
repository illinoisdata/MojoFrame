import tracemalloc

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 5


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    var1 = 43715.0
    var2 = 757382400.0
    var3 = 788918400.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        region_ds = utils.get_region_ds()
        nation_ds = utils.get_nation_ds()
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        supplier_ds = utils.get_supplier_ds()

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
