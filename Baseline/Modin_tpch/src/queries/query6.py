import tracemalloc

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 6


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    start_timestamp = 757382400.0
    end_timestamp = 788918400.0
    max_quantity = 24

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        lineitem_df = utils.get_line_item_ds()

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        filtered_df_new = lineitem_df[
            (lineitem_df["l_shipdate"] >= start_timestamp)
            & (lineitem_df["l_shipdate"] < end_timestamp)
            & (lineitem_df["l_discount"] >= (0.05))
            & (lineitem_df["l_discount"] <= (0.07))
            & (lineitem_df["l_quantity"] < max_quantity)
        ]
        revenue_new = (
            filtered_df_new["l_extendedprice"] * filtered_df_new["l_discount"]
        ).sum()

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return revenue_new
