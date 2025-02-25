import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 1


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    VAR1 = 904608000.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        lineitem_df: pd.DataFrame = utils.get_line_item_ds()

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        filtered_df = lineitem_df[lineitem_df["l_shipdate"] <= VAR1]

        result = (
            filtered_df.groupby(["l_returnflag", "l_linestatus"])
            .agg(
                sum_qty=("l_quantity", "sum"),
                sum_price=("l_extendedprice", "sum"),
                sum_disc=("l_discount", "sum"),
                sum_date=("l_shipdate", "sum"),
                avg_qty=("l_quantity", "mean"),
                avg_price=("l_extendedprice", "mean"),
                avg_disc=("l_discount", "mean"),
                avg_date=("l_shipdate", "mean"),
                count_order=("l_quantity", "size"),  # using size to count entries
            )
            .reset_index()
        )

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return result
