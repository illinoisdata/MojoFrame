import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 4


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = [
            "l_orderkey",
            "l_quantity",
            "l_extendedprice",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
        ]
        df_lineitem = df_lineitem[selected_columns]

        df_orders = utils.get_orders_ds()
        selected_columns = [
            "o_custkey",
            "o_orderkey",
            "o_orderdate",
            "o_shippriority",
            "o_orderpriority",
        ]
        df_orders = df_orders[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_orders_filtered = df_orders[
            (df_orders["o_orderdate"] >= 741484800.0)
            & (df_orders["o_orderdate"] < 749433600.0)
        ]

        df_lineitem_filtered = df_lineitem[
            df_lineitem["l_commitdate"] < df_lineitem["l_receiptdate"]
        ]

        df_merged = pd.merge(
            df_orders_filtered,
            df_lineitem_filtered,
            left_on="o_orderkey",
            right_on="l_orderkey",
            how="inner",
        )

        grouped_count = df_merged.groupby("o_orderpriority").size()

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return grouped_count
