import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 12


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = [
            "l_orderkey",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipmode",
        ]
        df_lineitem = df_lineitem[selected_columns]

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_orderkey", "o_orderpriority"]
        df_orders = df_orders[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        filtered_lineitem = df_lineitem[
            (df_lineitem["l_shipmode"].isin([5.0, 6.0]))
            & (df_lineitem["l_commitdate"] < df_lineitem["l_receiptdate"])
            & (df_lineitem["l_shipdate"] < df_lineitem["l_commitdate"])
            & (df_lineitem["l_receiptdate"] >= 757382400.0)
            & (df_lineitem["l_receiptdate"] < 788918400.0)
        ]

        joined_lo_df = pd.merge(
            filtered_lineitem,
            df_orders,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )

        joined_lo_df["high_line"] = joined_lo_df["o_orderpriority"].apply(
            lambda x: 1 if x in [1.0, 2.0] else 0
        )
        joined_lo_df["low_line"] = joined_lo_df["o_orderpriority"].apply(
            lambda x: 1 if x not in [1.0, 2.0] else 0
        )

        result_df = joined_lo_df.groupby("l_shipmode").agg("sum").reset_index()

        result_df = result_df.sort_values(by="l_shipmode")

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return result_df
