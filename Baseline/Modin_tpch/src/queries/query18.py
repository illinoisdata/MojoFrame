import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 18


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = ["l_orderkey", "l_quantity"]
        df_lineitem = df_lineitem[selected_columns]

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]
        df_orders = df_orders[selected_columns]

        df_customer = utils.get_customer_ds()
        selected_columns = ["c_custkey"]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        joined_lo_df = pd.merge(
            df_lineitem, df_orders, left_on="l_orderkey", right_on="o_orderkey"
        )

        lineitem_grouped = df_lineitem.groupby("l_orderkey", as_index=False).agg(
            l_quantity_sum=("l_quantity", "sum")
        )

        lineitem_filtered = lineitem_grouped[lineitem_grouped["l_quantity_sum"] > 300]

        filtered_lo_df = pd.merge(lineitem_filtered, joined_lo_df, on="l_orderkey")

        joined_oc_df = pd.merge(
            filtered_lo_df, df_customer, left_on="o_custkey", right_on="c_custkey"
        )

        grouped = joined_oc_df.groupby(
            ["c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"], as_index=False
        ).agg(quantity_sum_grouped=("l_quantity", "sum"))

        result = grouped.sort_values(
            by=["o_totalprice", "o_orderdate"], ascending=[False, True]
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
