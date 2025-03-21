import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 15


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_supplier = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_name"]
        df_supplier = df_supplier[selected_columns]

        df_lineitem = utils.get_line_item_ds()
        selected_columns = ["l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"]
        df_lineitem = df_lineitem[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_lineitem = df_lineitem[
            (df_lineitem["l_shipdate"] >= 820454400.0)
            & (df_lineitem["l_shipdate"] < 828230400.0)
        ]

        df_lineitem["total_revenue"] = df_lineitem["l_extendedprice"] * (
            1 - df_lineitem["l_discount"]
        )

        joined_ls_df = pd.merge(
            df_lineitem, df_supplier, left_on="l_suppkey", right_on="s_suppkey"
        )

        supplier_revenue = (
            joined_ls_df.groupby(["l_suppkey", "s_name"]).agg("sum").reset_index()
        )

        max_revenue = supplier_revenue["total_revenue"].max()

        supplier_revenue = supplier_revenue[
            supplier_revenue["total_revenue"] == max_revenue
        ]

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return supplier_revenue
