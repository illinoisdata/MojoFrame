import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 10


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_customer = utils.get_customer_ds()
        selected_columns = ["c_custkey", "c_nationkey", "c_acctbal"]
        df_customer = df_customer[selected_columns]

        df_lineitem = utils.get_line_item_ds()
        selected_columns = [
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_returnflag",
        ]
        df_lineitem = df_lineitem[selected_columns]

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_orderkey", "o_custkey", "o_orderdate"]
        df_orders = df_orders[selected_columns]

        df_nation = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        df_nation = df_nation[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_lineitem_filtered = df_lineitem[df_lineitem["l_returnflag"] == 82.0]
        df_orders_filtered = df_orders[
            (df_orders["o_orderdate"] >= 750643200.0)
            & (df_orders["o_orderdate"] < 757382400.0)
        ]

        joined_cn_df = pd.merge(
            df_customer,
            df_nation,
            left_on="c_nationkey",
            right_on="n_nationkey",
            how="inner",
        )

        joined_oc_df = pd.merge(
            df_orders_filtered,
            joined_cn_df,
            left_on="o_custkey",
            right_on="c_custkey",
            how="inner",
        )

        joined_final = pd.merge(
            df_lineitem_filtered,
            joined_oc_df,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )

        joined_final["revenue"] = joined_final["l_extendedprice"] * (
            1 - joined_final["l_discount"]
        )

        result = (
            joined_final.groupby(["c_custkey", "c_acctbal", "n_name"])
            .agg("sum")
            .reset_index()
            .sort_values(by=["revenue"])
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
