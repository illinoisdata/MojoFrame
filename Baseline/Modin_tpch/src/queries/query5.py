import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 5


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_region = utils.get_region_ds()
        selected_columns = ["r_regionkey", "r_name"]
        df_region = df_region[selected_columns]

        df_nation = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_regionkey", "n_name"]
        df_nation = df_nation[selected_columns]

        df_cust = utils.get_customer_ds()
        selected_columns = ["c_custkey", "c_nationkey"]
        df_cust = df_cust[selected_columns]

        df_lineitem = utils.get_line_item_ds()
        selected_columns = ["l_orderkey", "l_extendedprice", "l_discount", "l_suppkey"]
        df_lineitem = df_lineitem[selected_columns]
        df_lineitem["revenue"] = df_lineitem["l_extendedprice"] * (
            1 - df_lineitem["l_discount"]
        )

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_custkey", "o_orderkey", "o_orderdate"]
        df_orders = df_orders[selected_columns]

        df_supp = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_nationkey"]
        df_supp = df_supp[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_region_filtered = df_region[df_region["r_name"] == 43715.0]
        df_orders_filtered = df_orders[
            (df_orders["o_orderdate"] >= 757382400.0)
            & (df_orders["o_orderdate"] < 788918400.0)
        ]
        df_joined_co = pd.merge(
            df_cust,
            df_orders_filtered,
            left_on="c_custkey",
            right_on="o_custkey",
            how="inner",
        )
        df_joined_l = pd.merge(
            df_lineitem,
            df_joined_co,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )
        df_joined_supp = pd.merge(
            df_joined_l, df_supp, left_on="l_suppkey", right_on="s_suppkey", how="inner"
        )
        df_joined_supp = df_joined_supp[
            df_joined_supp["c_nationkey"] == df_joined_supp["s_nationkey"]
        ]
        df_joined_nation = pd.merge(
            df_joined_supp,
            df_nation,
            left_on="s_nationkey",
            right_on="n_nationkey",
            how="inner",
        )
        df_joined_final = pd.merge(
            df_joined_nation,
            df_region_filtered,
            left_on="n_regionkey",
            right_on="r_regionkey",
            how="inner",
        )
        grouped_sum = (
            df_joined_final.groupby("n_name")["revenue"]
            .sum()
            .reset_index()
            .sort_values(by="revenue")
        )

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return grouped_sum
