import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 7


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_nation1 = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        df_nation1 = df_nation1[selected_columns]
        df_nation1.rename(
            columns={"n_nationkey": "n_nationkey1", "n_name": "n_name1"}, inplace=True
        )

        df_nation2 = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        df_nation2 = df_nation2[selected_columns]
        df_nation2.rename(
            columns={"n_nationkey": "n_nationkey2", "n_name": "n_name2"}, inplace=True
        )

        df_cust = utils.get_customer_ds()
        selected_columns = ["c_custkey", "c_nationkey"]
        df_cust = df_cust[selected_columns]

        df_lineitem = utils.get_line_item_ds()
        selected_columns = [
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_suppkey",
            "l_shipdate",
        ]
        df_lineitem = df_lineitem[selected_columns]
        df_lineitem["revenue"] = df_lineitem["l_extendedprice"] * (
            1 - df_lineitem["l_discount"]
        )

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_custkey", "o_orderkey"]
        df_orders = df_orders[selected_columns]

        df_supp = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_nationkey"]
        df_supp = df_supp[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        # Filter orders between 1994-01-01 and 1995-01-01 using float Unix time
        df_lineitem = df_lineitem[
            (df_lineitem["l_shipdate"] >= 788918400.0)
            & (df_lineitem["l_shipdate"] <= 852076800.0)
        ]

        # Perform the joins with filtered DataFrames
        df_joined_ls = pd.merge(
            df_supp, df_lineitem, left_on="s_suppkey", right_on="l_suppkey", how="inner"
        )

        df_joined_o = pd.merge(
            df_joined_ls,
            df_orders,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )

        df_joined_c = pd.merge(
            df_joined_o, df_cust, left_on="o_custkey", right_on="c_custkey", how="inner"
        )

        df_joined_n1 = pd.merge(
            df_joined_c,
            df_nation1,
            left_on="s_nationkey",
            right_on="n_nationkey1",
            how="inner",
        )
        df_shipping = pd.merge(
            df_joined_n1,
            df_nation2,
            left_on="c_nationkey",
            right_on="n_nationkey2",
            how="inner",
        )

        df_shipping = df_shipping[
            ((df_shipping["n_name1"] == 38075.0) & (df_shipping["n_name2"] == 52342.0))
            | (
                (df_shipping["n_name1"] == 52342.0)
                & (df_shipping["n_name2"] == 38075.0)
            )
        ]
        df_shipping["l_shipdate"] = (
            1970.0 + (df_shipping["l_shipdate"] / 31536000.0)
        ).round()

        result = (
            df_shipping.groupby(["n_name1", "n_name2", "l_shipdate"])
            .agg("sum")
            .reset_index()
            .sort_values(by=["n_name1", "n_name2", "l_shipdate"])
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
