import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 9


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_part = utils.get_part_ds()
        selected_columns = ["p_partkey", "p_name"]
        df_part = df_part[selected_columns]

        df_supplier = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_nationkey"]
        df_supplier = df_supplier[selected_columns]

        df_lineitem = utils.get_line_item_ds()
        selected_columns = [
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
            "l_suppkey",
            "l_partkey",
        ]
        df_lineitem = df_lineitem[selected_columns]

        df_partsupp = utils.get_part_supp_ds()
        selected_columns = ["ps_partkey", "ps_suppkey", "ps_supplycost"]
        df_partsupp = df_partsupp[selected_columns]

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_orderkey", "o_orderdate"]
        df_orders = df_orders[selected_columns]

        df_nation = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        df_nation = df_nation[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_part_filtered = df_part[
            df_part["p_name"].str.contains("green", case=True, na=False)
        ]

        joined_sn_df = pd.merge(
            df_supplier,
            df_nation,
            left_on="s_nationkey",
            right_on="n_nationkey",
            how="inner",
        )

        joined_ls_df = pd.merge(
            df_lineitem,
            joined_sn_df,
            left_on="l_suppkey",
            right_on="s_suppkey",
            how="inner",
        )

        joined_lp_df = pd.merge(
            joined_ls_df,
            df_part_filtered,
            left_on="l_partkey",
            right_on="p_partkey",
            how="inner",
        )

        joined_lps_df = pd.merge(
            joined_lp_df,
            df_partsupp,
            left_on=["l_partkey", "l_suppkey"],
            right_on=["ps_partkey", "ps_suppkey"],
            how="inner",
        )

        joined_final = pd.merge(
            joined_lps_df,
            df_orders,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )

        joined_final["sum_profit"] = (
            joined_final["l_extendedprice"] * (1 - joined_final["l_discount"])
        ) - (joined_final["ps_supplycost"] * joined_final["l_quantity"])

        joined_final["o_orderdate"] = (
            1970.0 + (joined_final["o_orderdate"] / 31536000.0)
        ).round()

        result = (
            joined_final.groupby(["n_name", "o_orderdate"])
            .agg("sum")
            .reset_index()
            .sort_values(by=["n_name", "o_orderdate"])
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
