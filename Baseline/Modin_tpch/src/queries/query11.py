import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 11


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_partsupp = utils.get_part_supp_ds()
        selected_columns = ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]
        df_partsupp = df_partsupp[selected_columns]

        df_supplier = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_nationkey"]
        df_supplier = df_supplier[selected_columns]

        df_nation = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        df_nation = df_nation[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_nation_filtered = df_nation[df_nation["n_name"] == 52342.0]

        joined_sn_df = pd.merge(
            df_supplier,
            df_nation_filtered,
            left_on="s_nationkey",
            right_on="n_nationkey",
            how="inner",
        )
        joined_pss_df = pd.merge(
            df_partsupp,
            joined_sn_df,
            left_on="ps_suppkey",
            right_on="s_suppkey",
            how="inner",
        )

        joined_pss_df["value"] = (
            joined_pss_df["ps_supplycost"] * joined_pss_df["ps_availqty"]
        )

        joined_pss_df["value_percent"] = joined_pss_df["value"] * 0.00001

        grouped_df = joined_pss_df.groupby("ps_partkey").agg("sum").reset_index()

        value_percent_sum = joined_pss_df["value_percent"].sum()

        filtered_df = grouped_df[grouped_df["value"] > value_percent_sum]
        filtered_df = filtered_df.sort_values(by=["value"])

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return filtered_df
