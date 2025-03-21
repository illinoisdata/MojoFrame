import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 2


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_region = utils.get_region_ds()
        df_nation = utils.get_nation_ds()
        df_supp = utils.get_supplier_ds()
        df_part = utils.get_part_ds()
        df_partsupp = utils.get_part_supp_ds()

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_region_filtered = df_region[(df_region["r_name"] == 35796.0)]

        df_part_filtered_outer = df_part[
            (df_part["p_type"].str.endswith("BRASS")) & (df_part["p_size"] == 15)
        ]
        joined_region_nation = pd.merge(
            df_region_filtered,
            df_nation,
            left_on="r_regionkey",
            right_on="n_regionkey",
            how="inner",
        )
        joined_nation_supplier = pd.merge(
            joined_region_nation,
            df_supp,
            left_on="n_nationkey",
            right_on="s_nationkey",
            how="inner",
        )
        joined_partsupp_supplier = pd.merge(
            joined_nation_supplier,
            df_partsupp,
            left_on="s_suppkey",
            right_on="ps_suppkey",
            how="inner",
        )
        joined_df_final = pd.merge(
            joined_partsupp_supplier,
            df_part_filtered_outer,
            left_on="ps_partkey",
            right_on="p_partkey",
            how="inner",
        )

        min_supplycost = (
            joined_df_final.groupby("ps_partkey")["ps_supplycost"].min().reset_index()
        )
        min_supplycost.rename(
            columns={
                "ps_partkey": "ps_partkey_grp",
                "ps_supplycost": "ps_supplycost_min",
            },
            inplace=True,
        )

        min_supplycost_final = pd.merge(
            joined_df_final,
            min_supplycost,
            left_on="ps_partkey",
            right_on="ps_partkey_grp",
            how="inner",
        )
        min_supplycost_final = (
            min_supplycost_final[
                min_supplycost_final["ps_supplycost"]
                == min_supplycost_final["ps_supplycost_min"]
            ]
        ).sort_values(by=["s_acctbal", "n_name", "s_name", "ps_partkey"])

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return min_supplycost_final
