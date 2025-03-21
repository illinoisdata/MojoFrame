import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 20


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = ["l_partkey", "l_suppkey", "l_shipdate", "l_quantity"]
        df_lineitem = df_lineitem[selected_columns]

        df_part = utils.get_part_ds()
        selected_columns = ["p_partkey", "p_name"]
        df_part = df_part[selected_columns]

        df_partsupp = utils.get_part_supp_ds()
        selected_columns = ["ps_suppkey", "ps_partkey", "ps_availqty"]
        df_partsupp = df_partsupp[selected_columns]

        df_nation = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        df_nation = df_nation[selected_columns]

        df_supplier = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_nationkey", "s_name"]
        df_supplier = df_supplier[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        filtered_part = df_part[df_part["p_name"].str.startswith("forest")]

        joined_psp = pd.merge(
            df_partsupp,
            filtered_part,
            left_on="ps_partkey",
            right_on="p_partkey",
            how="inner",
        )

        filtered_lineitem = df_lineitem[
            (df_lineitem["l_shipdate"] >= 757382400.0)
            & (df_lineitem["l_shipdate"] < 788918400.0)
        ]

        lineitem_agg = filtered_lineitem.groupby(
            ["l_partkey", "l_suppkey"], as_index=False
        ).agg("sum")

        joined_lineitem_psp = pd.merge(
            joined_psp,
            lineitem_agg,
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
            how="inner",
        )

        filtered_psp = joined_lineitem_psp[
            joined_lineitem_psp["ps_availqty"] > 0.5 * joined_lineitem_psp["l_quantity"]
        ]

        joined_supplier = pd.merge(
            filtered_psp,
            df_supplier,
            left_on="ps_suppkey",
            right_on="s_suppkey",
            how="inner",
        )

        joined_nation = pd.merge(
            joined_supplier, df_nation, left_on="s_nationkey", right_on="n_nationkey"
        )

        filtered_nation = joined_nation[joined_nation["n_name"] == 35480.0]

        result = filtered_nation.sort_values(by="s_name")

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return result
