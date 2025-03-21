import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 17


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = ["l_partkey", "l_quantity", "l_extendedprice"]
        df_lineitem = df_lineitem[selected_columns]

        df_part = utils.get_part_ds()
        selected_columns = ["p_partkey", "p_brand", "p_container"]
        df_part = df_part[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_part_filtered = df_part[
            (df_part["p_brand"] == "Brand#23") & (df_part["p_container"] == "MED BOX")
        ]

        lineitem_part_joined = pd.merge(
            df_lineitem,
            df_part_filtered,
            left_on="l_partkey",
            right_on="p_partkey",
            how="inner",
        )

        lineitem_avg_quantity = lineitem_part_joined.groupby(
            "l_partkey", as_index=False
        ).agg(avg_quantity=("l_quantity", "mean"))
        lineitem_avg_quantity["avg_quantity_scaled"] = (
            0.2 * lineitem_avg_quantity["avg_quantity"]
        )

        lineitem_with_threshold = pd.merge(
            lineitem_part_joined, lineitem_avg_quantity, on="l_partkey"
        )

        filtered_lineitem = lineitem_with_threshold[
            lineitem_with_threshold["l_quantity"]
            < lineitem_with_threshold["avg_quantity_scaled"]
        ]

        avg_yearly = filtered_lineitem["l_extendedprice"].sum() / 7.0

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return avg_yearly
