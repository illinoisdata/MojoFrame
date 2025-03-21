import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 14


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = ["l_partkey", "l_extendedprice", "l_discount", "l_shipdate"]
        df_lineitem = df_lineitem[selected_columns]

        df_part = utils.get_part_ds()
        selected_columns = ["p_partkey", "p_type"]
        df_part = df_part[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_lineitem_filtered = df_lineitem[
            (df_lineitem["l_shipdate"] >= 809913600.0)
            & (df_lineitem["l_shipdate"] < 812505600.0)
        ]

        joined_lp_df = pd.merge(
            df_lineitem_filtered,
            df_part,
            left_on="l_partkey",
            right_on="p_partkey",
            how="inner",
        )

        joined_lp_df["total_revenue"] = joined_lp_df["l_extendedprice"] * (
            1 - joined_lp_df["l_discount"]
        )
        total_revenue_sum = joined_lp_df["total_revenue"].sum()

        promo_revenue_sum = joined_lp_df.loc[
            joined_lp_df["p_type"].str.startswith("PROMO"), "total_revenue"
        ].sum()

        promo_revenue_percentage = (promo_revenue_sum / total_revenue_sum) * 100

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return promo_revenue_percentage
