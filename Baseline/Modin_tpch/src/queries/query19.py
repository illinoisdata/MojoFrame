import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 19


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_lineitem = utils.get_line_item_ds()
        selected_columns = [
            "l_partkey",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_shipinstruct",
            "l_shipmode",
        ]
        df_lineitem = df_lineitem[selected_columns]

        df_part = utils.get_part_ds()
        selected_columns = ["p_partkey", "p_brand", "p_size", "p_container"]
        df_part = df_part[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        lineitem_filtered = df_lineitem[
            (df_lineitem["l_shipinstruct"] == "DELIVER IN PERSON")
            & (df_lineitem["l_shipmode"].isin([2.0, 4.0]))
        ]

        joined_lp_df = pd.merge(
            lineitem_filtered,
            df_part,
            left_on="l_partkey",
            right_on="p_partkey",
            how="inner",
        )

        filtered_df = joined_lp_df[
            (
                (joined_lp_df["p_brand"] == "Brand#12")
                & (
                    joined_lp_df["p_container"].isin(
                        ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
                    )
                )
                & (joined_lp_df["p_size"].between(1, 5))
                & (joined_lp_df["l_quantity"].between(1, 11))
            )
            | (
                (joined_lp_df["p_brand"] == "Brand#23")
                & (
                    joined_lp_df["p_container"].isin(
                        ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
                    )
                )
                & (joined_lp_df["p_size"].between(1, 10))
                & (joined_lp_df["l_quantity"].between(10, 20))
            )
            | (
                (joined_lp_df["p_brand"] == "Brand#34")
                & (
                    joined_lp_df["p_container"].isin(
                        ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
                    )
                )
                & (joined_lp_df["p_size"].between(1, 15))
                & (joined_lp_df["l_quantity"].between(20, 30))
            )
        ]

        disc_price = filtered_df["l_extendedprice"] * (1 - filtered_df["l_discount"])

        revenue = disc_price.sum()

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return revenue
