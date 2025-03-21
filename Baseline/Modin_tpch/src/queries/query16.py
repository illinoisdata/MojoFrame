import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 16


def filter_not_string_exists_before(comment, filter_str1, filter_str2):
    first_pos_str1 = comment.find(filter_str1)
    last_pos_str2 = comment.rfind(filter_str2)

    return not (
        first_pos_str1 != -1 and last_pos_str2 != -1 and first_pos_str1 < last_pos_str2
    )


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_part = utils.get_part_ds()
        selected_columns = ["p_partkey", "p_brand", "p_type", "p_size"]
        df_part = df_part[selected_columns]

        df_partsupp = utils.get_part_supp_ds()
        selected_columns = ["ps_partkey", "ps_suppkey"]
        df_partsupp = df_partsupp[selected_columns]

        df_supplier = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_comment"]
        df_supplier = df_supplier[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_part_filtered = df_part[
            (df_part["p_brand"] != "Brand#45")
            & (~df_part["p_type"].str.startswith("MEDIUM POLISHED"))
            & (df_part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9]))
        ]

        df_supplier_filtered = df_supplier[
            df_supplier["s_comment"].apply(
                lambda comment: filter_not_string_exists_before(
                    comment, "Customer", "Complaints"
                )
            )
        ]

        joined_pss_df = pd.merge(
            df_partsupp,
            df_supplier_filtered,
            left_on="ps_suppkey",
            right_on="s_suppkey",
            how="inner",
        )

        joined_psp_df = pd.merge(
            joined_pss_df,
            df_part_filtered,
            left_on="ps_partkey",
            right_on="p_partkey",
            how="inner",
        )

        result = (
            joined_psp_df.groupby("p_size")
            .agg(supplier_count=("ps_suppkey", "nunique"))
            .reset_index()
        )

        result = result.sort_values(
            by=["supplier_count", "p_size"], ascending=[False, True]
        ).reset_index(drop=True)

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return result
