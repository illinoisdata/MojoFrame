import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 13


def filter_not_string_exists_before(comment, filter_str1, filter_str2):
    first_pos_str1 = comment.find(filter_str1)
    last_pos_str2 = comment.rfind(filter_str2)

    return not (
        first_pos_str1 != -1 and last_pos_str2 != -1 and first_pos_str1 < last_pos_str2
    )


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_orders = utils.get_orders_ds()
        selected_columns = ["o_custkey", "o_comment"]
        df_orders = df_orders[selected_columns]

        df_customer = utils.get_customer_ds()
        selected_columns = ["c_custkey"]
        df_customer = df_customer[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        filter_str1 = "special"
        filter_str2 = "requests"

        df_orders_filtered = df_orders[
            df_orders["o_comment"].apply(
                lambda comment: filter_not_string_exists_before(
                    comment, filter_str1, filter_str2
                )
            )
        ]

        joined_co_df = pd.merge(
            df_customer,
            df_orders_filtered,
            left_on="c_custkey",
            right_on="o_custkey",
            how="inner",
        )

        customer_order_counts = (
            joined_co_df.groupby("c_custkey").size().reset_index(name="c_count")
        )

        order_count_distribution = (
            customer_order_counts.groupby("c_count").size().reset_index(name="custdist")
        )

        result_df = order_count_distribution.sort_values(by=["custdist", "c_count"])

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return result_df
