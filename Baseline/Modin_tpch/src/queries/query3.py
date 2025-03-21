import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 3


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        line_item_ds["revenue"] = line_item_ds["l_extendedprice"] * (
            1 - line_item_ds["l_discount"]
        )
        orders_ds = utils.get_orders_ds()

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_customer_filtered = customer_ds[customer_ds["c_mktsegment"] == 1.0]
        df_orders_filtered = orders_ds[orders_ds["o_orderdate"] < 794880000.0]
        df_lineitem_filtered = line_item_ds[line_item_ds["l_shipdate"] > 794880000.0]

        df_merged = pd.merge(
            df_customer_filtered,
            df_orders_filtered,
            left_on="c_custkey",
            right_on="o_custkey",
            how="inner",
        )

        df_final = pd.merge(
            df_lineitem_filtered,
            df_merged,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )

        agg_funcs = {
            col: "sum"
            for col in df_final.columns
            if col not in ["l_orderkey", "o_orderdate", "o_shippriority"]
        }
        result = (
            df_final.groupby(["l_orderkey", "o_orderdate", "o_shippriority"])
            .agg(agg_funcs)
            .reset_index()
            .sort_values(by=["revenue", "o_orderdate"])
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
