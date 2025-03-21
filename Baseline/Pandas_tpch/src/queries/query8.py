import tracemalloc

import pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 8


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_nation1 = utils.get_nation_ds()
        selected_columns_nation = ["n_nationkey", "n_name", "n_regionkey"]
        df_nation1 = df_nation1[selected_columns_nation]
        df_nation1.rename(
            columns={
                "n_nationkey": "n_nationkey1",
                "n_name": "n_name1",
                "n_regionkey": "n_regionkey1",
            },
            inplace=True,
        )

        df_nation2 = utils.get_nation_ds()
        selected_columns_nation = ["n_nationkey", "n_name", "n_regionkey"]
        df_nation2 = df_nation2[selected_columns_nation]
        df_nation2.rename(
            columns={
                "n_nationkey": "n_nationkey2",
                "n_name": "n_name2",
                "n_regionkey": "n_regionkey2",
            },
            inplace=True,
        )

        df_customer = utils.get_customer_ds()
        selected_columns_customer = ["c_custkey", "c_nationkey"]
        df_customer = df_customer[selected_columns_customer]

        df_lineitem = utils.get_line_item_ds()
        selected_columns_lineitem = [
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_suppkey",
            "l_partkey",
        ]
        df_lineitem = df_lineitem[selected_columns_lineitem]
        df_lineitem["volume"] = df_lineitem["l_extendedprice"] * (
            1 - df_lineitem["l_discount"]
        )

        df_orders = utils.get_orders_ds()
        selected_columns_orders = ["o_custkey", "o_orderkey", "o_orderdate"]
        df_orders = df_orders[selected_columns_orders]

        df_supplier = utils.get_supplier_ds()
        selected_columns_supplier = ["s_suppkey", "s_nationkey"]
        df_supplier = df_supplier[selected_columns_supplier]

        df_part = utils.get_part_ds()
        selected_columns_part = ["p_partkey", "p_type"]
        df_part = df_part[selected_columns_part]

        df_region = utils.get_region_ds()
        selected_columns = ["r_regionkey", "r_name"]
        df_region = df_region[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        df_part_filtered = df_part[df_part["p_type"] == "ECONOMY ANODIZED STEEL"]
        df_region_filtered = df_region[df_region["r_name"] == 3070.0]
        df_orders_filtered = df_orders[
            (df_orders["o_orderdate"] >= 788918400.0)
            & (df_orders["o_orderdate"] <= 852076800.0)
        ]
        df_joined_pl = pd.merge(
            df_part_filtered,
            df_lineitem,
            left_on="p_partkey",
            right_on="l_partkey",
            how="inner",
        )
        df_joined_s = pd.merge(
            df_supplier,
            df_joined_pl,
            left_on="s_suppkey",
            right_on="l_suppkey",
            how="inner",
        )
        df_joined_o = pd.merge(
            df_joined_s,
            df_orders_filtered,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",
        )
        df_joined_c = pd.merge(
            df_joined_o,
            df_customer,
            left_on="o_custkey",
            right_on="c_custkey",
            how="inner",
        )
        df_joined_n1 = pd.merge(
            df_joined_c,
            df_nation1,
            left_on="c_nationkey",
            right_on="n_nationkey1",
            how="inner",
        )
        df_joined_n1_r = pd.merge(
            df_joined_n1,
            df_region_filtered,
            left_on="n_regionkey1",
            right_on="r_regionkey",
            how="inner",
        )
        all_nations = pd.merge(
            df_joined_n1_r,
            df_nation2,
            left_on="s_nationkey",
            right_on="n_nationkey2",
            how="inner",
        )
        all_nations["o_orderdate"] = (
            1970.0 + (all_nations["o_orderdate"] / 31536000.0)
        ).round()

        result = (
            all_nations.groupby("o_orderdate")
            .apply(
                lambda x: pd.Series(
                    {
                        "mkt_share": x.loc[x["n_name2"] == 62514.0, "volume"].sum()
                        / x["volume"].sum()
                    }
                )
            )
            .reset_index()
        )

        result = result.sort_values(by="o_orderdate")

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return q_final
