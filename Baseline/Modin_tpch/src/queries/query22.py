import tracemalloc

import modin.pandas as pd

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 22


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        df_customer = utils.get_customer_ds()
        selected_columns = ["c_custkey", "c_nationkey", "c_acctbal", "c_phone"]
        df_customer = df_customer[selected_columns]

        df_orders = utils.get_orders_ds()
        selected_columns = ["o_custkey", "o_orderkey"]
        df_orders = df_orders[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        var_list = [13.0, 31.0, 23.0, 29.0, 30.0, 18.0, 17.0]

        avg_c_acctbal = df_customer[
            (df_customer["c_acctbal"] > 0)
            & (df_customer["c_phone"].str[:2].astype(float).isin(var_list))
        ]["c_acctbal"].mean()

        df_customer["cntrycode"] = df_customer["c_phone"].str[:2].astype(float)

        filtered_df = df_customer[
            (df_customer["cntrycode"].isin(var_list))
            & (df_customer["c_acctbal"] > avg_c_acctbal)
        ]

        joined_df = pd.merge(
            filtered_df,
            df_orders,
            left_on="c_custkey",  # Customer key in df_customer
            right_on="o_custkey",  # Customer key in df_orders
            how="left",
        )

        no_orders_df = joined_df[joined_df["o_orderkey"].isna()]

        q_final = (
            no_orders_df.groupby("cntrycode", as_index=False)
            .agg(numcust=("c_custkey", "count"), totacctbal=("c_acctbal", "sum"))
            .sort_values("cntrycode")
        )

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return q_final
