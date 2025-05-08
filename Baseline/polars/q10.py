import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()

    customer_ds = pl.read_csv("/datadrive/tpch_large/customer.csv", columns=["c_custkey", "c_nationkey", "c_acctbal"])
    line_item_ds = pl.read_csv("/datadrive/tpch_large/lineitem.csv", columns=["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag"])
    orders_ds = pl.read_csv("/datadrive/tpch_large/orders.csv", columns=["o_orderkey", "o_custkey", "o_orderdate"])
    nation_ds = pl.read_csv("/datadrive/tpch_large/nation.csv", columns=["n_nationkey", "n_name"])

    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    start = time.monotonic()

    var1 = 750643200.0
    var2 = 757382400.0
    var3 = 82.0

    q_final = (
        customer_ds.join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(nation_ds, left_on="c_nationkey", right_on="n_nationkey")
        .filter(pl.col("o_orderdate") >= var1)
        .filter(pl.col("o_orderdate") < var2)
        .filter(pl.col("l_returnflag") == var3)
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by(
            [
                "c_custkey",
                "c_acctbal",
                "n_name"
            ]
        )
        .agg([pl.sum("revenue")])
        .select(
            [
                "c_custkey",
                "revenue",
                "c_acctbal",
                "n_name"
            ]
        )
        .sort(by="revenue")
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()