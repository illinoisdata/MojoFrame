import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()

    region_ds = pl.read_csv("/datadrive/tpch_large/region.csv", columns=["r_regionkey", "r_name"])

    nation_ds = pl.read_csv("/datadrive/tpch_large/nation.csv", columns=["n_nationkey", "n_regionkey", "n_name"])

    customer_ds = pl.read_csv("/datadrive/tpch_large/customer.csv", columns=["c_custkey", "c_nationkey"])
    
    line_item_ds = pl.read_csv("/datadrive/tpch_large/lineitem.csv", columns=["l_orderkey", "l_extendedprice", "l_discount", "l_suppkey"])

    orders_ds = pl.read_csv("/datadrive/tpch_large/orders.csv", columns=["o_custkey", "o_orderkey", "o_orderdate"])

    supplier_ds = pl.read_csv("/datadrive/tpch_large/supplier.csv", columns=["s_suppkey", "s_nationkey"])

    end_load = time.perf_counter()

    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    start = time.monotonic()

    var1 = 43715.0
    var2 = 757382400.0
    var3 = 788918400.0

    q_final = (
        region_ds.join(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier_ds,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(pl.col("r_name") == var1)
        .filter(pl.col("o_orderdate") >= var2)
        .filter(pl.col("o_orderdate") < var3)
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg([pl.sum("revenue")])
        .sort(by="revenue", descending=True)
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()