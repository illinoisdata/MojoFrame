import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    nation_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/nation.csv", columns=["n_nationkey", "n_name"])

    customer_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/customer.csv", columns=["c_custkey", "c_nationkey"])

    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=["l_orderkey", "l_extendedprice", "l_discount", "l_suppkey", "l_shipdate"])

    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=["o_custkey", "o_orderkey"])

    supplier_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/supplier.csv", columns=["s_suppkey", "s_nationkey"])


    start = time.monotonic()

    n1 = nation_ds.filter(pl.col("n_name") == 38075.0)
    n2 = nation_ds.filter(pl.col("n_name") == 52342.0)

    df1 = (
        customer_ds.join(n1, left_on="c_nationkey", right_on="n_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .rename({"n_name": "cust_nation"})
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        .join(n2, left_on="s_nationkey", right_on="n_nationkey")
        .rename({"n_name": "supp_nation"})
    )

    df2 = (
        customer_ds.join(n2, left_on="c_nationkey", right_on="n_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .rename({"n_name": "cust_nation"})
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        .join(n1, left_on="s_nationkey", right_on="n_nationkey")
        .rename({"n_name": "supp_nation"})
    )

    q_final = (
        pl.concat([df1, df2])
        .filter(pl.col("l_shipdate") >= 788918400.0)
        .filter(pl.col("l_shipdate") <= 852076800.0)
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("volume")
        )
        .with_columns((pl.col("l_shipdate") / 31536000.0 + 1970.0).round().alias("l_year"))
        .group_by(["supp_nation", "cust_nation", "l_year"])
        .agg([pl.sum("volume").alias("revenue")])
        .sort(by=["supp_nation", "cust_nation", "l_year"])
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()