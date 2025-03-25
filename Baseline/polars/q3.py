import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    pl.enable_string_cache()

    customer_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/customer.csv", columns=['c_custkey', 'c_mktsegment'])

    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_returnflag', 'l_shipdate'])

    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=['o_custkey', 'o_orderkey', 'o_orderdate', 'o_shippriority'])
    

    customer_ds = customer_ds.with_columns(
        pl.col("c_custkey").cast(pl.Int64).cast(pl.Utf8).cast(pl.Categorical).alias("c_custkey_cat")
    )
    orders_ds = orders_ds.with_columns([
        pl.col("o_custkey").cast(pl.Int64).cast(pl.Utf8).cast(pl.Categorical).alias("o_custkey_cat"),
        pl.col("o_orderkey").cast(pl.Int64).cast(pl.Utf8).cast(pl.Categorical).alias("o_orderkey_cat")
    ])
    line_item_ds = line_item_ds.with_columns(
        pl.col("l_orderkey").cast(pl.Int64).cast(pl.Utf8).cast(pl.Categorical).alias("l_orderkey_cat")
    )


    start = time.monotonic()

    var1 = var2 = 794880000.0
    var3 = 1.0

    q_final = (
        customer_ds.filter(pl.col("c_mktsegment") == var3)
        .join(orders_ds, left_on="c_custkey_cat", right_on="o_custkey_cat")
        .join(line_item_ds, left_on="o_orderkey_cat", right_on="l_orderkey_cat")
        .filter(pl.col("o_orderdate") < var2)
        .filter(pl.col("l_shipdate") > var1)
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
        .agg([pl.sum("revenue")])
        .select(
            [
                pl.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            ]
        )
        .sort(by=["revenue", "o_orderdate"], descending=[True, False])
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()