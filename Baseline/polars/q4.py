import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_shipdate', 'l_commitdate', 'l_receiptdate'])

    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=['o_orderkey', 'o_custkey', 'o_orderdate', 'o_shippriority', 'o_orderpriority'])

    start = time.monotonic()

    var1 = 741484800.0
    var2 = 749433600.0

    q_final = (
        line_item_ds.join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .filter(pl.col("o_orderdate") >= var1)
        .filter(pl.col("o_orderdate") < var2)
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .group_by("o_orderpriority")
        .agg(pl.count().alias("order_count"))
        .sort(by="o_orderpriority")
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()