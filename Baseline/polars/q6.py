import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_quantity', 'l_extendedprice', 'l_discount', 'l_shipdate'])

    start = time.monotonic()

    var1 = 757382400.0
    var2 = 788918400.0
    var3 = 24.0

    q_final = (
        line_item_ds.filter(pl.col("l_shipdate") >= var1)
        .filter(pl.col("l_shipdate") < var2)
        .filter((pl.col("l_discount") >= 0.05) & (pl.col("l_discount") <= 0.07))
        .filter(pl.col("l_quantity") < var3)
        .with_columns(
            (pl.col("l_extendedprice") * pl.col("l_discount")).alias("revenue")
        )
        .select(pl.sum("revenue").alias("revenue"))
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()