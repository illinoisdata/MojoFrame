import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv",columns=['l_partkey', 'l_extendedprice', 'l_discount', 'l_shipdate'])
    part_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/part.csv", columns=['p_partkey', 'p_type'])
   

    start = time.monotonic()

    VAR1 = 809913600.0
    VAR2 = 812505600.0

    q_final = (
        line_item_ds.filter(
            (pl.col("l_shipdate") >= VAR1) & (pl.col("l_shipdate") < VAR2)
        )
        .join(part_ds, left_on="l_partkey", right_on="p_partkey")
        .with_columns(
            [
                pl.col("p_type").str.starts_with("PROMO").alias("is_promo"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "revenue"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("revenue") * pl.col("is_promo")).alias("num_revenue"),
                pl.col("revenue").alias("den_revenue"),
            ]
        )
        .sum()
        .select(
            (100.0 * pl.col("num_revenue") / pl.col("den_revenue")).alias(
                "promo_revenue"
            )
        )
    )
   

    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()