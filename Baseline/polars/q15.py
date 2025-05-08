import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()

    supplier_ds = pl.read_csv("/datadrive/tpch_large/supplier.csv",columns=['s_suppkey', 's_name'])
    line_item_ds = pl.read_csv("/datadrive/tpch_large/lineitem.csv",columns=['l_suppkey', 'l_extendedprice', 'l_discount', 'l_shipdate'])

    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    start = time.monotonic()

    var_date = 820454400.0
    var_date_interval_3mon = 828230400.0

    revenue0 = (
        line_item_ds.filter(
            (pl.col("l_shipdate") >= var_date)
            & (pl.col("l_shipdate") < var_date_interval_3mon)
        )  
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("l_suppkey")
        .agg(pl.sum("revenue").alias("total_revenue"))
        .rename({"l_suppkey": "supplier_no"})
    )

    max_total_revenue = revenue0.max().get_column("total_revenue")

    q_final = (
        supplier_ds.join(
            revenue0.filter(pl.col("total_revenue") == max_total_revenue),
            left_on="s_suppkey",
            right_on="supplier_no",
        )
        .sort("s_suppkey")
    )
   

    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()