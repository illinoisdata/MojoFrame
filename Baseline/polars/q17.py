import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()

    lineitem_ds = pl.read_csv("/datadrive/tpch_large/lineitem.csv",columns=['l_partkey', 'l_quantity', 'l_extendedprice'])
    part_ds = pl.read_csv("/datadrive/tpch_large/part.csv",columns=['p_partkey', 'p_brand', 'p_container'])
    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    start = time.monotonic()

    lineitem_grouped = lineitem_ds.group_by(["l_partkey"]).agg(
        pl.col("l_quantity").mean().alias("l_quantity_avg")
    )

    q_final = (
        part_ds.filter(
            (pl.col("p_brand") == "Brand#23") & (pl.col("p_container") == "MED BOX")
        )
        .join(lineitem_ds, left_on="p_partkey", right_on="l_partkey")
        .join(lineitem_grouped, left_on="p_partkey", right_on="l_partkey")
        .filter(pl.col("l_quantity") < 0.2 * pl.col("l_quantity_avg"))
        .sum()
        .with_columns((pl.col("l_extendedprice") / 7.0).alias("avg_yearly"))
        .select(pl.col("avg_yearly"))
    )

   
    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()