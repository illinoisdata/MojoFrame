
import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    q = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=["l_quantity", "l_extendedprice", "l_discount", "l_returnflag", "l_shipdate", "l_linestatus", "l_tax"])
    # q = utils.get_line_item_ds()
    #q = q.select(["l_quantity", "l_extendedprice", "l_discount", "l_returnflag", "l_shipdate", "l_linestatus", "l_tax"])
    start = time.monotonic()

    VAR1 = 904608000.0

    q_final = (
        q.filter(pl.col("l_shipdate") <= VAR1)
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                pl.sum("l_quantity").alias("sum_qty"),
                pl.sum("l_extendedprice").alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    pl.col("l_extendedprice")
                    * (1.0 - pl.col("l_discount"))
                    * (1.0 + pl.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                pl.mean("l_quantity").alias("avg_qty"),
                pl.mean("l_extendedprice").alias("avg_price"),
                pl.mean("l_discount").alias("avg_disc"),
                pl.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )

    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()