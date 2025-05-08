import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()

    line_item_ds = pl.read_csv(
        "/datadrive/tpch_large/lineitem.csv",
        columns=[
            "l_orderkey",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipmode",
        ]
    )

    orders_ds = pl.read_csv(
        "/datadrive/tpch_large/orders.csv",
        columns=["o_orderkey", "o_orderpriority"]
    )
    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    start = time.monotonic()
    
    var_ship_mode1 = 5.0
    var_ship_mode2 = 6.0
    var_date = 757382400.0
    var_date_interval_1yr = 788918400.0


    q_final = (
        orders_ds.join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .filter(
            (pl.col("l_shipmode") == var_ship_mode1)
            | (pl.col("l_shipmode") == var_ship_mode2)
        )
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .filter(pl.col("l_shipdate") < pl.col("l_commitdate"))
        .filter(pl.col("l_receiptdate") >= var_date)
        .filter(pl.col("l_receiptdate") < var_date_interval_1yr)
        .group_by(["l_shipmode"])
        .agg(
            [
                (
                    (pl.col("o_orderpriority") == 1.0)
                    | (pl.col("o_orderpriority") == 2.0)
                )
                .sum()
                .alias("high_line_count"),
                (
                    (pl.col("o_orderpriority") != 1.0)
                    & (pl.col("o_orderpriority") != 2.0)
                )
                .sum()
                .alias("low_line_count"),
            ]
        )
        .sort(by="l_shipmode")
    )


    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()