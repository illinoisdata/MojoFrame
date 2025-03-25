import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"


def q():
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_extendedprice', 'l_discount', 'l_quantity', 'l_partkey', 'l_shipmode', 'l_shipinstruct'])
    part_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/part.csv", columns=['p_brand', 'p_container', 'p_partkey', 'p_size'])

    start = time.monotonic()
    
    q_final = (
        line_item_ds.filter(
            (pl.col("l_shipmode").is_in(pl.lit(pl.Series([2.0, 4.0]))))
            & (pl.col("l_shipinstruct") == "DELIVER IN PERSON")
        )
        .join(part_ds, left_on="l_partkey", right_on="p_partkey")
        .filter(
            (
                (pl.col("p_brand") == "Brand#12")
                & (
                    pl.col("p_container").is_in(
                        pl.lit(pl.Series(["SM CASE", "SM BOX", "SM PACK", "SM PKG"]))
                    )
                )
                & (pl.col("l_quantity").is_between(1, 11, "both"))
                & (pl.col("p_size").is_between(1, 5, "both"))
            )
            | (
                (pl.col("p_brand") == "Brand#23")
                & (
                    pl.col("p_container").is_in(
                        pl.lit(pl.Series(["MED BAG", "MED BOX", "MED PKG", "MED PACK"]))
                    )
                )
                & (pl.col("l_quantity").is_between(10, 20, "both"))
                & (pl.col("p_size").is_between(1, 10, "both"))
            )
            | (
                (pl.col("p_brand") == "Brand#34")
                & (
                    pl.col("p_container").is_in(
                        pl.lit(pl.Series(["LG CASE", "LG BOX", "LG PACK", "LG PKG"]))
                    )
                )
                & (pl.col("l_quantity").is_between(20, 30, "both"))
                & (pl.col("p_size").is_between(1, 15, "both"))
            )
        )
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .select("revenue")
        .sum()
    )

    
    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()