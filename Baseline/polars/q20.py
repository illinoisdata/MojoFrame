import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"


def q():
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_partkey', 'l_suppkey', 'l_shipdate', 'l_quantity'])
    part_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/part.csv", columns=['p_partkey', 'p_name'])
    part_supp_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/partsupp.csv", columns=['ps_suppkey', 'ps_partkey', 'ps_availqty'])
    supp_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/supplier.csv", columns=['s_suppkey', 's_nationkey', 's_name'])
    nation_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/nation.csv", columns=['n_nationkey', 'n_name'])

    start = time.monotonic()
    
    VAR1 = 757382400.0
    VAR2 = 788918400.0

    filtered_part = part_ds.filter(
        pl.col("p_name").str.starts_with("forest")
    )

    joined_psp = part_supp_ds.join(
        filtered_part,
        left_on="ps_partkey",
        right_on="p_partkey",
        how="inner"
    )

    filtered_lineitem = line_item_ds.filter(
        (pl.col("l_shipdate") >= 757382400.0) &
        (pl.col("l_shipdate") < 788918400.0)
    )

    lineitem_agg = (
        filtered_lineitem
        .group_by(["l_partkey", "l_suppkey"])
        .agg(pl.col("l_quantity").sum().alias("l_quantity_sum"))
    )

    joined_lineitem_psp = joined_psp.join(
        lineitem_agg,
        left_on=["ps_partkey", "ps_suppkey"],
        right_on=["l_partkey", "l_suppkey"],
        how="inner"
    ).filter(
        # Filter where ps_availqty > 0.5 * l_quantity_sum
        pl.col("ps_availqty") > 0.5 * pl.col("l_quantity_sum")
    )

    joined_supplier = joined_lineitem_psp.join(
        supp_ds,
        left_on="ps_suppkey",
        right_on="s_suppkey",
        how="inner"
    )

    joined_nation = joined_supplier.join(
        nation_ds,
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="inner"
    ).filter(
        pl.col("n_name") == 35480.0
    )

    q_final = joined_nation.sort("s_name")


    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()