import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "4"

def q():
    start_load = time.perf_counter()
    nation_ds = pl.read_csv("/datadrive/tpch_large/nation.csv")
    nation_ds = nation_ds.select(['n_nationkey', 'n_name'])

    line_item_ds = pl.read_csv("/datadrive/tpch_large/lineitem.csv")
    line_item_ds = line_item_ds.select(['l_orderkey', 'l_extendedprice', 'l_discount', 'l_quantity', 'l_suppkey', 'l_partkey'])

    orders_ds = pl.read_csv("/datadrive/tpch_large/orders.csv")
    orders_ds = orders_ds.select(['o_orderkey', 'o_orderdate'])

    supplier_ds = pl.read_csv("/datadrive/tpch_large/supplier.csv")
    supplier_ds = supplier_ds.select(['s_suppkey', 's_nationkey'])

    part_ds = pl.read_csv("/datadrive/tpch_large/part.csv")
    part_ds = part_ds.select(['p_partkey', 'p_name'])

    part_supp_ds = pl.read_csv("/datadrive/tpch_large/partsupp.csv")
    part_supp_ds = part_supp_ds.select(['ps_partkey', 'ps_suppkey', 'ps_supplycost'])

    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")

    start = time.monotonic()

    var_color = 'green' #LIKE GREEN

    final_cols = [
        "nation",
        "o_year",
        "sum_profit",
    ]

    q_final = (
        line_item_ds.join(part_supp_ds, left_on=["l_suppkey", "l_partkey"], right_on=["ps_suppkey", "ps_partkey"])
        .join(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .join(part_ds, left_on="l_partkey", right_on="p_partkey")
        .join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .filter(pl.col("p_name").str.contains(var_color))
        .rename({"n_name": "nation"})
        .with_columns((pl.col("o_orderdate") / 31536000.0 + 1970.0).round().alias("o_year")) #extract year from orderdate
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")) - pl.col("ps_supplycost") * pl.col("l_quantity")).alias("amount")
            )
        .group_by(["nation", "o_year"])
        .agg([pl.sum("amount").alias("sum_profit")])
        .select(final_cols)
        .sort(by=["nation", "o_year"], descending=[False, False])
    )


    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()