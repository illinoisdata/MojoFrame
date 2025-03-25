import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    part_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/part.csv", columns=["p_partkey", "p_type"])

    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_orderkey', 'l_extendedprice', 'l_discount', 'l_suppkey', 'l_partkey'])

    supplier_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/supplier.csv", columns=["s_suppkey", "s_nationkey"])

    nation_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/nation.csv", columns=["n_nationkey", "n_name", "n_regionkey"])

    customer_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/customer.csv", columns=["c_custkey", "c_nationkey"])

    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=["o_orderkey", "o_custkey", "o_orderdate"])

    region_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/region.csv", columns=["r_regionkey", "r_name"])


    start = time.monotonic()
    
    var_date_start = 788918400.0
    var_date_end = 852076800.0
    var_r_name = 3070.0
    var_s_nation = 62514.0
    var_p_type = "ECONOMY ANODIZED STEEL"
    

    q_final = (
        line_item_ds.join(
            orders_ds.filter(
                (pl.col("o_orderdate") >= var_date_start)
                & (pl.col("o_orderdate") <= var_date_end)
            ),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .join(
            part_ds.filter(pl.col("p_type") == var_p_type),
            left_on="l_partkey",
            right_on="p_partkey",
        )
        .join(customer_ds, left_on="o_custkey", right_on="c_custkey")
        .join(
            nation_ds.join(
                region_ds.filter(pl.col("r_name") == var_r_name),
                left_on="n_regionkey",
                right_on="r_regionkey",
            ),
            left_on="c_nationkey",
            right_on="n_nationkey",
        )
        .join(
            supplier_ds.join(
                nation_ds, left_on="s_nationkey", right_on="n_nationkey"
            ).rename({"n_name": "n2.n_name"}),
            left_on="l_suppkey",
            right_on="s_suppkey",
        )
        .with_columns(
            [
                ((pl.col("o_orderdate") / 31536000.0 + 1970.0).round().alias("o_year")),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "volume"
                ),
                pl.col("n2.n_name").alias("nation"),
            ]
        )
        .group_by(["o_year"])
        .agg(
            [
                (
                    (pl.col("volume") * (pl.col("n2.n_name") == var_s_nation)).sum()
                    / pl.col("volume").sum()
                ).alias("mkt_share")
            ]
        )
        .sort(["o_year"])
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()