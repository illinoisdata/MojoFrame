import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()
    partsupp_ds = pl.read_csv("/datadrive/tpch_large/partsupp.csv", columns=['ps_partkey', 'ps_suppkey', 'ps_supplycost', 'ps_availqty'])
    supplier_ds = pl.read_csv("/datadrive/tpch_large/supplier.csv", columns=['s_suppkey', 's_nationkey']) 
    nation_ds = pl.read_csv("/datadrive/tpch_large/nation.csv", columns=['n_nationkey', 'n_name'])
    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    start = time.monotonic()

    var_n_name = 52342.0

    ps_supplycost_agg = (
        partsupp_ds.join(
            supplier_ds.join(
                nation_ds.filter(pl.col("n_name") == var_n_name),
                left_on="s_nationkey",
                right_on="n_nationkey",
            ),
            left_on="ps_suppkey",
            right_on="s_suppkey",
        )
        .with_columns(
            (pl.col("ps_supplycost") * pl.col("ps_availqty") * 0.00001).alias(
                "value_limit"
            )
        )
        .select(["value_limit"])
        .sum()
    )

    q_final = (
        partsupp_ds.join(
            supplier_ds.join(
                nation_ds.filter(pl.col("n_name") == var_n_name),
                left_on="s_nationkey",
                right_on="n_nationkey",
            ),
            left_on="ps_suppkey",
            right_on="s_suppkey",
        )
        .group_by(["ps_partkey"])
        .agg([(pl.col("ps_supplycost") * pl.col("ps_availqty")).sum().alias("value")])
        .filter(pl.col("value") > pl.lit(ps_supplycost_agg.get_column("value_limit")))
        .sort(["value"])
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()