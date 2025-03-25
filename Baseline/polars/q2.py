import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"


def q():
    part_supp_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/partsupp.csv", columns=['ps_partkey', 'ps_suppkey', 'ps_supplycost'])

    part_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/part.csv", columns=['p_partkey', 'p_size', 'p_type'])

    supplier_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/supplier.csv", columns=['s_suppkey', 's_nationkey', 's_acctbal', 's_name'])
    
    nation_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/nation.csv", columns=['n_nationkey', 'n_regionkey', 'n_name'])

    region_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/region.csv", columns=['r_regionkey', 'r_name'])

    start = time.monotonic()
    
    var1 = 15.0
    var2 = "BRASS"
    var3 = 35796.0

    final_cols = [
        "s_acctbal",
        "s_name",
        "n_name",
        "p_partkey"
    ]

    result_q1 = (
        part_ds.join(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .join(region_ds, left_on="n_regionkey", right_on="r_regionkey")
        .filter(pl.col("p_size") == var1)
        .filter(pl.col("p_type").str.ends_with(var2))
        .filter(pl.col("r_name") == var3)
    )

    result2 = result_q1.group_by("p_partkey").agg(
        pl.min("ps_supplycost").alias("ps_supplycost_min")
    )

    q_final = (
        result_q1.join(
            result2,
            left_on=["p_partkey", "ps_supplycost"],
            right_on=["p_partkey", "ps_supplycost_min"],
        )
        .select(final_cols)
        .sort(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .with_columns(pl.col(pl.datatypes.Utf8).str.strip_chars().name.keep())
    )

    end = time.monotonic()

    print(f"Elapsed Time (Monotonic): {end - start} seconds")

    print(q_final)


if __name__ == "__main__":
    q()