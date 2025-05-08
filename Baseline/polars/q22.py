import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"


def q():
    start_load = time.perf_counter()

    customer_ds = pl.read_csv("/datadrive/tpch_large/customer.csv", columns=["c_custkey", "c_nationkey", "c_acctbal", "c_phone"])
    orders_ds = pl.read_csv("/datadrive/tpch_large/orders.csv", columns=["o_custkey", "o_orderkey"])

    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    start = time.monotonic()
    
    var_list = [13.0, 31.0, 23.0, 29.0, 30.0, 18.0, 17.0]

    avg_c_acctbal = (
        customer_ds.filter(
            (pl.col("c_acctbal") > 0)
            & (
                pl.col("c_phone")
                .str.slice(0, 2)
                .cast(pl.Int32)
                .is_in(pl.lit(pl.Series(var_list)))
            )
        )
        .select(pl.col("c_acctbal"))
        .mean()
    )

    q_final = (
        customer_ds.with_columns(
            (pl.col("c_phone").str.slice(0, 2).alias("cntrycode").cast(pl.Int32))
        )
        .filter(pl.col("cntrycode").is_in(pl.lit(pl.Series(var_list))))
        .filter(pl.col("c_acctbal") > pl.lit(avg_c_acctbal.get_column("c_acctbal")))
        .join(orders_ds, left_on=["c_custkey"], right_on=["o_custkey"], how="left")
        .filter(pl.col("o_orderkey").is_null())
        .group_by(["cntrycode"])
        .agg(
            [pl.count().alias("numcust"), pl.col("c_acctbal").sum().alias("totacctbal")]
        )
        .sort("cntrycode")
    )


    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()