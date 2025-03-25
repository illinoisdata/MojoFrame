import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"

def q():
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_orderkey', 'l_quantity'])
    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=['o_orderkey', 'o_orderdate', 'o_totalprice', 'o_custkey'])
    customer_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/customer.csv", columns=['c_custkey'])
   

    start = time.monotonic()
    
    var_quantity = 300

    final_cols = [
        "c_custkey",
        "o_orderkey",
        "o_orderdate",
        "o_totalprice",
        "sum"
    ]

    filtered_line_item_ds = (line_item_ds.group_by("l_orderkey")
        .agg(pl.sum("l_quantity").alias("sum_l_quantity"))
        .select(["l_orderkey", "sum_l_quantity"])
        .filter(pl.col("sum_l_quantity") > var_quantity)
        .with_columns(pl.col("sum_l_quantity").cast(pl.datatypes.Float64).alias("sum")))

    q_final = (
        customer_ds.join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(filtered_line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .select(final_cols)
        .sort(["o_totalprice", "o_orderdate"], descending=[False, False])
    )


    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()