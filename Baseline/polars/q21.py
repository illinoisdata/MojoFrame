import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"


def q():
    supplier_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/supplier.csv", columns=['s_suppkey', 's_nationkey', 's_name'])
    line_item_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/lineitem-med.csv", columns=['l_orderkey', 'l_suppkey', 'l_commitdate', 'l_receiptdate'])
    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=['o_orderkey', 'o_orderstatus'])
    nation_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/nation.csv", columns=['n_nationkey', 'n_name'])

    start = time.monotonic()
    
    var_nation = 54189.0
    
    line_item_order_ds = line_item_ds.join(
        orders_ds.filter(pl.col("o_orderstatus") == 70.0),
        left_on="l_orderkey",
        right_on="o_orderkey",
    )

    line_item_order_faulty = line_item_order_ds.filter(
        pl.col("l_receiptdate") > pl.col("l_commitdate")
    )

    num_total_suppliers = (
        line_item_order_ds.group_by(["l_orderkey"])
        .agg(
            [
                pl.col("l_suppkey").n_unique().alias("num_total_suppliers"),
            ]
        )
        .filter(pl.col("num_total_suppliers") >= 2)
    )
    num_faulty_suppliers = (
        line_item_order_faulty.group_by(["l_orderkey"])
        .agg([pl.col("l_suppkey").n_unique().alias("num_faulty_supplier")])
        .filter(pl.col("num_faulty_supplier") == 1)
    )

    supplier_nation_ds = supplier_ds.join(
        nation_ds.filter(pl.col("n_name") == var_nation),
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    q_final = (
        line_item_order_faulty.join(
            supplier_nation_ds, left_on="l_suppkey", right_on="s_suppkey"
        )
        .group_by(["l_orderkey", "l_suppkey"])
        .agg(pl.count().alias("l_lineitem_count"))
        .join(num_total_suppliers, left_on="l_orderkey", right_on="l_orderkey")
        .join(num_faulty_suppliers, left_on="l_orderkey", right_on="l_orderkey")
        .join(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        .group_by(["s_name"])
        .agg(pl.col("l_lineitem_count").sum().alias("numwait"))
        .sort(["numwait", "s_name"], descending=[True, False])
    )


    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()