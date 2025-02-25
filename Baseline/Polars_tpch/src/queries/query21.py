import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 21


def q():
    var_nation = 54189.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        supplier_ds = utils.get_supplier_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        nation_ds = utils.get_nation_ds()

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

    utils.run_query(Q_NUM, q_final)
