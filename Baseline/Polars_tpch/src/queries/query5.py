import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 5


def q():
    var1 = 43715.0
    var2 = 757382400.0
    var3 = 788918400.0

    with TPCHTimer("Data load time for Query {Q_NUM}"):
        region_ds = utils.get_region_ds()
        nation_ds = utils.get_nation_ds()
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        supplier_ds = utils.get_supplier_ds()

    q_final = (
        region_ds.join(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier_ds,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(pl.col("r_name") == var1)
        .filter(pl.col("o_orderdate") >= var2)
        .filter(pl.col("o_orderdate") < var3)
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg([pl.sum("revenue")])
        .sort(by="revenue", descending=True)
    )

    utils.run_query(Q_NUM, q_final)
