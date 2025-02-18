import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 4


def q():
    var1 = 741484800
    var2 = 749433600

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()

    q_final = (
        line_item_ds.join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .filter(pl.col("o_orderdate") >= var1)
        .filter(pl.col("o_orderdate") < var2)
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .group_by("o_orderpriority")
        .agg(pl.count().alias("order_count"))
        .sort(by="o_orderpriority")
        .with_columns(pl.col("order_count").cast(pl.datatypes.Int64))
    )

    utils.run_query(Q_NUM, q_final)
