import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 13


def q():
    with TPCHTimer("Data load time for Query {Q_NUM}"):
        orders_ds = utils.get_orders_ds()
        customer_ds = utils.get_customer_ds()

    q_final = (
        customer_ds.join(
            orders_ds.filter(
                ~pl.col("o_comment").str.contains("(.*)special(.*)requests(.*)")
            ),
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )
        .group_by(["c_custkey"])
        .agg(pl.col("o_orderkey").drop_nulls().count().alias("c_count"))
        .group_by(["c_count"])
        .agg(pl.count().alias("custdist"))
        .sort(["custdist", "c_count"], descending=[True, True])
    )

    utils.run_query(Q_NUM, q_final)
