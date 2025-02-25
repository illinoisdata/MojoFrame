import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 10


def q():
    var1 = 750643200.0
    var2 = 757382400.0
    var3 = 82.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        nation_ds = utils.get_nation_ds()

    q_final = (
        customer_ds.join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(nation_ds, left_on="c_nationkey", right_on="n_nationkey")
        .filter(pl.col("o_orderdate") >= var1)
        .filter(pl.col("o_orderdate") < var2)
        .filter(pl.col("l_returnflag") == var3)
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by(
            [
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            ]
        )
        .agg([pl.sum("revenue")])
        .select(
            [
                "c_custkey",
                "c_name",
                "revenue",
                "c_acctbal",
                "n_name",
                "c_address",
                "c_phone",
                "c_comment",
            ]
        )
        .sort(by="revenue", descending=True)
        .with_columns(pl.col(pl.datatypes.Utf8).str.strip_chars().name.keep())
    )

    utils.run_query(Q_NUM, q_final)
