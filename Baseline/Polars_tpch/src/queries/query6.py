import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 6


def q():
    var1 = 757382400.0
    var2 = 788918400.0
    var3 = 24.0

    with TPCHTimer("Data load time for Query {Q_NUM}"):
        line_item_ds = utils.get_line_item_ds()

    q_final = (
        line_item_ds.filter(pl.col("l_shipdate") >= var1)
        .filter(pl.col("l_shipdate") < var2)
        .filter((pl.col("l_discount") >= 0.05) & (pl.col("l_discount") <= 0.07))
        .filter(pl.col("l_quantity") < var3)
        .with_columns(
            (pl.col("l_extendedprice") * pl.col("l_discount")).alias("revenue")
        )
        .select(pl.sum("revenue").alias("revenue"))
    )

    utils.run_query(Q_NUM, q_final)
