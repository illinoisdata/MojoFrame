import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 1


def q():
    VAR1 = 904608000.0

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        q = utils.get_line_item_ds()
    q_final = (
        q.filter(pl.col("l_shipdate") <= VAR1)
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                pl.sum("l_quantity").alias("sum_qty"),
                pl.sum("l_extendedprice").alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    pl.col("l_extendedprice")
                    * (1.0 - pl.col("l_discount"))
                    * (1.0 + pl.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                pl.mean("l_quantity").alias("avg_qty"),
                pl.mean("l_extendedprice").alias("avg_price"),
                pl.mean("l_discount").alias("avg_disc"),
                pl.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )

    utils.run_query(Q_NUM, q_final)
