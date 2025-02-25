import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 19


def q():
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        line_item_ds = utils.get_line_item_ds()
        part_ds = utils.get_part_ds()

    q_final = (
        line_item_ds.filter(
            (pl.col("l_shipmode").is_in(pl.lit(pl.Series([2.0, 4.0]))))
            & (pl.col("l_shipinstruct") == "DELIVER IN PERSON")
        )
        .join(part_ds, left_on="l_partkey", right_on="p_partkey")
        .filter(
            (
                (pl.col("p_brand") == "Brand#12")
                & (
                    pl.col("p_container").is_in(
                        pl.lit(pl.Series(["SM CASE", "SM BOX", "SM PACK", "SM PKG"]))
                    )
                )
                & (pl.col("l_quantity").is_between(1, 11, "none"))
                & (pl.col("p_size").is_between(1, 5, "none"))
                & (pl.col("l_shipmode").is_in(pl.lit(pl.Series([2.0, 4.0]))))
                & (pl.col("l_shipinstruct") == "DELIVER IN PERSON")
            )
            | (
                (pl.col("p_brand") == "Brand#23")
                & (
                    pl.col("p_container").is_in(
                        pl.lit(pl.Series(["MED BAG", "MED BOX", "MED PKG", "MED PACK"]))
                    )
                )
                & (pl.col("l_quantity").is_between(10, 20, "none"))
                & (pl.col("p_size").is_between(1, 10, "none"))
            )
            | (
                (pl.col("p_brand") == "Brand#34")
                & (
                    pl.col("p_container").is_in(
                        pl.lit(pl.Series(["LG CASE", "LG BOX", "LG PACK", "LG PKG"]))
                    )
                )
                & (pl.col("l_quantity").is_between(20, 30, "none"))
                & (pl.col("p_size").is_between(1, 15, "none"))
            )
        )
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .select("revenue")
        .sum()
    )

    utils.run_query(Q_NUM, q_final)
