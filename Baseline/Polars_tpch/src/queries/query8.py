import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 8


def q():
    var_date_start = 788918400.0
    var_date_end = 852076800.0
    var_r_name = 3070.0
    var_s_nation = 62514.0
    var_p_type = "ECONOMY ANODIZED STEEL"

    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        nation_ds = utils.get_nation_ds()
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        supplier_ds = utils.get_supplier_ds()
        part_ds = utils.get_part_ds()
        region_ds = utils.get_region_ds()

    q_final = (
        line_item_ds.join(
            orders_ds.filter(
                (pl.col("o_orderdate") >= var_date_start)
                & (pl.col("o_orderdate") <= var_date_end)
            ),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .join(
            part_ds.filter(pl.col("p_type") == var_p_type),
            left_on="l_partkey",
            right_on="p_partkey",
        )
        .join(customer_ds, left_on="o_custkey", right_on="c_custkey")
        .join(
            nation_ds.join(
                region_ds.filter(pl.col("r_name") == var_r_name),
                left_on="n_regionkey",
                right_on="r_regionkey",
            ),
            left_on="c_nationkey",
            right_on="n_nationkey",
        )
        .join(
            supplier_ds.join(
                nation_ds, left_on="s_nationkey", right_on="n_nationkey"
            ).rename({"n_name": "n2.n_name"}),
            left_on="l_suppkey",
            right_on="s_suppkey",
        )
        .with_columns(
            [
                ((pl.col("o_orderdate") / 31536000.0 + 1970.0).round().alias("o_year")),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "volume"
                ),
                pl.col("n2.n_name").alias("nation"),
            ]
        )
        .group_by(["o_year"])
        .agg(
            [
                (
                    (pl.col("volume") * (pl.col("n2.n_name") == var_s_nation)).sum()
                    / pl.col("volume").sum()
                ).alias("mkt_share")
            ]
        )
        .sort(["o_year"])
    )

    utils.run_query(Q_NUM, q_final)
