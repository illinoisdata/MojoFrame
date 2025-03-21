import tracemalloc

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 21


def q(INCLUDE_RAM: bool, RAM_USAGE: dict[str, float]):
    with TPCHTimer(f"Data load time for Query {Q_NUM}"):
        supplier = utils.get_supplier_ds()
        selected_columns = ["s_suppkey", "s_nationkey", "s_name"]
        supplier = supplier[selected_columns]

        lineitem = utils.get_line_item_ds()
        selected_columns = ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
        lineitem = lineitem[selected_columns]

        orders = utils.get_orders_ds()
        selected_columns = ["o_orderkey", "o_orderstatus"]
        orders = orders[selected_columns]

        nation = utils.get_nation_ds()
        selected_columns = ["n_nationkey", "n_name"]
        nation = nation[selected_columns]

    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {Q_NUM} execution", logging=False):
        orders_filtered = orders[orders["o_orderstatus"] == 70.0]

        lineitem_orders = lineitem.merge(
            orders_filtered,
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="inner",  # equivalent to an inner join
        )

        late_lineitems = lineitem_orders[
            lineitem_orders["l_receiptdate"] > lineitem_orders["l_commitdate"]
        ]

        grouped_lineitem = (
            lineitem_orders.groupby("l_orderkey")["l_suppkey"]
            .nunique()
            .reset_index(name="num_suppliers")
        )

        grouped_late = (
            late_lineitems.groupby("l_orderkey")["l_suppkey"]
            .nunique()
            .reset_index(name="num_late_suppliers")
        )

        order_stats = grouped_lineitem.merge(grouped_late, on="l_orderkey", how="inner")

        valid_orders = order_stats[
            (order_stats["num_suppliers"] > 1)
            & (order_stats["num_late_suppliers"] == 1)
        ][["l_orderkey"]]

        nation_sa = nation[nation["n_name"] == 54189]

        supplier_sa = supplier.merge(
            nation_sa, left_on="s_nationkey", right_on="n_nationkey", how="inner"
        )

        late_lineitems_sa = late_lineitems.merge(
            supplier_sa, left_on="l_suppkey", right_on="s_suppkey", how="inner"
        )

        valid_lineitems_sa = late_lineitems_sa.merge(
            valid_orders, on="l_orderkey", how="inner"
        )

        count_per_order = valid_lineitems_sa.groupby(
            ["s_name", "l_orderkey"], as_index=False
        )["l_suppkey"].count()

        result = (
            count_per_order.groupby("s_name")
            .agg("sum")
            .reset_index()
            .rename(columns={"l_suppkey": "numwait"})
        )

        result = result.sort_values(by=["numwait", "s_name"])

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        if f"Query {Q_NUM} peak RAM" in RAM_USAGE:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] += peak
        else:
            RAM_USAGE[f"Query {Q_NUM} peak RAM"] = peak

    return result
