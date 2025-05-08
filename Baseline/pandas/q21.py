import pandas as pd
import time

start_load = time.perf_counter()

lineitem = pd.read_csv(
    "/datadrive/tpch_large/lineitem.csv",
    usecols=["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"],
)
orders = pd.read_csv(
    "/datadrive/tpch_large/orders.csv",
    usecols=["o_orderkey", "o_orderstatus"]
)
supplier = pd.read_csv(
    "/datadrive/tpch_large/supplier.csv",
    usecols=["s_suppkey", "s_nationkey", "s_name"]
)
nation = pd.read_csv(
    "/datadrive/tpch_large/nation.csv",
    usecols=["n_nationkey", "n_name"]
)

end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

start_time = time.time()

orders_filtered = orders[orders["o_orderstatus"] == 70.0]

lineitem_orders = lineitem.merge(
    orders_filtered,
    left_on="l_orderkey",
    right_on="o_orderkey",
    how="inner"  # equivalent to an inner join
)

late_lineitems = lineitem_orders[
    lineitem_orders["l_receiptdate"] > lineitem_orders["l_commitdate"]
]

grouped_lineitem = (
    lineitem_orders.groupby("l_orderkey")["l_suppkey"]
    .nunique()
    .reset_index(name="num_suppliers")
)

print(grouped_lineitem.shape)

grouped_late = (
    late_lineitems.groupby("l_orderkey")["l_suppkey"]
    .nunique()
    .reset_index(name="num_late_suppliers")
)

print(grouped_late.shape)

order_stats = grouped_lineitem.merge(grouped_late, on="l_orderkey", how="inner")

valid_orders = order_stats[
    (order_stats["num_suppliers"] > 1) &
    (order_stats["num_late_suppliers"] == 1)
][["l_orderkey"]]


nation_sa = nation[nation["n_name"] == 54189]


supplier_sa = supplier.merge(
    nation_sa, 
    left_on="s_nationkey", 
    right_on="n_nationkey",
    how="inner"
)


late_lineitems_sa = late_lineitems.merge(
    supplier_sa,
    left_on="l_suppkey",
    right_on="s_suppkey",
    how="inner"
)

valid_lineitems_sa = late_lineitems_sa.merge(
    valid_orders,
    on="l_orderkey",
    how="inner"
)

print(valid_lineitems_sa.shape)

count_per_order = (
    valid_lineitems_sa
    .groupby(["s_name", "l_orderkey"], as_index=False)["l_suppkey"]
    .count()
)

print(count_per_order.shape)


result = (
    count_per_order
    .groupby("s_name")
    .agg("sum")
    .reset_index()
    .rename(columns={"l_suppkey": "numwait"})
)

print(result.shape)
# Sort by numwait desc, s_name asc
result = result.sort_values(by=["numwait", "s_name"])

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
print(result)