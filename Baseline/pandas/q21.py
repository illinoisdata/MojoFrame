import pandas as pd
import time
# ------------------------------------------------------------------
# 1. Read CSVs (equivalent to CSVReaderNodes)
# ------------------------------------------------------------------
lineitem = pd.read_csv(
    "/home/shengya4/data/tpch_3gb/lineitem-med.csv",
    usecols=["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"],
)
orders = pd.read_csv(
    "/home/shengya4/data/tpch_3gb/orders.csv",
    usecols=["o_orderkey", "o_orderstatus"]
)
supplier = pd.read_csv(
    "/home/shengya4/data/tpch_3gb/supplier.csv",
    usecols=["s_suppkey", "s_nationkey", "s_name"]
)
nation = pd.read_csv(
    "/home/shengya4/data/tpch_3gb/nation.csv",
    usecols=["n_nationkey", "n_name"]
)

start_time = time.time()

# ------------------------------------------------------------------
# 2. WHERE Node on orders: o_orderstatus = 'F' (orders_where_node)
# ------------------------------------------------------------------
orders_filtered = orders[orders["o_orderstatus"] == 70.0]

# ------------------------------------------------------------------
# 3. Merge JOIN lineitem + orders (lo_merge_join_node):
#    lineitem.l_orderkey == orders_filtered.o_orderkey
# ------------------------------------------------------------------
lineitem_orders = lineitem.merge(
    orders_filtered,
    left_on="l_orderkey",
    right_on="o_orderkey",
    how="inner"  # equivalent to an inner join
)

# ------------------------------------------------------------------
# 4. WHERE Node on lineitem: l_receiptdate > l_commitdate (lineitem_where_node)
#    Apply it to the merged lineitem_orders
# ------------------------------------------------------------------
late_lineitems = lineitem_orders[
    lineitem_orders["l_receiptdate"] > lineitem_orders["l_commitdate"]
]

# ------------------------------------------------------------------
# 5. orderkey_merger_node:
#    - For each order, compute:
#       * num_suppliers = distinct suppliers in lineitem_orders
#       * num_late_suppliers = distinct suppliers in late_lineitems
#    - Filter those with num_suppliers > 1 and num_late_suppliers == 1
#    (This replicates the EXISTS/NOT EXISTS logic in the SQL)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 6. WHERE Node on nation: n_name = 'SAUDI ARABIA' (nation_where_node)
# ------------------------------------------------------------------
nation_sa = nation[nation["n_name"] == 54189]

# ------------------------------------------------------------------
# 7. HASH JOIN supplier + nation_sa (sn_hash_join_node)
#    s_nationkey == n_nationkey
# ------------------------------------------------------------------
supplier_sa = supplier.merge(
    nation_sa, 
    left_on="s_nationkey", 
    right_on="n_nationkey",
    how="inner"
)

# ------------------------------------------------------------------
# 8. HASH JOIN lineitem_where_node + supplier_sa (ls_hash_join_node)
#    late_lineitems.l_suppkey == supplier_sa.s_suppkey
# ------------------------------------------------------------------
late_lineitems_sa = late_lineitems.merge(
    supplier_sa,
    left_on="l_suppkey",
    right_on="s_suppkey",
    how="inner"
)

# ------------------------------------------------------------------
# 9. supplier_merger_node:
#    Join late_lineitems_sa with valid_orders on l_orderkey
#    Then group by (s_name, l_orderkey) and count l_suppkey
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 10. GROUP BY Accumulator (groupby_node) and final SELECT (select_node)
#     - group by s_name, sum counts
#     - sort by sum(count) desc, s_name asc
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
# print(result)

end_time = time.time()

# Display the result
print(f"Execution time: {end_time - start_time} seconds")
print(result)