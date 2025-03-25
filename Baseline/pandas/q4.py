import time
import pandas as pd

pd.set_option('display.max_columns', None)

# Load orders and lineitem datasets
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_custkey', 'o_orderdate', 'o_shippriority', 'o_orderpriority'])

file_path_lineitem = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path_lineitem, usecols=['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_shipdate', "l_commitdate", "l_receiptdate"])

print("Original orders shape:", df_orders.shape)
print("Original lineitem shape:", df_lineitem.shape)

# Start timing
start_time = time.time()

# Step 1: Filter orders by o_orderdate range before joining
df_orders_filtered = df_orders[
    (df_orders['o_orderdate'] >= 741484800.0) & (df_orders['o_orderdate'] < 749433600.0)
]

# Step 2: Filter lineitems where l_commitdate < l_receiptdate before joining
df_lineitem_filtered = df_lineitem[
    df_lineitem['l_commitdate'] < df_lineitem['l_receiptdate']
]

# Step 3: Perform the join after filtering both DataFrames
df_merged = pd.merge(df_orders_filtered, df_lineitem_filtered, left_on='o_orderkey', right_on='l_orderkey', how='inner')

# Step 4: Group by 'o_orderpriority' and count the occurrences
grouped_count = df_merged.groupby('o_orderpriority').size()

# End timing
end_time = time.time()

# Output the results
print("Execution time:", end_time - start_time)
print("Grouped count shape:", grouped_count.shape)
print(grouped_count)