import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Step 1: Load only necessary columns from the CSV files
start_load = time.perf_counter()
df_customer = pd.read_csv('/datadrive/tpch_large/customer.csv', usecols=['c_custkey'])
df_orders = pd.read_csv('/datadrive/tpch_large/orders.csv', usecols=['o_orderkey', 'o_custkey', 'o_orderdate', 'o_totalprice'])
df_lineitem = pd.read_csv('/datadrive/tpch_large/lineitem.csv', usecols=['l_orderkey', 'l_quantity'])
end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

start_time = time.time()

# Step 2: Join `lineitem` and `orders` on `l_orderkey = o_orderkey`
joined_lo_df = pd.merge(
    df_lineitem, 
    df_orders, 
    left_on='l_orderkey', 
    right_on='o_orderkey'
)

# Step 3: Group `lineitem` by `l_orderkey` and calculate `SUM(l_quantity)`
lineitem_grouped = df_lineitem.groupby('l_orderkey', as_index=False).agg(l_quantity_sum=('l_quantity', 'sum'))

# Step 4: Filter `lineitem_grouped` where `SUM(l_quantity) > 300`
lineitem_filtered = lineitem_grouped[lineitem_grouped['l_quantity_sum'] > 300]

# Step 5: Join `lineitem_filtered` with `joined_lo_df` on `l_orderkey`
filtered_lo_df = pd.merge(
    lineitem_filtered, 
    joined_lo_df, 
    on='l_orderkey'
)

# Step 6: Join `filtered_lo_df` with `customer` on `o_custkey = c_custkey`
joined_oc_df = pd.merge(
    filtered_lo_df, 
    df_customer, 
    left_on='o_custkey', 
    right_on='c_custkey'
)

# Step 7: Group by required keys and calculate `SUM(l_quantity)`
grouped = joined_oc_df.groupby(
    ['c_custkey', 'o_orderkey', 'o_orderdate', 'o_totalprice'],
    as_index=False
).agg(quantity_sum_grouped=('l_quantity', 'sum'))

# Step 8: Sort by `o_totalprice` (descending) and `o_orderdate` (ascending)
result = grouped.sort_values(by=['o_totalprice', 'o_orderdate'], ascending=[False, True])

end_time = time.time()

# Display the result
print(f"Execution time: {end_time - start_time} seconds")
print(result)