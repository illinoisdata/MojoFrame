import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Load data
start_load = time.perf_counter()

df_lineitem = pd.read_csv('/datadrive/tpch_large/lineitem.csv', usecols=['l_suppkey', 'l_extendedprice', 'l_discount', 'l_shipdate'])
df_supplier = pd.read_csv('/datadrive/tpch_large/supplier.csv', usecols=['s_suppkey', 's_name'])

end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

start_time = time.time()

# Step 1: Filter `lineitem` by date range (January 1, 1996 to April 1, 1996)
df_lineitem = df_lineitem[
    (df_lineitem['l_shipdate'] >= 820454400.0) & (df_lineitem['l_shipdate'] < 828230400.0)
]

# Step 2: Calculate `total_revenue` for each `l_suppkey` in the filtered `lineitem` data
df_lineitem['total_revenue'] = df_lineitem['l_extendedprice'] * (1 - df_lineitem['l_discount'])

joined_ls_df = pd.merge(df_lineitem, df_supplier, left_on='l_suppkey', right_on='s_suppkey')

supplier_revenue = joined_ls_df.groupby(['l_suppkey', 's_name']).agg('sum').reset_index()
print("joined_ls_df after groupby: ", supplier_revenue.shape)

# Step 4: Find the maximum `total_revenue`
max_revenue = supplier_revenue['total_revenue'].max()

# Step 5: Filter for suppliers with the maximum `total_revenue`
supplier_revenue = supplier_revenue[supplier_revenue['total_revenue'] == max_revenue]

end_time = time.time()

# Display results
print(f"Execution time: {end_time - start_time} seconds")
print("Max revenue: ", max_revenue)
print(supplier_revenue)