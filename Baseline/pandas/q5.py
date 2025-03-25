import time
import pandas as pd

pd.set_option('display.max_columns', None)

# Load your datasets as before, keeping the necessary columns
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_custkey', 'o_orderdate'])

file_path_lineitem = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path_lineitem, usecols=['l_orderkey', 'l_extendedprice', 'l_discount', 'l_suppkey'])

# Calculate revenue early, as in your original code
df_lineitem['revenue'] = df_lineitem['l_extendedprice'] * (1 - df_lineitem['l_discount'])

file_path_cust = '/home/shengya4/data/tpch_3gb/customer.csv'
df_cust = pd.read_csv(file_path_cust, usecols=['c_custkey', 'c_nationkey'])

file_path_supp = '/home/shengya4/data/tpch_3gb/supplier.csv'
df_supp = pd.read_csv(file_path_supp, usecols=['s_suppkey', 's_nationkey'])

file_path_nation = '/home/shengya4/data/tpch_3gb/nation.csv'
df_nation = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_regionkey', 'n_name'])

file_path_region = '/home/shengya4/data/tpch_3gb/region.csv'
df_region = pd.read_csv(file_path_region, usecols=['r_regionkey', 'r_name'])

# Start filtering before the joins
start_time = time.time()

# Filter region for 'ASIA' (float representation 43715.0)
df_region_filtered = df_region[df_region['r_name'] == 43715.0]

# Filter orders between 1994-01-01 and 1995-01-01 using float Unix time
df_orders_filtered = df_orders[
    (df_orders['o_orderdate'] >= 757382400.0) & (df_orders['o_orderdate'] < 788918400.0)
]

# Perform the joins with filtered DataFrames
# Join customer and orders (filter before join)
df_joined_co = pd.merge(df_cust, df_orders_filtered, left_on='c_custkey', right_on='o_custkey', how='inner')

# Join lineitem with the result of customer-orders join
df_joined_l = pd.merge(df_lineitem, df_joined_co, left_on='l_orderkey', right_on='o_orderkey', how='inner')

# Join supplier and apply the filter for c_nationkey == s_nationkey before the join
df_joined_supp = pd.merge(df_joined_l, df_supp, left_on='l_suppkey', right_on='s_suppkey', how='inner')

# Apply the filter for matching nation keys early (before joining with nation and region)
df_joined_supp = df_joined_supp[df_joined_supp['c_nationkey'] == df_joined_supp['s_nationkey']]

# Join with nation and region (filtered earlier for region 'ASIA')
df_joined_nation = pd.merge(df_joined_supp, df_nation, left_on='s_nationkey', right_on='n_nationkey', how='inner')
df_joined_final = pd.merge(df_joined_nation, df_region_filtered, left_on='n_regionkey', right_on='r_regionkey', how='inner')

# Perform the aggregation and sum the revenue by 'n_name'
grouped_sum = df_joined_final.groupby('n_name')['revenue'].sum().reset_index().sort_values(by='revenue')

# End timing
end_time = time.time()

# Print execution time and the shape of the result
print(f"Execution time: {end_time - start_time} seconds")
print(grouped_sum.shape)

# Optionally, display the final result
print(grouped_sum)