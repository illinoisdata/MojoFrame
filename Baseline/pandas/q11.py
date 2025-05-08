import time
import pandas as pd

# Load the DataFrames
load_start = time.perf_counter()
df_partsupp = pd.read_csv('/datadrive/tpch_large/partsupp.csv', usecols=['ps_partkey', 'ps_suppkey', 'ps_supplycost', 'ps_availqty'])
df_supplier = pd.read_csv('/datadrive/tpch_large/supplier.csv', usecols=['s_suppkey', 's_nationkey'])
df_nation = pd.read_csv('/datadrive/tpch_large/nation.csv', usecols=['n_nationkey', 'n_name'])
load_end = time.perf_counter()
print("Data loading time: ", load_end - load_start)
start_time = time.time()

# Step 2: Filter rows where `n_name` is 'GERMANY'
df_nation_filtered = df_nation[df_nation['n_name'] == 52342.0]

# Step 1: Merge `partsupp`, `supplier`, and `nation` DataFrames on their respective keys
joined_sn_df = pd.merge(df_supplier, df_nation_filtered, left_on='s_nationkey', right_on='n_nationkey', how='inner')
joined_pss_df = pd.merge(df_partsupp, joined_sn_df, left_on='ps_suppkey', right_on='s_suppkey', how='inner')

# Step 3: Calculate `value` as `ps_supplycost * ps_availqty`
joined_pss_df['value'] = joined_pss_df['ps_supplycost'] * joined_pss_df['ps_availqty']

# Step 4: Calculate `value_percent` as `value * 0.0001`
joined_pss_df['value_percent'] = joined_pss_df['value'] * 0.00001

# Step 5: Group by `ps_partkey` and calculate the sum of `value`
grouped_df = joined_pss_df.groupby('ps_partkey').agg(
    'sum'
).reset_index()

# Step 6: Calculate the overall sum of `value_percent` for the HAVING condition
value_percent_sum = joined_pss_df['value_percent'].sum()

# Step 7: Apply the HAVING condition: `value_sum > value_percent_sum`
filtered_df = (grouped_df[grouped_df['value'] > value_percent_sum])
filtered_df = filtered_df.sort_values(by=['value'])

end_time = time.time()
print(joined_pss_df.shape)
print(grouped_df.shape)
print("Percent value sum:", value_percent_sum)
print(f"Execution time: {end_time - start_time} seconds")
print(filtered_df)