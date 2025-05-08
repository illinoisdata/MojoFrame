### Query 20

import pandas as pd
import time

# Step 1: Load only necessary columns from the CSV files
start_load = time.perf_counter()
df_lineitem = pd.read_csv('/datadrive/tpch_large/lineitem.csv', usecols=['l_partkey', 'l_suppkey', 'l_shipdate', 'l_quantity'])
df_part = pd.read_csv('/datadrive/tpch_large/part.csv', usecols=['p_partkey', 'p_name'])
df_supplier = pd.read_csv('/datadrive/tpch_large/supplier.csv', usecols=['s_suppkey', 's_nationkey', 's_name'])
df_nation = pd.read_csv('/datadrive/tpch_large/nation.csv', usecols=['n_nationkey', 'n_name'])
df_partsupp = pd.read_csv('/datadrive/tpch_large/partsupp.csv', usecols=['ps_suppkey', 'ps_partkey', 'ps_availqty'])
end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

start_time = time.time()

# Step 2: Filter `part` where `p_name` starts with 'forest'
filtered_part = df_part[df_part['p_name'].str.startswith('forest')]

# Step 3: Join `partsupp` with filtered `part` on `p_partkey = ps_partkey`
joined_psp = pd.merge(
    df_partsupp,
    filtered_part,
    left_on='ps_partkey',
    right_on='p_partkey',
    how='inner'
)

# Step 4: Filter `lineitem` by date range and group by `l_partkey`, `l_suppkey` to calculate sum of `l_quantity`

filtered_lineitem = df_lineitem[
    (df_lineitem['l_shipdate'] >= 757382400.0) &
    (df_lineitem['l_shipdate'] < 788918400.0)
]

print("lineitem after filter: ", filtered_lineitem.shape)

lineitem_agg = (
    filtered_lineitem
    .groupby(['l_partkey', 'l_suppkey'], as_index=False)
    .agg('sum')
)

# Step 5: Join `joined_psp` with `lineitem_agg` on `ps_partkey = l_partkey` and filter where `ps_availqty > 0.5 * SUM(l_quantity)`
joined_lineitem_psp = pd.merge(
    joined_psp,
    lineitem_agg,
    left_on=['ps_partkey', 'ps_suppkey'],
    right_on=['l_partkey', 'l_suppkey'],
    how='inner'
)

filtered_psp = joined_lineitem_psp[
    joined_lineitem_psp['ps_availqty'] > 0.5 * joined_lineitem_psp['l_quantity']
]

# Step 6: Join `filtered_psp` with `supplier` on `ps_suppkey = s_suppkey`
joined_supplier = pd.merge(
    filtered_psp,
    df_supplier,
    left_on='ps_suppkey',
    right_on='s_suppkey',
    how='inner'
)

# Step 7: Join `joined_supplier` with `nation` on `s_nationkey = n_nationkey` and filter where `n_name = 'CANADA'`
joined_nation = pd.merge(
    joined_supplier,
    df_nation,
    left_on='s_nationkey',
    right_on='n_nationkey'
)

filtered_nation = joined_nation[joined_nation['n_name'] == 35480.0]

print("final joined size: ", filtered_nation.shape)

# Step 8: Select required columns and sort by `s_name`
result = (
    filtered_nation.sort_values(by='s_name')
)

end_time = time.time()

# Display the result
print(f"Execution time: {end_time - start_time} seconds")
print(result)