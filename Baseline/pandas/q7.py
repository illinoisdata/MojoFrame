import time
import pandas as pd

pd.set_option('display.max_columns', None)

# Load your datasets as before, keeping the necessary columns
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_custkey'])

file_path_lineitem = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path_lineitem, usecols=['l_orderkey', 'l_extendedprice', 'l_discount', 'l_suppkey', 'l_shipdate'])

# Calculate revenue early, as in original code
df_lineitem['revenue'] = df_lineitem['l_extendedprice'] * (1 - df_lineitem['l_discount'])

file_path_cust = '/home/shengya4/data/tpch_3gb/customer.csv'
df_cust = pd.read_csv(file_path_cust, usecols=['c_custkey', 'c_nationkey'])

file_path_supp = '/home/shengya4/data/tpch_3gb/supplier.csv'
df_supp = pd.read_csv(file_path_supp, usecols=['s_suppkey', 's_nationkey'])

file_path_nation = '/home/shengya4/data/tpch_3gb/nation.csv'
df_nation1 = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_name'])
df_nation1.rename(columns={'n_nationkey': 'n_nationkey1', 'n_name': 'n_name1'}, inplace=True)


file_path_nation = '/home/shengya4/data/tpch_3gb/nation.csv'
df_nation2 = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_name'])
df_nation2.rename(columns={'n_nationkey': 'n_nationkey2', 'n_name': 'n_name2'}, inplace=True)

# Start filtering before the joins
start_time = time.time()

# Filter orders between 1994-01-01 and 1995-01-01 using float Unix time
df_lineitem = df_lineitem[
    (df_lineitem['l_shipdate'] >= 788918400.0) & (df_lineitem['l_shipdate'] <= 852076800.0)
]

# Perform the joins with filtered DataFrames
df_joined_ls = pd.merge(df_supp, df_lineitem, left_on='s_suppkey', right_on='l_suppkey', how='inner')

df_joined_o = pd.merge(df_joined_ls, df_orders, left_on='l_orderkey', right_on='o_orderkey', how='inner')

df_joined_c = pd.merge(df_joined_o, df_cust, left_on='o_custkey', right_on='c_custkey', how='inner')

df_joined_n1 = pd.merge(df_joined_c, df_nation1, left_on='s_nationkey', right_on='n_nationkey1', how='inner')
df_shipping = pd.merge(df_joined_n1, df_nation2, left_on='c_nationkey', right_on='n_nationkey2', how='inner')

df_shipping = df_shipping[
    ((df_shipping['n_name1'] == 38075.0) & (df_shipping['n_name2'] == 52342.0)) | \
    ((df_shipping['n_name1'] == 52342.0) & (df_shipping['n_name2'] == 38075.0))
]
df_shipping['l_shipdate'] = (1970.0 + (df_shipping['l_shipdate'] / 31536000.0)).round()

result = df_shipping.groupby(['n_name1', 'n_name2', 'l_shipdate']).agg(
    'sum'
).reset_index().sort_values(by=['n_name1', 'n_name2', 'l_shipdate'])


end_time = time.time()

print("result shape:", result.shape)
print(result)
# End timing


# Print execution time and the shape of the result
print(f"Execution time: {end_time - start_time} seconds")