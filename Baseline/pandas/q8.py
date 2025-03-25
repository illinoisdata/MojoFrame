import time
import pandas as pd

pd.set_option('display.max_columns', None)

# Query 8
file_path_part = '/home/shengya4/data/tpch_3gb/part.csv'
df_part = pd.read_csv(file_path_part, usecols=['p_partkey', 'p_type'])
print(df_part.shape)

# Load lineitem-med.csv and select relevant columns
file_path_lineitem = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path_lineitem, usecols=['l_orderkey', 'l_extendedprice', 'l_discount', 'l_suppkey', 'l_partkey'])
df_lineitem['volume'] = df_lineitem['l_extendedprice'] * (1 - df_lineitem['l_discount'])

print(df_lineitem.shape)

# Load supplier.csv and select relevant columns
file_path_supplier = '/home/shengya4/data/tpch_3gb/supplier.csv'
df_supplier = pd.read_csv(file_path_supplier, usecols=['s_suppkey', 's_nationkey'])
print(df_supplier.shape)

# Load nation.csv and select relevant columns
file_path_nation = '/home/shengya4/data/tpch_3gb/nation.csv'
df_nation1 = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_name', 'n_regionkey'])
df_nation1.rename(columns={'n_nationkey': 'n_nationkey1', 'n_name': 'n_name1', 'n_regionkey': 'n_regionkey1'}, inplace=True)
print(df_nation1.shape)

file_path_nation = '/home/shengya4/data/tpch_3gb/nation.csv'
df_nation2 = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_name', 'n_regionkey'])
df_nation2.rename(columns={'n_nationkey': 'n_nationkey2', 'n_name': 'n_name2', 'n_regionkey': 'n_regionkey2'}, inplace=True)
print(df_nation2.shape)

# Load customer.csv and select relevant columns
file_path_customer = '/home/shengya4/data/tpch_3gb/customer.csv'
df_customer = pd.read_csv(file_path_customer, usecols=['c_custkey', 'c_nationkey'])
print(df_customer.shape)

# Load orders.csv and select relevant columns
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_custkey', 'o_orderdate'])
print(df_orders.shape)


file_path_region = '/home/shengya4/data/tpch_3gb/region.csv'
df_region = pd.read_csv(file_path_region, usecols=['r_regionkey', 'r_name'])
print(df_region.shape)

start_time = time.time()

# Filter part_df for p_type
df_part_filtered = df_part[df_part['p_type'] == 'ECONOMY ANODIZED STEEL']
# Filter region_df for r_name
df_region_filtered = df_region[df_region['r_name'] == 3070.0]

# Filter orders_df by order date
df_orders_filtered = df_orders[
    (df_orders['o_orderdate'] >= 788918400.0) &
    (df_orders['o_orderdate'] <= 852076800.0)
]


df_joined_pl = pd.merge(df_part_filtered, df_lineitem, left_on='p_partkey', right_on='l_partkey', how='inner')
#print(df_joined_pl.shape)
df_joined_s = pd.merge(df_supplier, df_joined_pl, left_on='s_suppkey', right_on='l_suppkey', how='inner')
#print(df_joined_s.shape)
df_joined_o = pd.merge(df_joined_s, df_orders_filtered, left_on='l_orderkey', right_on='o_orderkey', how='inner')
#print(df_joined_o.shape)
df_joined_c = pd.merge(df_joined_o, df_customer, left_on='o_custkey', right_on='c_custkey', how='inner')
#print(df_joined_c.shape)
df_joined_n1 = pd.merge(df_joined_c, df_nation1, left_on='c_nationkey', right_on='n_nationkey1', how='inner')
#print(df_joined_n1.shape)
df_joined_n1_r = pd.merge(df_joined_n1, df_region_filtered, left_on='n_regionkey1', right_on='r_regionkey', how='inner')
#print(df_joined_n1_r.shape)
all_nations = pd.merge(df_joined_n1_r, df_nation2, left_on='s_nationkey', right_on='n_nationkey2', how='inner')
#print(all_nations.shape)
all_nations['o_orderdate'] = (1970.0 + (all_nations['o_orderdate'] / 31536000.0)).round()

# Group by year and calculate the market share for Brazil
result = all_nations.groupby('o_orderdate').apply(
    lambda x: pd.Series({
        'mkt_share': x.loc[x['n_name2'] == 62514.0, 'volume'].sum() / x['volume'].sum()
    })
).reset_index()


# Sort by o_year
result = result.sort_values(by='o_orderdate')

# End timing
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
print(result)