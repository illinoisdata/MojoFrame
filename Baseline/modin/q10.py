### Query 10
import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Load `lineitem-med.csv` with selected columns
start_load_time = time.perf_counter()

file_path = '/datadrive/tpch_large/lineitem.csv'
df_lineitem = pd.read_csv(file_path, usecols=['l_orderkey', 'l_extendedprice', 'l_discount', 'l_returnflag'])
print(df_lineitem.head())
print("Shape of df_lineitem:", df_lineitem.shape)

# Load `nation.csv` with selected columns
file_path_nation = '/datadrive/tpch_large/nation.csv'
df_nation = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_name'])
print(df_nation.head())
print("Shape of df_nation:", df_nation.shape)

# Load `orders.csv` with selected columns
file_path_orders = '/datadrive/tpch_large/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_custkey', 'o_orderdate'])
print(df_orders.head())
print("Shape of df_orders:", df_orders.shape)

# Load `customer.csv` with selected columns
file_path_customer = '/datadrive/tpch_large/customer.csv'
df_customer = pd.read_csv(file_path_customer, usecols=['c_custkey', 'c_nationkey', 'c_acctbal'])
print(df_customer.head())
print("Shape of df_customer:", df_customer.shape)

end_load_time = time.perf_counter()
print("Data loading time: ", end_load_time - start_load_time)

start_time = time.time()

df_lineitem_filtered = df_lineitem[df_lineitem['l_returnflag'] == 82.0]
df_orders_filtered = df_orders[(df_orders['o_orderdate'] >= 750643200.0) &
                               (df_orders['o_orderdate'] < 757382400.0)]


# Step 1: Join `df_customer` with `df_nation` on `nationkey`
joined_cn_df = pd.merge(df_customer, df_nation, left_on='c_nationkey', right_on='n_nationkey', how='inner')
print("After joining customer and nation:", joined_cn_df.shape)

# Step 2: Join `df_orders` with `joined_cn_df` on `custkey`
joined_oc_df = pd.merge(df_orders_filtered, joined_cn_df, left_on='o_custkey', right_on='c_custkey', how='inner')
print("After joining orders and customer-nation:", joined_oc_df.shape)

# Step 3: Join `df_lineitem` with `joined_oc_df` on `orderkey`
joined_final = pd.merge(df_lineitem_filtered, joined_oc_df, left_on='l_orderkey', right_on='o_orderkey', how='inner')
print("After joining lineitem with orders-customer-nation:", joined_final.shape)

joined_final['revenue'] = joined_final['l_extendedprice'] * (1 - joined_final['l_discount'])

result = joined_final.groupby(['c_custkey', 'c_acctbal', 'n_name']).agg(
    'sum'
).reset_index().sort_values(by=['revenue'])

end_time = time.time()
print(joined_final.shape)
print(result.shape)
print(f"Execution time: {end_time - start_time} seconds")
print(result)