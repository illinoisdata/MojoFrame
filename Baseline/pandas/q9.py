import time
import pandas as pd

# Load `part.csv` with selected columns
file_path_part = '/home/shengya4/data/tpch_3gb/part.csv'
df_part = pd.read_csv(file_path_part, usecols=['p_partkey', 'p_name'])
print(df_part.head())
print("Shape of df_pt:", df_part.shape)

# Load `lineitem-med.csv` with selected columns
file_path = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path, usecols=['l_orderkey', 'l_extendedprice', 'l_discount', 'l_quantity', 'l_suppkey', 'l_partkey'])
print(df_lineitem.head())
print("Shape of df_lineitem:", df_lineitem.shape)

# Load `partsupp.csv` with selected columns
file_path_psupp = '/home/shengya4/data/tpch_3gb/partsupp.csv'
df_partsupp = pd.read_csv(file_path_psupp, usecols=['ps_partkey', 'ps_suppkey', 'ps_supplycost'])
print(df_partsupp.head())
print("Shape of df_psupp:", df_partsupp.shape)

# Load `supplier.csv` with selected columns
file_path_supp = '/home/shengya4/data/tpch_3gb/supplier.csv'
df_supplier = pd.read_csv(file_path_supp, usecols=['s_suppkey', 's_nationkey'])
print(df_supplier.head())
print("Shape of df_supp:", df_supplier.shape)

# Load `nation.csv` with selected columns
file_path_nation = '/home/shengya4/data/tpch_3gb/nation.csv'
df_nation = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_name'])
print(df_nation.head())
print("Shape of df_nat:", df_nation.shape)

# Load `orders.csv` with selected columns
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_orderdate'])
print(df_orders.head())
print("Shape of df_orders:", df_orders.shape)

start_time = time.time()

df_part_filtered = df_part[df_part['p_name'].str.contains("green", case=True, na=False)]

# Perform the initial joins in sequence

# Step 1: Join `df_supplier` with `df_nation` on `s_nationkey = n_nationkey`
joined_sn_df = pd.merge(df_supplier, df_nation, left_on='s_nationkey', right_on='n_nationkey', how='inner')
print("After joining supplier and nation:", joined_sn_df.shape)

# Step 2: Join `df_lineitem` with `joined_sn_df` on `l_suppkey = s_suppkey`
joined_ls_df = pd.merge(df_lineitem, joined_sn_df, left_on='l_suppkey', right_on='s_suppkey', how='inner')
print("After joining lineitem and supplier-nation:", joined_ls_df.shape)

# Step 3: Join `joined_ls_df` with `df_part` on `l_partkey = p_partkey`
joined_lp_df = pd.merge(joined_ls_df, df_part_filtered, left_on='l_partkey', right_on='p_partkey', how='inner')
print("After joining with part:", joined_lp_df.shape)

# Step 4: Join `joined_lp_df` with `df_partsupp` on `l_partkey = ps_partkey`
# Then filter rows where `suppkey` matches `ps_suppkey`
joined_lps_df = pd.merge(
    joined_lp_df, 
    df_partsupp, 
    left_on=['l_partkey', 'l_suppkey'], 
    right_on=['ps_partkey', 'ps_suppkey'], 
    how='inner'
)
print("After joining with partsupp and filtering suppkey match:", joined_lps_df.shape)

# Step 5: Join `joined_lps_df` with `df_orders` on `l_orderkey = o_orderkey`
joined_final = pd.merge(joined_lps_df, df_orders, left_on='l_orderkey', right_on='o_orderkey', how='inner')
print("Final joined result:", joined_final.shape)


joined_final['sum_profit'] = (joined_final['l_extendedprice'] * (1 - joined_final['l_discount'])) - \
                          (joined_final['ps_supplycost'] * joined_final['l_quantity'])

joined_final['o_orderdate'] = (1970.0 + (joined_final['o_orderdate'] / 31536000.0)).round()

result = joined_final.groupby(['n_name', 'o_orderdate']).agg(
    'sum'
).reset_index().sort_values(by=['n_name', 'o_orderdate'])

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
print(result)