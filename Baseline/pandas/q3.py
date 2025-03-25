import time
import pandas as pd

pd.set_option('display.max_columns', None)

start_time = time.perf_counter()

# Load and filter customer data
file_path_customer = '/home/shengya4/data/tpch_3gb/customer.csv'
df_customer = pd.read_csv(file_path_customer, usecols=['c_custkey', 'c_mktsegment'])

# Load and filter orders data
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_custkey', 'o_orderkey', 'o_orderdate', 'o_shippriority'])

# Load and calculate revenue in lineitem data
file_path_lineitem = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path_lineitem, usecols=['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_returnflag', 'l_shipdate'])
df_lineitem['revenue'] = df_lineitem['l_extendedprice'] * (1 - df_lineitem['l_discount'])


print(df_lineitem)

end_time = time.perf_counter()

print("Data loading time: ", end_time - start_time)

print("Customer shape:", df_customer.shape)
print("Orders shape:", df_orders.shape)
print("Lineitem shape:", df_lineitem.shape)



start_time = time.perf_counter()

codes, _ = pd.factorize(df_lineitem['l_orderkey'])
print(codes)

end_time = time.perf_counter()

print("Factorize time: ", end_time - start_time)


# Start timing the process
start_time = time.perf_counter()

# Step 1: Filter customers where c_mktsegment == 1.0 before joining
df_customer_filtered = df_customer[df_customer['c_mktsegment'] == 1.0]

# Step 2: Filter orders where o_orderdate < 794880000.0 before joining
df_orders_filtered = df_orders[df_orders['o_orderdate'] < 794880000.0]

# Step 3: Filter lineitem where l_shipdate > 794880000.0 before joining
df_lineitem_filtered = df_lineitem[df_lineitem['l_shipdate'] > 794880000.0]

# Step 4: Perform the join between filtered customer and orders
df_merged = pd.merge(df_customer_filtered, df_orders_filtered, left_on='c_custkey', right_on='o_custkey', how='inner')

# Step 5: Perform the join with lineitem after filtering
df_final = pd.merge(df_lineitem_filtered, df_merged, left_on='l_orderkey', right_on='o_orderkey', how='inner')

# df_merged = pd.merge(df_customer_filtered, df_orders_filtered, left_on='c_custkey', right_on='o_orderkey', how='inner')

# # Step 5: Perform the join with lineitem after filtering
# df_final = pd.merge(df_lineitem_filtered, df_merged, left_on='l_orderkey', right_on='o_orderkey', how='inner')

# print("joined final shape", df_final.shape)
# print(df_final)
# print(df_final['l_orderkey_unique'].cat.codes)
agg_funcs = {col: 'sum' for col in df_final.columns if col not in ['l_orderkey', 'o_orderdate', 'o_shippriority', 'c_custkey_unique', 'o_custkey_unique', 'o_orderkey_unique', 'l_orderkey_unique']}
result = df_final.groupby(['l_orderkey', 'o_orderdate', 'o_shippriority']).agg(
    agg_funcs
).reset_index().sort_values(by=['revenue', 'o_orderdate'])

# End timing
end_time = time.perf_counter()

# Output the results
print("Execution time:", end_time - start_time)
print("Result shape:", result.shape)
print(result)
# print(df_lineitem.shape)