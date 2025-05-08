import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Load DataFrames (assuming the columns are named as in the query)

df_customer = pd.read_csv('/datadrive/tpch_large/customer.csv', usecols=['c_custkey'])
start_load = time.perf_counter()
df_orders = pd.read_csv('/datadrive/tpch_large/orders.csv', usecols=['o_custkey', 'o_comment'])
end_load = time.perf_counter()
print("Data loading time ORD: ", end_load - start_load)

def filter_not_string_exists_before(comment, filter_str1, filter_str2):
    # Find the index of the first occurrence of `filter_str1`
    first_pos_str1 = comment.find(filter_str1)
    # Find the index of the last occurrence of `filter_str2`
    last_pos_str2 = comment.rfind(filter_str2)
    
    # Return True if `filter_str1` does not occur before `filter_str2`
    return not (first_pos_str1 != -1 and last_pos_str2 != -1 and first_pos_str1 < last_pos_str2)


start_time = time.time()

# 16.478629s with regular expression
# df_orders_filtered = df_orders[~(df_orders['o_comment'].str.contains(".*special.*requests.*", regex=True, na=False))]
filter_str1 = "special"
filter_str2 = "requests"

# 2.765572071s with custom lambda function filtering
df_orders_filtered = df_orders[df_orders['o_comment'].apply(lambda comment: filter_not_string_exists_before(comment, filter_str1, filter_str2))]

# Step 1: Perform an inner join on `c_custkey = o_custkey`
joined_co_df = pd.merge(df_customer, df_orders_filtered, left_on='c_custkey', right_on='o_custkey', how='inner')

print("joined_co_df size:", joined_co_df.shape)
# Step 3: Group by `c_custkey` and count the number of orders for each customer
customer_order_counts = joined_co_df.groupby('c_custkey').size().reset_index(name='c_count')

# Step 4: Group by `c_count` to get the distribution of customers by their order count
order_count_distribution = customer_order_counts.groupby('c_count').size().reset_index(name='custdist')

# Step 5: Sort by `custdist` and `c_count` in descending order
result_df = order_count_distribution.sort_values(by=['custdist', 'c_count'])

end_time = time.time()

print(result_df.shape)
print(result_df)
print(f"Execution time: {end_time - start_time} seconds")