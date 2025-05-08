import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Step 1: Load only necessary columns from CSV files
start_load = time.perf_counter()
df_lineitem = pd.read_csv('/datadrive/tpch_large/lineitem.csv', usecols=['l_partkey', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_shipinstruct', 'l_shipmode'])
end_load = time.perf_counter()
print("Data loading time lineitem: ", end_load - start_load)
df_part = pd.read_csv('/datadrive/tpch_large/part.csv', usecols=['p_partkey', 'p_brand', 'p_size', 'p_container'])



start_time = time.time()

# Step 2: Filter `lineitem` based on `l_shipmode` and `l_shipinstruct`
lineitem_filtered = df_lineitem[
    (df_lineitem['l_shipinstruct'] == 'DELIVER IN PERSON') &
    (df_lineitem['l_shipmode'].isin([2.0, 4.0]))
]

# Step 3: Merge `lineitem_filtered` with `part` on `l_partkey = p_partkey`
joined_lp_df = pd.merge(lineitem_filtered, df_part, left_on='l_partkey', right_on='p_partkey', how='inner')

# Step 4: Apply the conditions for `p_brand`, `p_container`, `p_size`, and `l_quantity`
filtered_df = joined_lp_df[
    (
        (joined_lp_df['p_brand'] == 'Brand#12') &
        (joined_lp_df['p_container'].isin(['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'])) &
        (joined_lp_df['p_size'].between(1, 5)) &
        (joined_lp_df['l_quantity'].between(1, 11))
    ) |
    (
        (joined_lp_df['p_brand'] == 'Brand#23') &
        (joined_lp_df['p_container'].isin(['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'])) &
        (joined_lp_df['p_size'].between(1, 10)) &
        (joined_lp_df['l_quantity'].between(10, 20))
    ) |
    (
        (joined_lp_df['p_brand'] == 'Brand#34') &
        (joined_lp_df['p_container'].isin(['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'])) &
        (joined_lp_df['p_size'].between(1, 15)) &
        (joined_lp_df['l_quantity'].between(20, 30))
    )
]

# Step 5: Calculate the revenue
disc_price = filtered_df['l_extendedprice'] * (1 - filtered_df['l_discount'])

# Step 6: Aggregate to calculate the total revenue
revenue = disc_price.sum()

end_time = time.time()

# Display the result
print(f"Execution time: {end_time - start_time} seconds")

print(filtered_df.shape)
# Display the result
print(f"Total Revenue: {revenue}")