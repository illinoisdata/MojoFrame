import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

pd.set_option('display.max_columns', None)

start_time = time.perf_counter()

file_path_partsupp = '/datadrive/tpch_large/partsupp.csv'
df_partsupp = pd.read_csv(file_path_partsupp, usecols=['ps_partkey', 'ps_suppkey', 'ps_supplycost'])

end_time = time.perf_counter()
print("Time taken to load data PARTSUPP:", end_time - start_time)

file_path_part = '/datadrive/tpch_large/part.csv'
df_part = pd.read_csv(file_path_part, usecols=['p_partkey', 'p_size', 'p_type'])

file_path_supp = '/datadrive/tpch_large/supplier.csv'
df_supp = pd.read_csv(file_path_supp, usecols=['s_suppkey', 's_nationkey', 's_acctbal', 's_name'])

file_path_nation = '/datadrive/tpch_large/nation.csv'
df_nation = pd.read_csv(file_path_nation, usecols=['n_nationkey', 'n_regionkey', 'n_name'])

file_path_region = '/datadrive/tpch_large/region.csv'
df_region = pd.read_csv(file_path_region, usecols=['r_regionkey', 'r_name'])



print(df_partsupp.shape)
print(df_part.shape)
print(df_supp.shape)
print(df_nation.shape)
print(df_region.shape)


start_time = time.time()

df_region_filtered = df_region[
    # (df_region['r_name'] == 35796.0) | (df_region['r_name'] == 43715.0)
    (df_region['r_name'] == 35796.0)
]


##### Run on pre-conversion datasets..
df_part_filtered_outer = df_part[
    (df_part['p_type'].str.endswith('BRASS')) & ((df_part['p_size'] == 15))

]
print("df_part filtered shape:", df_part_filtered_outer.shape)
# df_partsupp_filtered_outer = df_partsupp[
#     df_partsupp['ps_supplycost'] == min_suppcost
# ]
# df_region_filtered_outer = df_region[
#     (df_region['r_name'] == 35796.0) | (df_region['r_name'] == 43715.0)
# ]


# Step 1: Start with the region and nation join
joined_region_nation = pd.merge(df_region_filtered, df_nation, left_on='r_regionkey', right_on='n_regionkey', how='inner')
print("joined_region_nation:", joined_region_nation.shape)

# Step 2: Join with the supplier table
joined_nation_supplier = pd.merge(joined_region_nation, df_supp, left_on='n_nationkey', right_on='s_nationkey', how='inner')
print("joined_nation_supplier:", joined_nation_supplier.shape)

# Step 3: Join with partsupp table to link part and supplier information
joined_partsupp_supplier = pd.merge(joined_nation_supplier, df_partsupp, left_on='s_suppkey', right_on='ps_suppkey', how='inner')
print("joined_partsupp_supplier:", joined_partsupp_supplier.shape)

# Step 4: Finally, join with the filtered part table to complete the join chain
joined_df_final = pd.merge(joined_partsupp_supplier, df_part_filtered_outer, left_on='ps_partkey', right_on='p_partkey', how='inner')
print("joined_df_final:", joined_df_final.shape)


min_supplycost = joined_df_final.groupby('ps_partkey')['ps_supplycost'].min().reset_index()
min_supplycost.rename(columns={'ps_partkey': 'ps_partkey_grp', 'ps_supplycost': 'ps_supplycost_min'}, inplace=True)

min_supplycost_final = pd.merge(joined_df_final, min_supplycost, left_on='ps_partkey', right_on='ps_partkey_grp', how='inner')
min_supplycost_final = (min_supplycost_final[min_supplycost_final['ps_supplycost'] == min_supplycost_final['ps_supplycost_min']]) \
    .sort_values(by=['s_acctbal', 'n_name', 's_name', 'ps_partkey'])
#joined_df_final = joined_df_final.sort_values(by='s_acctbal')
end_time = time.time()

print(end_time - start_time)
#print(joined_reg_df.shape)
print(min_supplycost_final.shape)
print(min_supplycost_final)