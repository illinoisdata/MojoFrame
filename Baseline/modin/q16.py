import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Load only the necessary columns from each CSV
start_load = time.perf_counter()

df_supplier = pd.read_csv('/datadrive/tpch_large/supplier.csv', usecols=['s_suppkey', 's_comment'])
df_part = pd.read_csv('/datadrive/tpch_large/part.csv', usecols=['p_partkey', 'p_brand', 'p_type', 'p_size'])
df_partsupp = pd.read_csv('/datadrive/tpch_large/partsupp.csv', usecols=['ps_partkey', 'ps_suppkey'])

end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

def filter_not_string_exists_before(comment, filter_str1, filter_str2):
    # Find the index of the first occurrence of `filter_str1`
    first_pos_str1 = comment.find(filter_str1)
    # Find the index of the last occurrence of `filter_str2`
    last_pos_str2 = comment.rfind(filter_str2)
    
    # Return True if `filter_str1` does not occur before `filter_str2`
    return not (first_pos_str1 != -1 and last_pos_str2 != -1 and first_pos_str1 < last_pos_str2)

start_time = time.time()

df_part_filtered = df_part[
    (df_part['p_brand'] != 'Brand#45') &
    (~df_part['p_type'].str.startswith('MEDIUM POLISHED')) &
    (df_part['p_size'].isin([49, 14, 23, 45, 19, 3, 36, 9]))
]

print(df_part_filtered.shape)

df_supplier_filtered = df_supplier[df_supplier['s_comment'].apply(lambda comment: filter_not_string_exists_before(comment, "Customer", "Complaints"))]

print(df_supplier_filtered.shape)

joined_pss_df = pd.merge(
    df_partsupp,
    df_supplier_filtered,
    left_on='ps_suppkey',
    right_on='s_suppkey',
    how='inner'
)
print(joined_pss_df.shape)


joined_psp_df = pd.merge(
    joined_pss_df,
    df_part_filtered,
    left_on='ps_partkey',
    right_on='p_partkey',
    how='inner'
)
print(joined_psp_df.shape)


result = (
    joined_psp_df.groupby('p_size')
    .agg(supplier_count=('ps_suppkey', 'nunique'))
    .reset_index()
)


result = result.sort_values(by=['supplier_count', 'p_size'], ascending=[False, False]).reset_index(drop=True)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
print(result)