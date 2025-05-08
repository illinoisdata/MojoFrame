import pandas as pd
import time

start_load = time.perf_counter()
df_lineitem = pd.read_csv('/datadrive/tpch_large/lineitem.csv', usecols=['l_partkey', 'l_quantity', 'l_extendedprice'])
df_part = pd.read_csv('/datadrive/tpch_large/part.csv', usecols=['p_partkey', 'p_brand', 'p_container'])
end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

start_time = time.time()

df_part_filtered = df_part[(df_part['p_brand'] == 'Brand#23') & (df_part['p_container'] == 'MED BOX')]


lineitem_part_joined = pd.merge(
    df_lineitem, 
    df_part_filtered, 
    left_on='l_partkey', 
    right_on='p_partkey',
    how='inner'
)


lineitem_avg_quantity = (
    lineitem_part_joined
    .groupby('l_partkey', as_index=False)
    .agg(avg_quantity=('l_quantity', 'mean'))
)

lineitem_avg_quantity['avg_quantity_scaled'] = 0.2 * lineitem_avg_quantity['avg_quantity']

print(lineitem_avg_quantity.shape)


lineitem_with_threshold = pd.merge(
    lineitem_part_joined, 
    lineitem_avg_quantity,
    on='l_partkey'
)
      

filtered_lineitem = lineitem_with_threshold[
    lineitem_with_threshold['l_quantity'] < lineitem_with_threshold['avg_quantity_scaled']
]

print(filtered_lineitem.shape)


avg_yearly = filtered_lineitem['l_extendedprice'].sum() / 7.0

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
print("avg_yearly:", avg_yearly)