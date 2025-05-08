import time
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
start_load = time.perf_counter()
file_path_new = '/datadrive/tpch_large/lineitem.csv'
lineitem_df_new = pd.read_csv(file_path_new, usecols=['l_quantity', 'l_extendedprice', 'l_discount', 'l_shipdate'])
end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

print(lineitem_df_new.shape)
print(lineitem_df_new.head())
# Define the filter conditions using timestamps

start_time = time.time()

start_timestamp = 757382400.0
end_timestamp = 788918400.0
max_quantity = 24

filtered_df_new = lineitem_df_new[
    (lineitem_df_new['l_shipdate'] >= start_timestamp) &
    (lineitem_df_new['l_shipdate'] < end_timestamp) &
    # 0.7 <= 0.0699999?
    (lineitem_df_new['l_discount'] >= (0.05)) &
    (lineitem_df_new['l_discount'] <= (0.07)) &
    (lineitem_df_new['l_quantity'] < max_quantity)
]

# Calculate the revenue
revenue_new = (filtered_df_new['l_extendedprice'] * filtered_df_new['l_discount']).sum()

end_time = time.time()
print(filtered_df_new.shape)
print("revenue: ", revenue_new)
print("runtime: ", end_time - start_time)