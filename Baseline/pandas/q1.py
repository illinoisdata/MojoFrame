import time
import pandas as pd


pd.set_option('display.max_columns', None)

start_time = time.perf_counter()

file_path_new = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
lineitem_df_new = pd.read_csv(file_path_new, usecols=["l_quantity", "l_extendedprice", "l_discount", "l_returnflag",
                                                        "l_shipdate", "l_linestatus", "l_tax"])

end_time = time.perf_counter()

lineitem_df_new["l_discprice"] = lineitem_df_new["l_extendedprice"] * (1 - lineitem_df_new["l_discount"])
lineitem_df_new["l_charge"] = lineitem_df_new["l_discprice"] * (1 + lineitem_df_new["l_tax"])

print("Data loading time: ", end_time - start_time)


start_time = time.perf_counter()

cutoff_date_unix = 904608000.0  # September 2, 1998

# Filter the DataFrame based on the date condition
filtered_df = lineitem_df_new[lineitem_df_new['l_shipdate'] <= cutoff_date_unix]

# Group by 'l_returnflag' and perform aggregations
result = filtered_df.groupby(["l_returnflag", "l_linestatus"]).agg(
    sum_qty=('l_quantity', 'sum'),
    sum_price=('l_extendedprice', 'sum'),
    sum_disc=('l_discount', 'sum'),
    sum_date=('l_shipdate', 'sum'),
    sum_tax=('l_tax', 'sum'),
    sum_discprice=('l_discprice', 'sum'),
    sum_charge=('l_charge', 'sum'),
    avg_qty=('l_quantity', 'mean'),
    avg_price=('l_extendedprice', 'mean'),
    avg_disc=('l_discount', 'mean'),
    avg_date=('l_shipdate', 'mean'),
    avg_tax=('l_tax', 'mean'),
    avg_discprice=('l_discprice', 'mean'),
    avg_charge=('l_charge', 'mean'),
    count_order=('l_quantity', 'size')  # using size to count entries
).reset_index()

end_time = time.perf_counter()
# Display the result
print(result)
print(result.shape)
print("runtime: ", end_time - start_time)   