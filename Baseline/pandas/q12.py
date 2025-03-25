import time
import pandas as pd

# Load the DataFrames
# Load `lineitem-med.csv` with the required columns
file_path_lineitem = '/home/shengya4/data/tpch_3gb/lineitem-med.csv'
df_lineitem = pd.read_csv(file_path_lineitem, usecols=['l_orderkey', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipmode'])
print(df_lineitem.head())
print("Shape of df_lineitem:", df_lineitem.shape)

# Load `orders.csv` with the required columns
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=['o_orderkey', 'o_orderpriority'])
print(df_orders.head())
print("Shape of df_orders:", df_orders.shape)

start_time = time.time()

# Filter `lineitem` based on conditions
filtered_lineitem = df_lineitem[
    (df_lineitem['l_shipmode'].isin([5.0, 6.0])) &
    (df_lineitem['l_commitdate'] < df_lineitem['l_receiptdate']) &
    (df_lineitem['l_shipdate'] < df_lineitem['l_commitdate']) &
    (df_lineitem['l_receiptdate'] >= 757382400.0) &
    (df_lineitem['l_receiptdate'] < 788918400.0)
]

# Merge the filtered `lineitem` DataFrame with `orders` on the order key
joined_lo_df = pd.merge(
    filtered_lineitem, df_orders,
    left_on='l_orderkey', right_on='o_orderkey', how='inner'
)

# Add columns for high and low priority line counts
joined_lo_df['high_line'] = joined_lo_df['o_orderpriority'].apply(
    lambda x: 1 if x in [1.0, 2.0] else 0
)
joined_lo_df['low_line'] = joined_lo_df['o_orderpriority'].apply(
    lambda x: 1 if x not in [1.0, 2.0] else 0
)

# Group by `l_shipmode` and aggregate high and low line counts
result_df = joined_lo_df.groupby('l_shipmode').agg(
    'sum'
).reset_index()

# Sort the result by `l_shipmode`
result_df = result_df.sort_values(by='l_shipmode')

end_time = time.time()

print(result_df.shape)
print(f"Execution time: {end_time - start_time} seconds")
print(result_df)