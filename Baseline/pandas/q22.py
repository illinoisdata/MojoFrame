import pandas as pd
import time
import numpy as np 


# Load customer data
file_path_customer = '/home/shengya4/data/tpch_3gb/customer.csv'
df_customer = pd.read_csv(file_path_customer, usecols=["c_custkey", "c_nationkey", "c_acctbal", "c_phone"])
print(df_customer.head())
print("Customer DataFrame shape:", df_customer.shape)

# Load orders data
file_path_orders = '/home/shengya4/data/tpch_3gb/orders.csv'
df_orders = pd.read_csv(file_path_orders, usecols=["o_custkey", "o_orderkey"])
print(df_orders.head())
print("Orders DataFrame shape:", df_orders.shape)

# Measure start time
start_time = time.time()

var_list = [13.0, 31.0, 23.0, 29.0, 30.0, 18.0, 17.0]

avg_c_acctbal = (
    df_customer[
        (df_customer["c_acctbal"] > 0)
        & (df_customer["c_phone"].str[:2].astype(float).isin(var_list))
    ]["c_acctbal"]
    .mean()
)

print("avg_c_acctbal: ", avg_c_acctbal)

# 2) Create a cntrycode column by slicing the first two digits of c_phone, and filtering
df_customer["cntrycode"] = df_customer["c_phone"].str[:2].astype(float)

filtered_df = df_customer[
    (df_customer["cntrycode"].isin(var_list)) 
    & (df_customer["c_acctbal"] > avg_c_acctbal)
]

print(filtered_df.shape)

joined_df = pd.merge(
    filtered_df,
    df_orders,
    left_on="c_custkey",  # Customer key in df_customer
    right_on="o_custkey",  # Customer key in df_orders
    how="left"
)

print(joined_df.shape)


# Filter rows where there are no matching orders (i.e., `o_orderkey` is NaN)
no_orders_df = joined_df[joined_df["o_orderkey"].isna()]


q_final = (
        no_orders_df.groupby("cntrycode", as_index=False)
        .agg(
            numcust=("c_custkey", "count"),
            totacctbal=("c_acctbal", "sum")
        )
        .sort_values("cntrycode")
    )

# Measure end time
end_time = time.time()
execution_time_seconds = end_time - start_time

print("Execution time:", execution_time_seconds, "seconds")

print(q_final)