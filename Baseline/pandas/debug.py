import pandas as pd
import pdb



# Sample DataFrames
df_customer = pd.DataFrame({'c_custkey': [2.0, 1.0, 3.0]})
df_orders = pd.DataFrame({'o_custkey': [4.0, 3.0, 2.0]})

# Create a shared category mapping
unique_custkeys = pd.concat([
    df_customer['c_custkey'], 
    df_orders['o_custkey']
], ignore_index=True).drop_duplicates()

print("unique_custkeys:\n", unique_custkeys)

# Convert both columns to categorical with the same categories
df_customer['c_custkey'] = pd.Categorical(df_customer['c_custkey'], categories=unique_custkeys)
df_orders['o_custkey'] = pd.Categorical(df_orders['o_custkey'], categories=unique_custkeys)

print(df_orders['o_custkey'].cat.codes)

def f():
    df_merged = df_customer.merge(df_orders, left_on='c_custkey', right_on='o_custkey', how='inner')

pdb.run("f()")