import time
import modin.pandas as pd
import ray

ray.init(num_cpus=8) 

# Load the necessary columns from lineitem and part tables
start_load = time.perf_counter()

df_lineitem = pd.read_csv('/datadrive/tpch_large/lineitem.csv', usecols=['l_partkey', 'l_extendedprice', 'l_discount', 'l_shipdate'])
df_part = pd.read_csv('/datadrive/tpch_large/part.csv', usecols=['p_partkey', 'p_type'])

end_load = time.perf_counter()
print("Data loading time: ", end_load - start_load)

start_time = time.time()

df_lineitem_filtered = df_lineitem[(df_lineitem['l_shipdate'] >= 809913600.0) & (df_lineitem['l_shipdate'] < 812505600.0)]

joined_lp_df = pd.merge(df_lineitem_filtered, df_part, left_on='l_partkey', right_on='p_partkey', how='inner')


joined_lp_df['total_revenue'] = joined_lp_df['l_extendedprice'] * (1 - joined_lp_df['l_discount'])
total_revenue_sum = joined_lp_df['total_revenue'].sum()


promo_revenue_sum = joined_lp_df.loc[joined_lp_df['p_type'].str.startswith('PROMO'), 'total_revenue'].sum()

promo_revenue_percentage = (promo_revenue_sum / total_revenue_sum) * 100

end_time = time.time()

# Display results
print(f"Execution time: {end_time - start_time} seconds")
print("Total Revenue Sum:", total_revenue_sum)
print("Promo Revenue Sum:", promo_revenue_sum)
print("Promo Revenue Percentage:", promo_revenue_percentage)