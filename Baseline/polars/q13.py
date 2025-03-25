import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "2"


def q():
    customer_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/customer.csv", columns=['c_custkey'])
    orders_ds = pl.read_csv("/home/shengya4/data/tpch_3gb/orders.csv", columns=['o_orderkey', 'o_custkey', 'o_comment'])
   
    start = time.monotonic()
    
    filter_str1 = "special"
    filter_str2 = "requests"

    def filter_not_string_exists_before(comment, filter_str1=filter_str1, filter_str2=filter_str2):
        # Find the index of the first occurrence of `filter_str1`
        first_pos_str1 = comment.find(filter_str1)
        # Find the index of the last occurrence of `filter_str2`
        last_pos_str2 = comment.rfind(filter_str2)
        
        # Return True if `filter_str1` does not occur before `filter_str2`
        return not (first_pos_str1 != -1 and last_pos_str2 != -1 and first_pos_str1 < last_pos_str2)

    q_final = (
        customer_ds.join(
            orders_ds.filter(
                pl.col("o_comment").map_elements(filter_not_string_exists_before, return_dtype=pl.Boolean)
            ),
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )
        .group_by(["c_custkey"])
        .agg(pl.col("o_orderkey").drop_nulls().count().alias("c_count"))
        .group_by(["c_count"])
        .agg(pl.count().alias("custdist"))
        .sort(["custdist", "c_count"], descending=[True, True])
    )

    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()