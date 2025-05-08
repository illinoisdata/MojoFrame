import polars as pl
import time
import os
os.environ["POLARS_MAX_THREADS"] = "8"

def q():
    start_load = time.perf_counter()
    part_ds = pl.read_csv("/datadrive/tpch_large/part.csv",columns=['p_partkey', 'p_brand', 'p_type', 'p_size'])
    part_supp_ds = pl.read_csv("/datadrive/tpch_large/partsupp.csv",columns=['ps_partkey', 'ps_suppkey'])
    supp_ds = pl.read_csv("/datadrive/tpch_large/supplier.csv",columns=['s_suppkey', 's_comment'])
    end_load = time.perf_counter()
    print(f"Elapsed Time (Load): {end_load - start_load} seconds")
    
    filter_str1 = "Customer"
    filter_str2 = "Complaints"

    def filter_not_string_exists_before(comment, filter_str1=filter_str1, filter_str2=filter_str2):
        # Find the index of the first occurrence of `filter_str1`
        first_pos_str1 = comment.find(filter_str1)
        # Find the index of the last occurrence of `filter_str2`
        last_pos_str2 = comment.rfind(filter_str2)
        
        # Return True if `filter_str1` does not occur before `filter_str2`
        return not (first_pos_str1 != -1 and last_pos_str2 != -1 and first_pos_str1 < last_pos_str2)

    start = time.monotonic()

    supp_ds_filter = (
        supp_ds.filter(
            pl.col("s_comment").map_elements(filter_not_string_exists_before, return_dtype=pl.Boolean)
        )
    )

    part_ds_filter = part_ds.filter(
        (pl.col("p_brand") != "Brand#45")
        & (~pl.col("p_type").str.starts_with("MEDIUM POLISHED"))
        & (pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
    )

    part_supp_ds_filter = part_supp_ds.join(
        supp_ds_filter, left_on="ps_suppkey", right_on="s_suppkey"
    )

    q_final = (
        part_supp_ds_filter
        .join(part_ds_filter, left_on="ps_partkey", right_on="p_partkey")
        .group_by(["p_size"])
        .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
        .sort(
            ["supplier_cnt", "p_size"],
            descending=[True, True],
        )
    )

   
    end = time.monotonic()
    print(f"Elapsed Time (Monotonic): {end - start} seconds")
    print(q_final)

if __name__ == "__main__":
    q()