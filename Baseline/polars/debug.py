import polars as pl
import pdb

#pl.enable_string_cache()

# 2. Create some example data:
#    - Floats that are really integer IDs (1.0, 2.0, 3.0, etc.)
df1 = pl.DataFrame({
    "o_custkey": [1.0, 2.0, 3.0],
    "val1": [10, 20, 30],
})
df2 = pl.DataFrame({
    "c_custkey": [2.0, 3.0, 4.0],
    "val2": [200, 300, 400],
})

print("Original df1:\n", df1)
print("Original df2:\n", df2)

# 3. Cast float → int → string → categorical
# df1 = df1.with_columns(
#     pl.col("o_custkey")
#     .cast(pl.Int64)
#     .cast(pl.Utf8)
#     .cast(pl.Categorical)
#     .alias("o_custkey_cat")
# )
# df2 = df2.with_columns(
#     pl.col("c_custkey")
#     .cast(pl.Int64)
#     .cast(pl.Utf8)
#     .cast(pl.Categorical)
#     .alias("c_custkey_cat")
# )

# Now both dataframes have "float_id" as a categorical column
# that share the same dictionary encoding thanks to string cache.

print("\nAfter casting to categorical:")
print("df1:\n", df1)
print("df2:\n", df2)

# 4. Perform a join on the categorical column


def f():
    joined = df1.join(df2, left_on="o_custkey", right_on="c_custkey")

pdb.run("f()")

print("\nJoined result:\n", joined)