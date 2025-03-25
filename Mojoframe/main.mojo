from core.Arrays import Float64Array, Float32Array, Int32Array
from core.Calculations import pairwise_sum_f64
from core.Calculations import pairwise_sum_f64, pairwise_sum_f32, pairwise_sum_i32,
aggregation_sum_i32, aggregation_mean_f64,aggregation_all_f64,
filter_string_equal, filter_string_contains, filter_string_endwith, filter_string_startwith, filter_not_string_exists_before, filter_string_equal_mask, filter_string_not_equal_mask, filter_string_not_startwith_mask, filter_f64_IN_mask, filter_string_IN_mask,
reindex_string_column, combine_masks,
evaluate_i32, aggregation_sum_f64, evaluate_f64, evaluate_f32, evaluate_query6, cast_as_float64,
PredicateF64, EQPredF64, NEQPredF64, GTPredF64, GTEPredF64, LEPredF64, LTPredF64, inner_join_i32, left_join_f64, inner_join_f64, inner_join_f64_reindex, element_mult_f64, mergesort, TripleTup, TupleKey
from core.DataFrame import DataFrameF64, DataFrameF32, DataFrameI32, SetElement
from random import random_si64, random_float64
from tensor import Tensor
from python import Python
from time import monotonic, perf_counter
from utils.numerics import neg_inf
from pathlib import Path
from sys.info import simdwidthof
from core.dict import CompactDict
from core.keys_container import KeysBuilder, KeyRef, Keyable
from hashlib import _hasher

# import numojo as nm
# from numojo.prelude import *


# from core.Calculations import pairwise_sum_f64, pairwise_sum_f32, pairwise_sum_i32,
# filter_string_equal, filter_string_contains, filter_string_endwith, filter_string_startwith, filter_not_string_exists_before, filter_string_equal_mask, filter_string_not_equal_mask, filter_string_not_startwith_mask, filter_string_IN_mask,
# reindex_string_column, combine_masks,
# evaluate_i32, evaluate_f64, evaluate_f32, evaluate_query6,
# PredicateF64, EQPredF64, NEQPredF64, GTPredF64, GTEPredF64, LEPredF64, LTPredF64, inner_join_f64, inner_join_f64_reindex, element_mult_f64, mergesort, TripleTup, TupleKey

fn main() raises:
    # test_array_creation()
    #print()
    # test_array_vector_creation()
    # # print()
    #test_pairwise_sum()
    # print()
    # test_df_creation()
    # print()
    # test_df_sum()
    # print()
    # #test_groupby_all()
    # test_groupby_sum()
    # print()
    # test_groupby_mean()
    # print()
    # test_groupby_sum_large()
    # print()
    # test_sum_large()
    # print()
    # test_filter_large_i32()
    # print()
    # test_filter_large_f64()
    # print()
    # test_filter_small_f32()
    # print()
    # test_inner_join_f64()
    # test_left_join_f64()
    # test_inner_join_f64_large()
    # test_merge_sort()
    # test_sort_by_large()
    # test_lexsort()
    # test_lexsort_large()
    # var tup = TripleTup(Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]](2.4, 2, 3))
    
    # print(TupleKey(tup).__hash__())
    #var group_to_idx = Dict[Tuple, Int]()
    # print(hash(tup))
    # test_query_1()
    # test_query_2()
    # test_query_3()
    # test_query_4()
    # test_query_5()
    # test_query_6()
    # test_query_7()
    # test_query_8()
    # test_query_9()
    # test_query_10()
    # test_query_11()
    # test_query_12()
    # test_query_13()
    # test_query_14()
    # test_query_15()
    # test_query_16()
    # test_query_17()
    # test_query_18()
    test_query_19()
    # test_query_20()
    # test_query_21()
    # test_query_22()
    # test_groupby_multiple()
    # var list_sets = List[SetElement]()
    # var ele = SIMD[DType.float64, 1](2.5)
    # var ele2 = SIMD[DType.float64, 1](6.0)
    # list_sets.resize(2000000, SetElement())

    # list_sets[5].distinct_elements.add(ele)

    # print(ele in list_sets[5].distinct_elements)
    # print(ele2 in list_sets[5].distinct_elements)

    # var pd = Python.import_module("pandas")
    # # var np = Python.import_module("numpy")
    # pd.set_option('display.max_columns', None)

    # var file_path_customer = 'customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey']
    # var c_nationkey_arr = df_cust['c_nationkey']
    # var c_acctbal_arr = df_cust['c_acctbal']

    # var c_custkey = Float64Array(450000)
    # var c_nationkey = Float64Array(450000)
    # var c_acctbal = Float64Array(450000)

    # for i in range(450000):
    #     c_custkey[i] = float(c_custkey_arr[i])
    #     c_nationkey[i] = float(c_nationkey_arr[i])
    #     c_acctbal[i] = float(c_acctbal_arr[i])
    

    # var f1 = FloatKey(SIMD[DType.float64, 1](1.0))
    # var f2 = FloatKey(SIMD[DType.float64, 1](2.0))
    # var f3 = FloatKey(SIMD[DType.float64, 1](3.0))
    # var f4 = FloatKey(SIMD[DType.float64, 1](4.0))

    # var f5 = FloatKey(SIMD[DType.float64, 1](5.0))
    # var d = CompactDict[Int]()

    # _= d.put(f1, 4)
    # _= d.put(f2, 3)
    # _= d.put(f3, 2)
    # _= d.put(f4, 1)

    # var got_val = d.get(FloatKey(SIMD[DType.float64, 1](1.0)), 0)
    # var got_val2 = d.get(f5, 0)
    # print(got_val)
    # print(got_val2)

# fn test_array_creation() raises:
#     # Creating a small Float64 array with 2 elements
#     var small_arr_f64 = Float64Array(2)
#     small_arr_f64[0] = 5
#     small_arr_f64[1] = 10

#     var small_arr_f32 = Float32Array(2)
#     small_arr_f32[0] = 5
#     small_arr_f32[1] = 10

#     var small_arr_i32 = Int32Array(2)
#     small_arr_i32[0] = 5
#     small_arr_i32[1] = 10

#     print("Small array with 2 elements")
#     print(small_arr_f64[0], small_arr_f64[1], small_arr_f32[0], small_arr_f32[1],  small_arr_i32[0], small_arr_i32[1])

# # fn test_array_vector_creation() raises:
# #     # Creating a vector of two arrays
# #     var vector = List[Float64Array]()
# #     var arr_ele1 = Float64Array(2)
# #     var arr_ele2 = Float64Array(2)

# #     arr_ele1[0] = 5
# #     arr_ele1[1] = 10
# #     arr_ele2[0] = 0.2
# #     arr_ele2[1] = 0.065

# #     vector.append(arr_ele1)
# #     vector.append(arr_ele2)

# #     print("Print elements from the vector of arrays")
# #     print(vector[0][0], vector[0][1], vector[1][0], vector[1][1])

# fn test_pairwise_sum() raises:
#     # Test that pairwise sum works and its compare its accuracy against naive, Numpy, and high precision sum
#     var np = Python.import_module("numpy")
#     var decimal = Python.import_module("decimal")

#     var max_num = 1
#     var size = 10000000
#     var small_float = 3.1415926585
#     var np_arr = np.random.randint(0, max_num + 1, size)
#     np_arr = np_arr.astype(np.float64)
#     np_arr /= small_float

#     # Use Decimal for high-precision sum
#     # Set high precision
#     decimal.getcontext().prec = 50

#     var decimal_sum = decimal.Decimal('0')
#     var naive_sum = SIMD[DType.float64, 1](0)
#     var mojo_arr = Float64Array(size)
#     var np_sum = np.sum(np_arr)

#     for i in range(size):
#         decimal_sum += decimal.Decimal(np_arr[i])
#         naive_sum += np_arr[i].to_float64()
#         mojo_arr[i] = np_arr[i].to_float64()
    
#     var pairwise_sum = pairwise_sum_f64(mojo_arr, size, 0, size)

#     # Compare
#     print("High precicion sum:", decimal_sum)
#     print("Naive sum:", naive_sum)
#     print("Numpy sum:", np_sum)
#     print("Pairwise sum:", pairwise_sum)

# # fn test_df_creation() raises:
# #     var size = 100000
# #     var col1 = Float64Array(size)
# #     var col2 = Float64Array(size)
    
# #     for i in range(size):
# #         col1[i] = random_float64(SIMD[DType.float64, 1](0), SIMD[DType.float64, 1](100000))
# #         col1[i] = random_float64(SIMD[DType.float64, 1](0), SIMD[DType.float64, 1](100000))

# #     var col_data = List[Float64Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)

# #     var col1_name = "Units Sold"
# #     var col2_name = "UID"
# #     var col_names = List[String]()
# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
    
# #     var df = DataFrameF64(col_data, col_names)

# #     var df_col1_using_index = df[0][0]
# #     var df_col2_using_name = df["UID"][0]

# #     print("DataFrame first column first element:", df_col1_using_index)
# #     print("DataFrame second column first element:", df_col2_using_name)

# # fn test_df_sum() raises:
# #     var col1 = Int32Array(3)
# #     var col2 = Int32Array(3)
# #     col1[0] = 1
# #     col1[1] = 2
# #     col1[2] = 3

# #     col2[0] = 3
# #     col2[1] = 4
# #     col2[2] = 5

# #     var col_data = List[Int32Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Number of Customers"
# #     var col_names = List[String]()
# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
    
# #     var df = DataFrameI32(col_data, col_names)
# #     var df_sums = df.sum(0)

# #     print("DataFrame 1st column sum:", df_sums[0])
# #     print("DataFrame 2nd column sum:", df_sums[1])

# # # fn test_groupby_all() raises:
# # #     var col1 = Float64Array(6)
# # #     var col2 = Float64Array(6)
# # #     var col3 = Float64Array(6)
# # #     col1[0] = 1
# # #     col1[1] = 2
# # #     col1[2] = 3
# # #     col1[3] = 10
# # #     col1[4] = 20
# # #     col1[5] = 30
    
# # #     col2[0] = 1
# # #     col2[1] = 3
# # #     col2[2] = 5
# # #     col2[3] = 5
# # #     col2[4] = 5
# # #     col2[5] = 5

# # #     col3[0] = 100
# # #     col3[1] = 200
# # #     col3[2] = 300
# # #     col3[3] = 1000
# # #     col3[4] = 400
# # #     col3[5] = 250

# # #     var col_data = List[Float64Array]()
# # #     col_data.append(col1)
# # #     col_data.append(col2)
# # #     col_data.append(col3)

# # #     var col1_name = "Units Sold"
# # #     var col2_name = "Customer Group"
# # #     var col3_name = "Number of Customers"
# # #     var col_names = List[String]()

# # #     col_names.append(col1_name)
# # #     col_names.append(col2_name)
# # #     col_names.append(col3_name)

    
# # #     var df = DataFrameF64(col_data, col_names)
# # #     var start_time = monotonic()
    
# # #     df.groupby("Customer Group", "all")

# # #     var end_time = monotonic()

# # #     var execution_time_nanoseconds = end_time - start_time

# # #     var execution_time_seconds = execution_time_nanoseconds / 1000000000
# # #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

# # #     for i in range(df.columns.size):
# # #         print("Current column idx:", i)
# # #         for j in range(df.columns[i].size):
# # #             print("Aggregated sum for the current group: ", df.columns[i][j])

# # fn test_groupby_sum() raises:
# #     var col1 = Int32Array(6)
# #     var col2 = Int32Array(6)
# #     var col3 = Int32Array(6)

# #     # var col1 = Float64Array(6)
# #     # var col2 = Float64Array(6)
# #     # var col3 = Float64Array(6)
# #     col1[0] = 1
# #     col1[1] = 2
# #     col1[2] = 3
# #     col1[3] = 10
# #     col1[4] = 20
# #     col1[5] = 30
    
# #     col2[0] = 1
# #     col2[1] = 3
# #     col2[2] = 5
# #     col2[3] = 5
# #     col2[4] = 5
# #     col2[5] = 5

# #     col3[0] = 100
# #     col3[1] = 200
# #     col3[2] = 300
# #     col3[3] = 1000
# #     col3[4] = 400
# #     col3[5] = 250

# #     var col_data = List[Int32Array]()
# #     # var col_data = List[Float64Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Customer Group"
# #     var col3_name = "Number of Customers"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

    
# #     var df = DataFrameI32(col_data, col_names)
# #     # var df = DataFrameF64(col_data, col_names)
# #     var start_time = monotonic()

# #     df.groupby("Customer Group", "sum")

# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000
# #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

# #     for i in range(df.columns.size):
# #         for j in range(df.columns[i].size):
# #             print("Aggregated sum for the current group: ", df.columns[i][j])

# # fn test_groupby_mean() raises:
# #     var col1 = Float64Array(6)
# #     var col2 = Float64Array(6)
# #     var col3 = Float64Array(6)
# #     col1[0] = 1
# #     col1[1] = 2
# #     col1[2] = 3
# #     col1[3] = 10
# #     col1[4] = 20
# #     col1[5] = 30
    
# #     col2[0] = 1
# #     col2[1] = 3
# #     col2[2] = 5
# #     col2[3] = 5
# #     col2[4] = 5
# #     col2[5] = 5

# #     col3[0] = 100
# #     col3[1] = 200
# #     col3[2] = 300
# #     col3[3] = 1000
# #     col3[4] = 400
# #     col3[5] = 250

# #     var col_data = List[Float64Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Customer Group"
# #     var col3_name = "Number of Customers"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

    
# #     var df = DataFrameF64(col_data, col_names)
# #     var start_time = monotonic()
    
# #     df.groupby("Customer Group", "mean")

# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000
# #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

# #     for i in range(df.columns.size):
# #         for j in range(df.columns[i].size):
# #             print("Aggregation for the current group: ", df.columns[i][j])
   


# # fn test_groupby_sum_large() raises:
# #     var size = 10000000

# #     var col1 = Int32Array(size)
# #     var col2 = Int32Array(size)
# #     var col3 = Int32Array(size)

# #     for i in range(size):
# #         col1[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](99)).cast[DType.int32]()
# #         col2[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](9)).cast[DType.int32]()
# #         col3[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](99)).cast[DType.int32]()

# #     print("Inspect a few groups for groupby:", col2[0], col2[1], col2[2])
# #     print("Inspect a few rows for groupby:", col3[0], col3[1], col3[2])

# #     var col_data = List[Int32Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Customer Group"
# #     var col3_name = "Number of Customers"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

    
# #     var df = DataFrameI32(col_data, col_names)
# #     var start_time = monotonic()
# #     df.groupby("Customer Group", "sum")
# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000
# #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")
    
# #     print(df.columns.size, df.columns[0].size)
# #     for i in range(df.columns.size):
# #         for j in range(df.columns[i].size):
# #             print("Aggregated sum for the current group: ", df.columns[i][j])

# # fn test_sum_large() raises:
# #     var size = 100000000

# #     var col1 = Int32Array(size)
# #     var col2 = Int32Array(size)
# #     var col3 = Int32Array(size)

# #     for i in range(size):
# #         col1[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](9)).cast[DType.int32]()
# #         col2[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](9)).cast[DType.int32]()
# #         col3[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](9)).cast[DType.int32]()

# #     var col_data = List[Int32Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Customer Group"
# #     var col3_name = "Number of Customers"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

    
# #     var df = DataFrameI32(col_data, col_names)
# #     var start_time = monotonic()
# #     var sums = df.sum(0)
# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000
# #     print(sums[0], sums[1], sums[2])
# #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

fn test_filter_large_f64() raises:
    var size = 8

    var col1 = Float64Array(size)
    # var col2 = Float64Array(size)
    # var col3 = Float64Array(size)
    #var data = Tensor[DType.float64](size)
    for i in range(size):
        # col1[i] = random_float64(SIMD[DType.float64, 1](0), SIMD[DType.float64, 1](999))
        # col2[i] = random_float64(SIMD[DType.float64, 1](0), SIMD[DType.float64, 1](999))
        # col3[i] = random_float64(SIMD[DType.float64, 1](0), SIMD[DType.float64, 1](999))
        col1[i] = SIMD[DType.float64, 1](i)
        # data[i] = col1[i]
        #col3[i] = SIMD[DType.float64, 1](3.5) * i
    col1.data.store[width=8](8, SIMD[DType.float64, 1](8, 9, 10, 11, 12, 13, 14, 15))
    # var start_time = monotonic()
    # for i in range(size):
    #     # col3._setitem(i, val=SIMD[DType.float64, 1](1.0))
    #     col1[i] = SIMD[DType.float64, 1](1.0) * 2.568 + 10.09
    # var load_ele = col1.data.load[width=8](0)
    print(col1.data)
    # var ele = SIMD[DType.float64, 8](4, 5, 6, 7, 8, 9, 10, 11)
    #col1.data.store[width=8](0, ele)
    #print(col1.data)
    # var end_time = monotonic()
    # print(col3[Item(10000)])
    # var col_data = List[Float64Array]()
    # col_data.append(col1)
    # col_data.append(col2)
    # col_data.append(col3)

    # var col1_name = "Units Sold"
    # var col2_name = "Customer Group"
    # var col3_name = "Number of Customers"
    # var col_names = List[String]()

    # col_names.append(col1_name)
    # col_names.append(col2_name)
    # col_names.append(col3_name)

    
    # var df = DataFrameF64(col_data, col_names)

    
    # Equivalent to df[df['Customer Group'] > 231]
    # df.select("Units Sold", "Number of Customers", GTPredF64(), LEPredF64(), 127.86546, 897.9871234, "OR")
    # var vec_of_index = List[Int]()
    # for i in range(col1.size):
    #     if col1[i] > 37560:
    #         vec_of_index.append(i)
    # var idxs = nm.greater(col2, SIMD[DType.float64, 1](37560.0))
    

    # print(idxs.size)

    # var execution_time_nanoseconds = end_time - start_time

    # var execution_time_seconds = execution_time_nanoseconds / 1000000000

    # # print("Rows satisfying filter condition:", df.columns[0].size)
    # print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

# # fn test_filter_small_f32() raises:
# #     var col1 = Float32Array(6)
# #     var col2 = Float32Array(6)
# #     var col3 = Float32Array(6)
# #     col1[0] = 1
# #     col1[1] = 2
# #     col1[2] = 3
# #     col1[3] = 10
# #     col1[4] = 20
# #     col1[5] = 30
    
# #     col2[0] = 1
# #     col2[1] = 3
# #     col2[2] = 5
# #     col2[3] = 5
# #     col2[4] = 5
# #     col2[5] = 5

# #     col3[0] = 100
# #     col3[1] = 200
# #     col3[2] = 300
# #     col3[3] = 1000
# #     col3[4] = 400
# #     col3[5] = 250

# #     var col_data = List[Float32Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Customer Group"
# #     var col3_name = "Number of Customers"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

# #     var df = DataFrameF32(col_data, col_names)

# #     var start_time = monotonic()
   
# #     df.select("Units Sold", "Number of Customers", 3, 800)

# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000

# #     print("Rows satisfying filter condition:", df.columns[0].size)
# #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

# # fn test_filter_large_i32() raises:
# #     var size = 10000000

# #     var col1 = Int32Array(size)
# #     var col2 = Int32Array(size)
# #     var col3 = Int32Array(size)

# #     for i in range(size):
# #         col1[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](999)).cast[DType.int32]()
# #         col2[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](999)).cast[DType.int32]()
# #         col3[i] = random_si64(SIMD[DType.int64, 1](0), SIMD[DType.int64, 1](999)).cast[DType.int32]()


# #     var col_data = List[Int32Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "Units Sold"
# #     var col2_name = "Customer Group"
# #     var col3_name = "Number of Customers"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

    
# #     var df = DataFrameI32(col_data, col_names)
# #     var start_time = monotonic()
# #     # Equivalent to df[df['Customer Group'] > 231]
# #     df.select("Units Sold", ">", 500)
# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000

# #     print("Rows satisfying filter condition:", df.columns[0].size)
# #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")


# # # fn test_inner_join_i32() raises:
# # #     var col1 = Int32Array(3)
# # #     var col2 = Int32Array(3)
  
# # #     col1[0] = 1
# # #     col1[1] = 2
# # #     col1[2] = 3

# # #     col2[0] = 20
# # #     col2[1] = 10
# # #     col2[2] = 5
   

# # #     var col_data = List[Int32Array]()
# # #     col_data.append(col1)
# # #     col_data.append(col2)

# # #     var col1_name = "Employee ID"
# # #     var col2_name = "Number of Products Returned"
# # #     var col_names = List[String]()

# # #     col_names.append(col1_name)
# # #     col_names.append(col2_name)

# # #     var df2_col1 = Int32Array(5)
# # #     var df2_col2 = Int32Array(5)
# # #     var df2_col3 = Int32Array(5)
  
# # #     df2_col1[0] = 101
# # #     df2_col1[1] = 102
# # #     df2_col1[2] = 103
# # #     df2_col1[3] = 104
# # #     df2_col1[4] = 105

# # #     df2_col2[0] = 1
# # #     df2_col2[1] = 2
# # #     df2_col2[2] = 1
# # #     df2_col2[3] = 3
# # #     df2_col2[4] = 2

# # #     df2_col3[0] = 250
# # #     df2_col3[1] = 300
# # #     df2_col3[2] = 500
# # #     df2_col3[3] = 150
# # #     df2_col3[4] = 100
   

# # #     var col_data2 = List[Int32Array]()
# # #     col_data2.append(df2_col1)
# # #     col_data2.append(df2_col2)
# # #     col_data2.append(df2_col3)

# # #     var df2_col1_name = "Sale ID"
# # #     var df2_col2_name = "Employee ID"
# # #     var df2_col3_name = "Number of Products Sold"
# # #     var col_names2 = List[String]()

# # #     col_names2.append(df2_col1_name)
# # #     col_names2.append(df2_col2_name)
# # #     col_names2.append(df2_col3_name)


    
# # #     var df = DataFrameI32(col_data, col_names)
# # #     var df2 = DataFrameI32(col_data2, col_names2)

# # #     var start_time = monotonic()

# # #     var joined_df = inner_join_i32(df, df2, "Employee ID")

# # #     var end_time = monotonic()

# # #     var execution_time_nanoseconds = end_time - start_time

# # #     var execution_time_seconds = execution_time_nanoseconds / 1000000000

# # #     for i in range(joined_df.columns.size):
# # #         print("Column: ", i)
# # #         for j in range(joined_df.columns[i].size):
# # #             print("Element: ", joined_df.columns[i][j])

# # #     print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

fn test_left_join_f64() raises:
    var col1 = Float64Array(4)
    var col2 = Float64Array(4)
  
    # Employee IDs in the left DataFrame
    # Note: we added an extra key '4' to demonstrate an unmatched row
    col1[0] = 1
    col1[1] = 2
    col1[2] = 3
    col1[3] = 4   # <-- This will have no match in df2

    # Some other data in the left DataFrame
    col2[0] = 20
    col2[1] = 10
    col2[2] = 5
    col2[3] = 99  # Arbitrary value

    var col_data = List[Float64Array]()
    col_data.append(col1)
    col_data.append(col2)

    var col1_name = "Employee ID"
    var col2_name = "Number of Products Returned"
    var col_names = List[String]()
    col_names.append(col1_name)
    col_names.append(col2_name)

    # Build second DataFrame
    var df2_col1 = Float64Array(5)
    var df2_col2 = Float64Array(5)
    var df2_col3 = Float64Array(5)
  
    # Some sample IDs / data for the right DataFrame
    df2_col1[0] = 101
    df2_col1[1] = 102
    df2_col1[2] = 103
    df2_col1[3] = 104
    df2_col1[4] = 105

    df2_col2[0] = 1
    df2_col2[1] = 2
    df2_col2[2] = 1
    df2_col2[3] = 3
    df2_col2[4] = 2

    df2_col3[0] = 250
    df2_col3[1] = 300
    df2_col3[2] = 500
    df2_col3[3] = 150
    df2_col3[4] = 100

    var col_data2 = List[Float64Array]()
    col_data2.append(df2_col1)
    col_data2.append(df2_col2)
    col_data2.append(df2_col3)

    var df2_col1_name = "SID2"
    var df2_col2_name = "Employee ID"
    var df2_col3_name = "NPS2"
    var col_names2 = List[String]()

    col_names2.append(df2_col1_name)
    col_names2.append(df2_col2_name)
    col_names2.append(df2_col3_name)

    # Construct DataFrames
    var df  = DataFrameF64(col_data,  col_names)
    var df2 = DataFrameF64(col_data2, col_names2)

    # Record start time
    var start_time = monotonic()

    # Perform LEFT JOIN on the "Employee ID" column
    var joined_df = left_join_f64(df, df2, "Employee ID")

    # Record end time
    var end_time = monotonic()

    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds     = execution_time_nanoseconds / 1000000000

    # Print the results
    print("Result of LEFT JOIN on 'Employee ID'")
    for i in range(joined_df.columns.size):
        print("Column Index:", i, "| Name:", joined_df.column_names[i])
        for j in range(joined_df.columns[i].size):
            print("  Row", j, ":", joined_df.columns[i][j])

    print("Execution time:", execution_time_seconds)


fn test_inner_join_f64() raises:
    var col1 = Float64Array(3)
    var col2 = Float64Array(3)
  
    col1[0] = 1
    col1[1] = 2
    col1[2] = 3

    col2[0] = 20
    col2[1] = 10
    col2[2] = 5
   

    var col_data = List[Float64Array]()
    col_data.append(col1)
    col_data.append(col2)

    var col1_name = "Employee ID"
    var col2_name = "Number of Products Returned"
    var col_names = List[String]()

    col_names.append(col1_name)
    col_names.append(col2_name)

    var df2_col1 = Float64Array(5)
    var df2_col2 = Float64Array(5)
    var df2_col3 = Float64Array(5)
  
    df2_col1[0] = 101
    df2_col1[1] = 102
    df2_col1[2] = 103
    df2_col1[3] = 104
    df2_col1[4] = 105

    df2_col2[0] = 1
    df2_col2[1] = 2
    df2_col2[2] = 1
    df2_col2[3] = 3
    df2_col2[4] = 2

    df2_col3[0] = 250
    df2_col3[1] = 300
    df2_col3[2] = 500
    df2_col3[3] = 150
    df2_col3[4] = 100
   

    var col_data2 = List[Float64Array]()
    col_data2.append(df2_col1)
    col_data2.append(df2_col2)
    col_data2.append(df2_col3)

    var df2_col1_name = "SID2"
    var df2_col2_name = "Employee ID"
    var df2_col3_name = "NPS2"
    var col_names2 = List[String]()

    col_names2.append(df2_col1_name)
    col_names2.append(df2_col2_name)
    col_names2.append(df2_col3_name)

    # var df3_col1 = Float64Array(7)
    # var df3_col2 = Float64Array(7)
    # var df3_col3 = Float64Array(7)
  
    # df3_col1[0] = 101
    # df3_col1[1] = 102
    # df3_col1[2] = 103
    # df3_col1[3] = 104
    # df3_col1[4] = 105
    # df3_col1[5] = 106
    # df3_col1[6] = 107

    # df3_col2[0] = 1
    # df3_col2[1] = 2
    # df3_col2[2] = 1
    # df3_col2[3] = 3
    # df3_col2[4] = 2
    # df3_col2[5] = 7
    # df3_col2[6] = 9

    # df3_col3[0] = 250
    # df3_col3[1] = 300
    # df3_col3[2] = 500
    # df3_col3[3] = 150
    # df3_col3[4] = 100
    # df3_col3[5] = 555
    # df3_col3[6] = 666
   

    # var col_data3 = List[Float64Array]()
    # col_data3.append(df3_col1)
    # col_data3.append(df3_col2)
    # col_data3.append(df3_col3)

    # var df3_col1_name = "SID3"
    # var df3_col2_name = "Employee ID"
    # var df3_col3_name = "NPS3"
    # var col_names3 = List[String]()

    # col_names3.append(df3_col1_name)
    # col_names3.append(df3_col2_name)
    # col_names3.append(df3_col3_name)


    
    var df = DataFrameF64(col_data, col_names)
    var df2 = DataFrameF64(col_data2, col_names2)
    #var df3 = DataFrameF64(col_data3, col_names3)

    var start_time = monotonic()

    var joined_df = inner_join_f64(df, df2, "Employee ID")
    #var final = inner_join_f64(joined_df, df3, "Employee ID")

    var end_time = monotonic()

    var execution_time_nanoseconds = end_time - start_time

    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    for i in range(joined_df.columns.size):
        print("Col: ", i)
        for j in range(joined_df.columns[i].size):
            print(joined_df.columns[i][j])

    print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

fn test_inner_join_f64_large() raises:
    var size = 500000000

    var col1 = Float64Array(size)
    var col2 = Float64Array(size)

    for i in range(size):
        col1[i] = i + 1
        col2[i] = random_float64(SIMD[DType.float64, 1](1.0), SIMD[DType.float64, 1](1000.0))


    var col_data = List[Float64Array]()
    col_data.append(col1)
    col_data.append(col2)

    var col1_name = "Employee ID"
    var col2_name = "Department"
    var col_names = List[String]()

    col_names.append(col1_name)
    col_names.append(col2_name)

    var df_employee = DataFrameF64(col_data, col_names)

    var size2 = 10000000

    var df2_col1 = Float64Array(size2)
    var df2_col2 = Float64Array(size2)
    var df2_col3 = Float64Array(size2)

    for i in range(size2):
        df2_col1[i] = i + 1
        df2_col2[i] = random_si64(SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](5000000)).cast[DType.float64]()
        df2_col3[i] = random_float64(SIMD[DType.float64, 1](100.0), SIMD[DType.float64, 1](1000.0))

    var col_data2 = List[Float64Array]()
    col_data2.append(df2_col1)
    col_data2.append(df2_col2)
    col_data2.append(df2_col3)

    var df2_col1_name = "Sale ID"
    var df2_col2_name = "Employee ID"
    var df2_col3_name = "Sale Amount"
    var col_names2 = List[String]()

    col_names2.append(df2_col1_name)
    col_names2.append(df2_col2_name)
    col_names2.append(df2_col3_name)

    var df_sales = DataFrameF64(col_data2, col_names2)

    var start_time = perf_counter()

    var joined_df = inner_join_f64(df_sales, df_employee, "Employee ID")

    var end_time = perf_counter()

    var execution_time_nanoseconds = end_time - start_time

    print("Num rows:", joined_df[0].size)
    print("Num cols:", joined_df.columns.__len__())

    print("First col:", joined_df[0][1000])
    print("Second col:", joined_df[1][1000])
    print("Third col:", joined_df[2][1000])

    var execution_time_seconds = execution_time_nanoseconds / 1000000000
    print("Execution time: ", end_time - start_time)


# # fn test_merge_sort() raises:
# #     var size = 8
# #     var indices = List[Int](capacity=size)
# #     for i in range(size):
# #         indices.append(i)
# #     # empty array
# #     # var vals_to_sort = Float64Array(0)

# #     # single element array
# #     # var vals_to_sort = Float64Array(1)
# #     # vals_to_sort[0] = 12

# #     # already sorted array
# #     # var vals_to_sort = Float64Array(8)
# #     # vals_to_sort[0] = 0
# #     # vals_to_sort[1] = 1
# #     # vals_to_sort[2] = 2
# #     # vals_to_sort[3] = 21
# #     # vals_to_sort[4] = 23
# #     # vals_to_sort[5] = 25
# #     # vals_to_sort[6] = 90
# #     # vals_to_sort[7] = 100

# #     # reverse sorted array
# #     # var vals_to_sort = Float64Array(8)
# #     # vals_to_sort[0] = 100
# #     # vals_to_sort[1] = 90
# #     # vals_to_sort[2] = 25
# #     # vals_to_sort[3] = 25
# #     # vals_to_sort[4] = 24
# #     # vals_to_sort[5] = 24
# #     # vals_to_sort[6] = 2
# #     # vals_to_sort[7] = 1

# #     # negative numbers array
# #     # var vals_to_sort = Float64Array(8)
# #     # vals_to_sort[0] = -1
# #     # vals_to_sort[1] = -3
# #     # vals_to_sort[2] = 0
# #     # vals_to_sort[3] = -25
# #     # vals_to_sort[4] = -9
# #     # vals_to_sort[5] = -1199
# #     # vals_to_sort[6] = -58
# #     # vals_to_sort[7] = -3344556

# #     # mixed numbers array
# #     var vals_to_sort = Float64Array(8)
# #     vals_to_sort[0] = -1
# #     vals_to_sort[1] = -3
# #     vals_to_sort[2] = 0
# #     vals_to_sort[3] = 25
# #     vals_to_sort[4] = -25
# #     vals_to_sort[5] = 25
# #     vals_to_sort[6] = 2
# #     vals_to_sort[7] = -11223

    
# #     _ = mergesort(vals_to_sort, indices)
# #     for i in range(indices.size):
# #         print(indices[i])


# # fn test_sort_by_large() raises:
# #     var size = 10000000
# #     var df1_col1 = Float64Array(size)

# #     for i in range(size):
# #         df1_col1[i] = random_si64(SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](5000000)).cast[DType.float64]()

# #     var indices = List[Int](capacity=size)
# #     for i in range(size):
# #         indices.append(i)

# #     var start_time = monotonic()
# #     var sorted_indexer = mergesort(df1_col1, indices)
# #     var end_time = monotonic()

# #     var execution_time_nanoseconds = end_time - start_time

# #     var execution_time_seconds = execution_time_nanoseconds / 1000000000
# #     print("Time: ", execution_time_seconds)

# # fn test_lexsort() raises:
# #     var col1 = Float64Array(3)
# #     var col2 = Float64Array(3)
# #     var col3 = Float64Array(3)
# #     col1[0] = 1
# #     col1[1] = 2
# #     col1[2] = 3
    
# #     col2[0] = 1
# #     col2[1] = 3
# #     col2[2] = 5

# #     col3[0] = 100
# #     col3[1] = 200
# #     col3[2] = 300

# #     var col_data = List[Float64Array]()
# #     col_data.append(col1)
# #     col_data.append(col2)
# #     col_data.append(col3)

# #     var col1_name = "key1"
# #     var col2_name = "key2"
# #     var col3_name = "key3"
# #     var col_names = List[String]()

# #     col_names.append(col1_name)
# #     col_names.append(col2_name)
# #     col_names.append(col3_name)

# #     var df = DataFrameF64(col_data, col_names)
# #     df.sort_by(List[String]("key1", "key2", "key3"))

# #     for i in range(df.columns.size):
# #         print("Current column:", df.column_names[i])
# #         for j in range(df.columns[i].size):
# #             print(df.columns[i][j])

# fn test_lexsort_large() raises:
#     var size = 10000000

#     var col1 = Float64Array(size)
#     var col2 = Float64Array(size)
#     var col3 = Float64Array(size)

#     for i in range(size):
#         col1[i] = random_si64(SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](100)).cast[DType.float64]()
#         col2[i] = random_si64(SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](50000)).cast[DType.float64]()
#         col3[i] = random_si64(SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2500)).cast[DType.float64]()


#     var col_data = List[Float64Array]()
#     col_data.append(col1)
#     col_data.append(col2)
#     col_data.append(col3)

#     var col1_name = "key1"
#     var col2_name = "key2"
#     var col3_name = "key3"
#     var col_names = List[String]()

#     col_names.append(col1_name)
#     col_names.append(col2_name)
#     col_names.append(col3_name)

    
#     var df = DataFrameF64(col_data, col_names)

#     var start_time = monotonic()
#     df.sort_by(List[String]("key1", "key2", "key3"))
#     var end_time = monotonic()

#     var execution_time_nanoseconds = end_time - start_time
#     var execution_time_seconds = execution_time_nanoseconds / 1000000000
#     print("Df sort by time: ", execution_time_seconds)


fn test_query_1() raises:
    ### TPC-H Query 1

    # SELECT
    #     l_returnflag,
    #     sum(l_quantity) as sum_qty,
    #     sum(l_extendedprice) as sum_base_price,
    #     sum(l_discount) as sum_disc,
    #     avg(l_quantity) as avg_qty,
    #     avg(l_extendedprice) as avg_price,
    #     avg(l_discount) as avg_disc,
    #     count(*) as count_order
    # FROM
    #     lineitem
    # WHERE
    #     l_shipdate <= 904608000.0
    # GROUP BY
    #     l_returnflag;

    # var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    var start_time = monotonic()

    # var file_path = '../../data/tpch_3gb/lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.shape)
    # print(df.head())

    # var l_extendedprice_arr = df['l_extendedprice']
    # var l_discount_arr = df['l_discount']
    # var l_linestatus_arr = df['l_linestatus']
    # var l_returnflag_arr = df['l_returnflag']
    # var l_tax_arr = df['l_tax']
    # var l_quantity_arr = df['l_quantity']
    # var l_shipdate_arr = df['l_shipdate']

    # print("arrays extracted")
    
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_returnflag = Float64Array("../Data/tpch_med/l_returnflag_tensor")
    var l_linestatus = Float64Array("../Data/tpch_med/l_linestatus_tensor")
    var l_tax = Float64Array("../Data/tpch_med/l_tax_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")

    var l_discprice = Float64Array("../Data/tpch_med/l_discprice_tensor")
    var l_charge = Float64Array("../Data/tpch_med/l_charge_tensor")
    
    
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_returnflag = Float64Array(17996609)
    # var l_linestatus = Float64Array(17996609)
    # var l_tax = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)

    # var l_discprice = Float64Array(17996609)
    # var l_charge = Float64Array(17996609)

    # for i in range(17996609):
    #     l_extendedprice[i] = Float64(l_extendedprice_arr[i])
    #     l_discount[i] = Float64(l_discount_arr[i])
    #     l_linestatus[i] = Float64(l_linestatus_arr[i])
    #     l_returnflag[i] = Float64(l_returnflag_arr[i])
    #     l_tax[i] = Float64(l_tax_arr[i])
    #     l_quantity[i] = Float64(l_quantity_arr[i])
    #     l_shipdate[i] = Float64(l_shipdate_arr[i])
    #     # discount price = l_extendedprice * (1 - l_discount)
    #     l_discprice[i] = l_extendedprice[i] * (1 - l_discount[i])
    #     # charge = l_extendedprice * (1 - l_discount) * (1 + l_tax)
    #     l_charge[i] = l_discprice[i] * (1 + l_tax[i])

    print(l_linestatus.size)

    # l_extendedprice.data.tofile(Path("../Data/tpch_med/l_extendedprice_tensor"))
    # l_discount.data.tofile(Path("../Data/tpch_med/l_discount_tensor"))
    # l_linestatus.data.tofile(Path("../Data/tpch_med/l_linestatus_tensor"))
    # l_returnflag.data.tofile(Path("../Data/tpch_med/l_returnflag_tensor"))
    # l_tax.data.tofile(Path("../Data/tpch_med/l_tax_tensor"))
    # l_quantity.data.tofile(Path("../Data/tpch_med/l_quantity_tensor"))
    # l_shipdate.data.tofile(Path("../Data/tpch_med/l_shipdate_tensor"))
    # l_discprice.data.tofile(Path("../Data/tpch_med/l_discprice_tensor"))
    # l_charge.data.tofile(Path("../Data/tpch_med/l_charge_tensor"))

    var col_data = List[Float64Array](l_quantity, l_extendedprice, l_discount, l_returnflag,
                                      l_shipdate, l_linestatus, l_tax, l_discprice, l_charge)

    var col_names = List[String]("l_quantity", "l_extendedprice", "l_discount", "l_returnflag",
                                 "l_shipdate", "l_linestatus", "l_tax", "l_discprice", "l_charge")

    var df_lineitem = DataFrameF64(col_data, col_names)
    
    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000
    print("Data loading time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

    start_time = monotonic()

    var group_by_cols = List[String]("l_returnflag", "l_linestatus")
    var aggregated_col_names = List[String]("l_returnflag", "l_linestatus", "sum_qty", "sum_base_price",
                                            "sum_discount", "sum_date", "sum_tax",
                                            "sum_discount_price", "sum_charge", "avg_qty", "avg_base_price",
                                            "avg_discount", "avg_date", "avg_tax", "avg_discount_price",
                                            "avg_charge", "group_count")

    df_lineitem.select("l_shipdate", "l_shipdate", LEPredF64(), LEPredF64(), 904608000.0, 904608000.0, "")

    end_time = monotonic()
    print("filter time: ", (end_time - start_time) / 1000000000)
    # print("filtered")
    # print(df_lineitem.columns[0].size)
    df_lineitem.groupby_multicol(group_by_cols, "all", aggregated_col_names)

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000
    print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")

    print("num groups: ", df_lineitem[0].size)
    for i in range(df_lineitem.columns.size):
        print(df_lineitem.column_names[i])
        print(df_lineitem[df_lineitem.column_names[i]][0])


    

fn test_query_3() raises:
    ### TPC-H Query 3
    # SELECT
    #     sum(l_extendedprice * l_discount) as revenue,
    #     o_shippriority
    # FROM
    #     customer,
    #     orders,
    #     lineitem
    # WHERE
    #     c_mktsegment = 'BUILDING'
    #     AND c_custkey = o_custkey
    #     AND l_orderkey = o_orderkey
    #     AND o_orderdate < date '1995-03-15'
    #     AND l_shipdate > date '1995-03-15'
    # GROUP BY
    #     o_shippriority
    # LIMIT 20;
    
    # var pd = Python.import_module("pandas")

    # pd.set_option('display.max_columns', None)

    var start_time = monotonic()

    # var file_path = '../../../app/data/tpch_3gb/lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    #var l_orderkey_arr = df['l_orderkey']
    #var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()
    # var l_quantity_arr = df['l_quantity'].to_numpy()
    # var l_returnflag_arr = df['l_returnflag'].to_numpy()


    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")
    var l_returnflag = Float64Array("../Data/tpch_med/l_returnflag_tensor")
    var l_revenue = Float64Array("../Data/tpch_med/l_discprice_tensor")

    # for i in range(17996609):
    #     #l_orderkey[i] = Float64(l_orderkey_arr[i])
    #     l_revenue[i] = l_extendedprice[i] * (1.0 - l_discount[i])
    
    # l_orderkey.data.tofile(Path("l_orderkey_tensor"))
    print(l_orderkey.size)

    var col_data = List[Float64Array](l_orderkey, l_quantity, l_extendedprice,
                                      l_discount, l_returnflag, l_shipdate, l_revenue)


    var col_names = List[String]("orderkey", "l_quantity", "l_extendedprice", "l_discount",
                                 "l_returnflag", "l_shipdate", "l_revenue")

    var df_lineitem = DataFrameF64(col_data, col_names)

    # var file_path_customer = '../../../app/data/tpch_3gb/customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()
    # var c_mktsegment_arr = df_cust['c_mktsegment']

    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")
    var c_mktsegment = Float64Array("../Data/tpch_med/c_mktsegment_tensor")

    # for i in range(450000):
    #     # c_custkey[i] = float(c_custkey_arr[i])
    #     c_mktsegment[i] = Float64(c_mktsegment_arr[i])
    
    # c_mktsegment.data.tofile(Path("c_mktsegment_tensor"))

    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey, c_mktsegment)
    
    var cust_col_names = List[String]("custkey", "c_mktsegment")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    # var file_path_orders = '../../../app/data/tpch_3gb/orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_custkey_arr = df_ord['o_custkey'].to_numpy()
    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_orderdate_arr = df_ord['o_orderdate']
    # var o_shippriority_arr = df_ord['o_shippriority']

    # var o_custkey = Float64Array(4500000)
    # var o_orderkey = Float64Array(4500000)
    # var o_orderdate = Float64Array(4500000)
    # var o_shippriority = Float64Array(4500000)

    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")
    var o_shippriority = Float64Array("../Data/tpch_med/o_shippriority_tensor")

    # for i in range(4500000):
    #     # o_custkey[i] = float(o_custkey_arr[i])
    #     # o_orderkey[i] = float(o_orderkey_arr[i])
    #     o_orderdate[i] = float(o_orderdate_arr[i])
    #     o_shippriority[i] = float(o_shippriority_arr[i])
    
    # o_orderdate.data.tofile(Path("o_orderdate_tensor"))
    # o_shippriority.data.tofile(Path("o_shippriority_tensor"))

    print(o_orderdate.size)

    var orders_col_data = List[Float64Array](o_custkey, o_orderkey, o_orderdate, o_shippriority)

    
    var orders_col_names = List[String]("custkey", "orderkey", "o_orderdate", "o_shippriority")


    var df_orders = DataFrameF64(orders_col_data, orders_col_names)

    var end_time = monotonic()

    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000
    #print(joined_df_final[0].size)
    print("Data loading time:", execution_time_seconds)


    start_time = monotonic()

    df_customer.select("c_mktsegment", "c_mktsegment", EQPredF64(), EQPredF64(), 1.0, 1.0, "")
    df_orders.select("o_orderdate", "o_orderdate", LTPredF64(), LTPredF64(), 794880000.0, 794880000.0, "")
    df_lineitem.select("l_shipdate", "l_shipdate", GTPredF64(), GTPredF64(), 794880000.0, 794880000.0, "")

    var joined_co_df = inner_join_f64(df_customer, df_orders, "custkey")
    var joined_df_final = inner_join_f64(df_lineitem, joined_co_df, "orderkey")

    var aggregated_col_names = List[String]("orderkey", "o_orderdate", "o_shippriority", "quantity", 
                                            "price", "discount", "return_flag", "shipdate", "revenue",
                                            "custkey", "mktsegment")

       
    var group_by_cols = List[String]("orderkey", "o_orderdate", "o_shippriority")
    joined_df_final.groupby_multicol(group_by_cols, "sum", aggregated_col_names)
    

    joined_df_final.sort_by(List[String]("revenue", "o_orderdate"))

    
    end_time = monotonic()

    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000
    #print(joined_df_final[0].size)
    print("exec time:", execution_time_seconds)

    for i in range(joined_df_final.columns.size):
        print("Current col :", joined_df_final.column_names[i])
        print("Aggregation for the current group: ", joined_df_final.columns[i][0])
    
    print()

    for i in range(joined_df_final.columns.size):
        print("Current col :", joined_df_final.column_names[i])
        print("Aggregation for the current group: ", joined_df_final.columns[i][joined_df_final.columns[0].size - 1])


fn test_query_6() raises:
    print("Current system time:", monotonic())
    ### TPC-H Query 6
    
    # SELECT
    #     sum(l_extendedprice * l_discount) as revenue
    # FROM
    #     lineitem
    # WHERE
    #     l_shipdate >= date '1994-01-01'
    #     AND l_shipdate < date '1994-01-01' + interval '1' year
    #     AND l_discount between 0.06 - 0.01 AND 0.06 + 0.01
    #     AND l_quantity < 24;
    
    ### Since String is implemented as List[si8] in Mojo, we will use the UNIX representation for core testing
    ### Translated Query
    
    # SELECT
    #     sum(l_extendedprice * l_discount) as revenue
    # FROM
    #     lineitem
    # WHERE
    #     l_shipdate >= 757382400.0
    #     AND l_shipdate < 788918400.0
    #     AND l_discount between 0.06 - 0.01 AND 0.06 + 0.01
    #     AND l_quantity < 24;
    

    # var pd = Python.import_module("pandas")
    # #var np = Python.import_module("numpy")

    # pd.set_option('display.max_columns', None)
    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var filtered_df_new = df[
    #     (df['l_shipdate'] >= 757382400.0) &
    #     (df['l_shipdate'] < 788918400.0) &
    #     (df['l_discount'] >= (0.06 - 0.01)) &
    #     (df['l_discount'] <= (0.06 + 0.01)) &
    #     (lineitem_dfdf_new['l_quantity'] < max_quantity)
    # ]

    # var selected_indices = filtered_df_new.index.tolist()
    # var indices = List[Int]()
    # for num in selected_indices:
    #     indices.append(int(num))
    #print(selected_indices[0:10])
    #print("select len:", indices.__len__())
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()
    # var l_quantity_arr = df['l_quantity'].to_numpy()


    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)

    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")

    # for i in range(17996609):
    #     l_extendedprice[i] = float(l_extendedprice_arr[i])
    #     l_discount[i] = float(l_discount_arr[i])
    #     l_shipdate[i] = float(l_shipdate_arr[i])
    #     l_quantity[i] = float(l_quantity_arr[i])
    
    print(l_quantity.size)

    #print("disc idx 5:", l_discount[5])

    #print(l_discount[3333])
    var col_data = List[Float64Array](l_extendedprice, l_discount, l_shipdate, l_quantity)

    var col_names = List[String]("l_extendedprice", "l_discount", "l_shipdate", "l_quantity")

    var df_lineitem = DataFrameF64(col_data, col_names)


    var start_time = monotonic()
    # df_lineitem.select("l_shipdate", "l_discount", "l_quantity",
    #                     GTEPredF64(), LTPredF64(), GTEPredF64(), LEPredF64(), LTPredF64(),
    #                     757382400.0, 788918400.0, (0.06 - 0.01), (0.06 + 0.01), 24, "AND")
    
    df_lineitem.select("l_shipdate", "l_shipdate", GTEPredF64(), LTPredF64(), 757382400.0, 788918400.0, "AND")
    df_lineitem.select("l_discount", "l_discount", GTEPredF64(), LEPredF64(), (0.06 - 0.01), (0.06 + 0.01), "AND")
    df_lineitem.select("l_quantity", "l_quantity", LTPredF64(), LTPredF64(), 24, 24, "")

    # print(df_lineitem["l_discount"].size)
    var price = df_lineitem["l_extendedprice"]
    var discount = df_lineitem["l_discount"]

    var price_discount = element_mult_f64(price, discount)
    var revenue = pairwise_sum_f64(price_discount, price_discount.size, 0, price_discount.size)
    
    var end_time = monotonic()
    print(df_lineitem["l_discount"].size)
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000
    
    print("Revenue:", revenue)
    print("exec time:", execution_time_seconds)

# fn test_query_3() raises:
#     var pd = Python.import_module("pandas")
#     # var np = Python.import_module("numpy")

#     pd.set_option('display.max_columns', None)
#     var file_path = 'lineitem-med.csv'
#     var df = pd.read_csv(file_path)
#     print(df.head())
#     print(df.shape)

#     var l_orderkey_arr = df['l_orderkey'].to_numpy()
#     var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
#     var l_discount_arr = df['l_discount'].to_numpy()
#     var l_shipdate_arr = df['l_shipdate'].to_numpy()
#     var l_quantity_arr = df['l_quantity'].to_numpy()
#     var l_returnflag_arr = df['l_returnflag'].to_numpy()


#     var l_orderkey = Float64Array(17996609)
#     var l_extendedprice = Float64Array(17996609)
#     var l_discount = Float64Array(17996609)
#     var l_shipdate = Float64Array(17996609)
#     var l_quantity = Float64Array(17996609)
#     var l_returnflag = Float64Array(17996609)
#     var l_discprice = Float64Array(17996609)

#     for i in range(17996609):
#         l_orderkey[i] = l_orderkey_arr[i].to_float64()
#         l_quantity[i] = l_quantity_arr[i].to_float64()
#         l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
#         l_discount[i] = l_discount_arr[i].to_float64()
#         l_returnflag[i] = l_returnflag_arr[i].to_float64()
#         l_shipdate[i] = l_shipdate_arr[i].to_float64()
#         l_discprice[i] = l_extendedprice[i] * (1 - l_discount[i])
        
#     print(l_orderkey.size)

#     var col_data = List[Float64Array]()
#     col_data.append(l_orderkey)
#     col_data.append(l_quantity)
#     col_data.append(l_extendedprice)
#     col_data.append(l_discount)
#     col_data.append(l_returnflag)
#     col_data.append(l_shipdate)
#     col_data.append(l_discprice)

#     var col1_name = "orderkey"
#     var col2_name = "l_quantity"
#     var col3_name = "l_extendedprice"
#     var col4_name = "l_discount"
#     var col5_name = "l_returnflag"
#     var col6_name = "l_shipdate"
#     var col7_name = "l_discprice"

#     var col_names = List[String]()
#     col_names.append(col1_name)
#     col_names.append(col2_name)
#     col_names.append(col3_name)
#     col_names.append(col4_name)
#     col_names.append(col5_name)
#     col_names.append(col6_name)
#     col_names.append(col7_name)

#     var df_lineitem = DataFrameF64(col_data, col_names)

#     var file_path_customer = 'customer.csv'
#     var df_cust = pd.read_csv(file_path_customer)
#     print(df_cust.head())
#     print(df_cust.shape)

#     var c_custkey_arr = df_cust['c_custkey'].to_numpy()
#     var c_mktsegment_arr = df_cust['c_mktsegment'].to_numpy()

#     var c_custkey = Float64Array(450000)
#     var c_mktsegment = Float64Array(450000)

#     for i in range(450000):
#         c_custkey[i] = c_custkey_arr[i].to_float64()
#         c_mktsegment[i] = c_mktsegment_arr[i].to_float64()
    
#     print(c_custkey.size)

#     var cust_col_data = List[Float64Array]()
#     cust_col_data.append(c_custkey)
#     cust_col_data.append(c_mktsegment)

#     var cust_col1_name = "custkey"
#     var cust_col2_name = "c_mktsegment"
    
#     var cust_col_names = List[String]()
#     cust_col_names.append(cust_col1_name)
#     cust_col_names.append(cust_col2_name)

#     var df_customer = DataFrameF64(cust_col_data, cust_col_names)


#     var file_path_orders = 'orders.csv'
#     var df_ord = pd.read_csv(file_path_orders)
#     print(df_ord.head())
#     print(df_ord.shape)

#     var o_custkey_arr = df_ord['o_custkey'].to_numpy()
#     var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
#     var o_orderdate_arr = df_ord['o_orderdate'].to_numpy()
#     var o_shippriority_arr = df_ord['o_shippriority'].to_numpy()

#     var o_custkey = Float64Array(4500000)
#     var o_orderkey = Float64Array(4500000)
#     var o_orderdate = Float64Array(4500000)
#     var o_shippriority = Float64Array(4500000)

#     for i in range(4500000):
#         o_custkey[i] = o_custkey_arr[i].to_float64()
#         o_orderkey[i] = o_orderkey_arr[i].to_float64()
#         o_orderdate[i] = o_orderdate_arr[i].to_float64()
#         o_shippriority[i] = o_shippriority_arr[i].to_float64()
    
#     print(o_custkey.size)

#     var orders_col_data = List[Float64Array]()
#     orders_col_data.append(o_custkey)
#     orders_col_data.append(o_orderkey)
#     orders_col_data.append(o_orderdate)
#     orders_col_data.append(o_shippriority)

#     var orders_col1_name = "custkey"
#     var orders_col2_name = "orderkey"
#     var orders_col3_name = "o_orderdate"
#     var orders_col4_name = "o_shippriority"
    
#     var orders_col_names = List[String]()
#     orders_col_names.append(orders_col1_name)
#     orders_col_names.append(orders_col2_name)
#     orders_col_names.append(orders_col3_name)
#     orders_col_names.append(orders_col4_name)


#     var df_orders = DataFrameF64(orders_col_data, orders_col_names)

#     var start_time = monotonic()
#     df_customer.select("c_mktsegment", "c_mktsegment", EQPredF64(), EQPredF64(), 1.0, 1.0, "")
#     df_lineitem.select("l_shipdate", "l_shipdate", GTPredF64(), GTPredF64(), 794880000.0, 794880000.0, "")
#     df_orders.select("o_orderdate", "o_orderdate", LTPredF64(), LTPredF64(), 794880000.0, 794880000.0, "")
#     var joined_df = inner_join_f64(df_customer, df_orders, "custkey")
#     #joined_df.select("orderkey", "orderkey", EQPredF64(), EQPredF64(), 359.0, 359.0, "", indices)
#     var joined_df_final = inner_join_f64(df_lineitem, joined_df, "orderkey")
    

#     var group_by_cols = List[String]("orderkey", "o_orderdate", "o_shippriority")
#     var aggregated_col_names = List[String]("orderkey", "o_orderdate", "o_shippriority",
#                                             "quantity", "price", "discount", 
#                                             "return_flag", "shipdate", "revenue", "custkey", "mktsegment")
    

#     var end_time = monotonic()
#     var execution_time_nanoseconds = end_time - start_time
#     var execution_time_seconds = execution_time_nanoseconds / 1000000000

#     print("perform groupby on multiple cols, not aggregating yet!")
#     print("num records after join and filter: ", joined_df_final.columns[0].size)

#     print("exec time after join and filter: ", execution_time_seconds)

#     joined_df_final.groupby_multicol(group_by_cols, "sum", aggregated_col_names)
#     joined_df_final.sort_by(List[String]("revenue", "o_orderdate"))

#     end_time = monotonic()
#     execution_time_nanoseconds = end_time - start_time
#     execution_time_seconds = execution_time_nanoseconds / 1000000000

#     # print("flat keys:", flat_keys.size)
#     #print("compound keys:", groups_vec.size)
#     # for val in group_to_count.values():
#     #     print(val[])
#     #     break
        
#     print("exec time: ", execution_time_seconds)
#     # print("group 34240 summed vals:")
#     # print(joined_df_final.columns.size)
#     for i in range(joined_df_final.columns.size):
#         print(joined_df_final.column_names[i])
#         print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 3])

#     print()

#     for i in range(joined_df_final.columns.size):
#         print(joined_df_final.column_names[i])
#         print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 2])
    

#     print()

#     for i in range(joined_df_final.columns.size):
#         print(joined_df_final.column_names[i])
#         print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 1])


fn test_query_4() raises:
    # var pd = Python.import_module("pandas")
    # # var np = Python.import_module("numpy")

    # pd.set_option('display.max_columns', None)
    # var file_path = '../../../app/data/tpch_large/lineitem.csv'
    # var df = pd.read_csv(file_path)
    #print(df.head())
    #print(df.shape)

    # var l_orderkey_arr = df['l_orderkey']
    # var l_extendedprice_arr = df['l_extendedprice']
    # var l_shipdate_arr = df['l_shipdate']
    # var l_commitdate_arr = df['l_commitdate']
    # var l_receiptdate_arr = df['l_receiptdate']
    # var l_quantity_arr = df['l_quantity']
    

    # var l_orderkey = Float64Array(59986052)
    # var l_extendedprice = Float64Array(59986052)
    # var l_shipdate = Float64Array(59986052)
    # var l_commitdate = Float64Array(59986052)
    # var l_receiptdate = Float64Array(59986052)
    # var l_quantity = Float64Array(59986052)

    var start_time = monotonic()

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")
    var l_commitdate = Float64Array("../Data/tpch_med/l_commitdate_tensor")
    var l_receiptdate = Float64Array("../Data/tpch_med/l_receiptdate_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")

    # for i in range(59986052):
    #     l_orderkey[i] = l_orderkey_arr[i].to_float64()
    #     l_quantity[i] = l_quantity_arr[i].to_float64()
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     l_shipdate[i] = l_shipdate_arr[i].to_float64()
    #     l_commitdate[i] = l_commitdate_arr[i].to_float64()
    #     l_receiptdate[i] = l_receiptdate_arr[i].to_float64()
    
    # l_orderkey.data.tofile(Path("../Data/tpch_large/l_orderkey_tensor"))
    # l_quantity.data.tofile(Path("../Data/tpch_large/l_quantity_tensor"))
    # l_extendedprice.data.tofile(Path("../Data/tpch_large/l_extendedprice_tensor"))
    # l_shipdate.data.tofile(Path("../Data/tpch_large/l_shipdate_tensor"))
    # l_commitdate.data.tofile(Path("../Data/tpch_large/l_commitdate_tensor"))
    # l_receiptdate.data.tofile(Path("../Data/tpch_large/l_receiptdate_tensor"))

    print(l_receiptdate.size)

    var col_data = List[Float64Array](l_orderkey, l_quantity, l_extendedprice,
                                      l_shipdate, l_commitdate, l_receiptdate)


    var col_names = List[String]("orderkey", "l_quantity", "l_extendedprice", "l_shipdate",
                                 "l_commitdate", "l_receiptdate")

    var df_lineitem = DataFrameF64(col_data, col_names)


    # var file_path_orders = '../../../app/data/tpch_large/orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # # print(df_ord.head())
    # print(df_ord.shape)

    # var o_custkey_arr = df_ord['o_custkey']
    # var o_orderkey_arr = df_ord['o_orderkey']
    # var o_orderdate_arr = df_ord['o_orderdate']
    # var o_shippriority_arr = df_ord['o_shippriority']
    # var o_orderpriority_arr = df_ord['o_orderpriority']

    # var o_custkey = Float64Array(15000000)
    # var o_orderkey = Float64Array(15000000)
    # var o_orderdate = Float64Array(15000000)
    # var o_shippriority = Float64Array(15000000)
    # var o_orderpriority = Float64Array(15000000)

    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")
    var o_shippriority = Float64Array("../Data/tpch_med/o_shippriority_tensor")
    var o_orderpriority = Float64Array("../Data/tpch_med/o_orderpriority_tensor")

    # for i in range(15000000):
    #     o_custkey[i] = o_custkey_arr[i].to_float64()
    #     o_orderkey[i] = o_orderkey_arr[i].to_float64()
    #     o_orderdate[i] = o_orderdate_arr[i].to_float64()
    #     o_shippriority[i] = o_shippriority_arr[i].to_float64()
    #     o_orderpriority[i] = o_orderpriority_arr[i].to_float64()
    
    # o_custkey.data.tofile(Path("../Data/tpch_large/o_custkey_tensor"))
    # o_orderkey.data.tofile(Path("../Data/tpch_large/o_orderkey_tensor"))
    # o_orderdate.data.tofile(Path("../Data/tpch_large/o_orderdate_tensor"))
    # o_shippriority.data.tofile(Path("../Data/tpch_large/o_shippriority_tensor"))
    # o_orderpriority.data.tofile(Path("../Data/tpch_large/o_orderpriority_tensor"))

    print(o_orderpriority.size)

    var orders_col_data = List[Float64Array](o_custkey, o_orderkey, o_orderdate, o_shippriority, o_orderpriority)
    
    var orders_col_names = List[String]("custkey", "orderkey", "o_orderdate", "o_shippriority", "o_orderpriority")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("Data loading time:", execution_time_seconds)

    # start_time = monotonic()
    # # exists (
	# # 	select
	# # 		*
	# # 	from
	# # 		lineitem
	# # 	where
	# # 		l_orderkey = o_orderkey
	# # 		and l_commitdate < l_receiptdate
	# # )

    start_time = monotonic()
    var exec_start = perf_counter()
    
    df_lineitem.select("l_commitdate", "l_receiptdate", LTPredF64(), LTPredF64(), 0, 0, "COL")
    df_orders.select("o_orderdate", "o_orderdate", GTEPredF64(), LTPredF64(), 741484800.0, 749433600.0, "AND")

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("filter time:", execution_time_seconds)

    start_time = monotonic()

    var joined_df = inner_join_f64(df_orders, df_lineitem, "orderkey")
    
    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("join time:", execution_time_seconds)

    # print("Performed join and filter for EXISTS")
    # print("num records after join and filter: ", joined_df.columns[0].size)

    # print("exec time after join and filter: ", execution_time_seconds)


    start_time = monotonic()

    var aggregated_col_names = List[String]("order_priority", "order_count")
    #joined_df.select("o_orderdate", "o_orderdate", GTEPredF64(), LTPredF64(), 741484800.0, 749433600.0, "AND")
    # print("complete second filter")
    joined_df.groupby("o_orderpriority", "count", aggregated_col_names)

    end_time = monotonic()

    print("groupby time: ", (end_time - start_time) / 1000000000)

    start_time = monotonic()

    joined_df.sort_by(List[String]("order_priority"))
    # print("complete groupby count")
    # joined_df_final.groupby_multicol(group_by_cols, "sum", aggregated_col_names)
    # joined_df_final.sort_by(List[String]("revenue", "o_orderdate"))

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000
    print("sort time: ", execution_time_seconds)

    # print("flat keys:", flat_keys.size)
    #print("compound keys:", groups_vec.size)
    # for val in group_to_count.values():
    #     print(val[])
    #     break
    
    var exec_end = perf_counter()
    print("exec time: ", exec_end - exec_start)
    # print("exec time: ", execution_time_seconds)
    # print("group 34240 summed vals:")
    # print(joined_df.columns.size)
    for i in range(joined_df.columns.size):
        print(joined_df.column_names[i])
        for j in range(joined_df.columns[0].size):
            print(joined_df[joined_df.column_names[i]][j])


fn test_query_5() raises:
    # var pd = Python.import_module("pandas")

    # pd.set_option('display.max_columns', None)
    # var file_path = '../../data/tpch_3gb/lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey']
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_suppkey_arr = df['l_suppkey']


    # var l_orderkey = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_discprice = Float64Array(17996609)
    # var l_suppkey = Float64Array(17996609)

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_discprice = Float64Array("../Data/tpch_med/l_discprice_tensor")
    var l_suppkey = Float64Array("../Data//tpch_med/l_suppkey_tensor")

    # for i in range(17996609):
    #     l_orderkey[i] = Float64(l_orderkey_arr[i])
    #     # l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     # l_discount[i] = l_discount_arr[i].to_float64()
    #     # l_discprice[i] = l_extendedprice[i] * (1 - l_discount[i])
    #     l_suppkey[i] = Float64(l_suppkey_arr[i])

    # l_orderkey.data.tofile(Path("../Data/tpch_med/l_orderkey_tensor"))
    # l_suppkey.data.tofile(Path("../Data/tpch_med/l_suppkey_tensor"))
        
    print(l_orderkey.size)

    var col_data = List[Float64Array](l_orderkey, l_extendedprice, l_discount, l_discprice, l_suppkey)

    var col_names = List[String]("orderkey", "l_extendedprice", "l_discount", "l_discprice", "suppkey")

    var df_lineitem = DataFrameF64(col_data, col_names)

    # var file_path_customer = '/app/data/tpch_3gb/customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()
    # var c_nationkey_arr = df_cust['c_nationkey'].to_numpy()

    # var c_custkey = Float64Array(450000)
    # var c_nationkey = Float64Array(450000)

    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")
    var c_nationkey = Float64Array("../Data/tpch_med/c_nationkey_tensor")

    # for i in range(450000):
    #     c_custkey[i] = c_custkey_arr[i].to_float64()
    #     c_nationkey[i] = c_nationkey_arr[i].to_float64()
    
    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey, c_nationkey)
    
    var cust_col_names = List[String]("custkey", "c_nationkey")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    # var file_path_orders = '../../data/tpch_3gb/orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_custkey_arr = df_ord['o_custkey'].to_numpy()
    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_orderdate_arr = df_ord['o_orderdate']

    # var o_custkey = Float64Array(4500000)
    # var o_orderkey = Float64Array(4500000)
    # var o_orderdate = Float64Array(4500000)

    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")

    # for i in range(4500000):
    #     # o_custkey[i] = o_custkey_arr[i].to_float64()
    #     # o_orderkey[i] = o_orderkey_arr[i].to_float64()
    #     o_orderdate[i] = Float64(o_orderdate_arr[i])
    
    # o_orderdate.data.tofile(Path("../Data/tpch_med/o_orderdate_tensor"))
    print(o_orderdate.size)
    

    var orders_col_data = List[Float64Array](o_custkey, o_orderkey, o_orderdate)
    
    var orders_col_names = List[String]("custkey", "orderkey", "o_orderdate")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    # var file_path_supp = '../../data/tpch_3gb/supplier.csv'
    # var df_supp = pd.read_csv(file_path_supp)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey']
    # var s_nationkey_arr = df_supp['s_nationkey']

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")

    # var s_suppkey = Float64Array(30000)
    # var s_nationkey = Float64Array(30000)

    # for i in range(30000):
    #     s_suppkey[i] = Float64(s_suppkey_arr[i])
    #     s_nationkey[i] = Float64(s_nationkey_arr[i])
    
    # s_suppkey.data.tofile(Path("../Data/tpch_med/s_suppkey_tensor"))
    # s_nationkey.data.tofile(Path("../Data/tpch_med/s_nationkey_tensor"))
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey)
    
    var supp_col_names = List[String]("suppkey", "nationkey")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)

    # var file_path_nation = '../../data/tpch_3gb/nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey']
    # var n_regionkey_arr = df_nat['n_regionkey']
    # var n_name_arr = df_nat['n_name']

    # var n_nationkey = Float64Array(25)
    # var n_regionkey = Float64Array(25)
    # var n_name = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_regionkey = Float64Array("../Data/tpch_med/n_regionkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # for i in range(25):
    #     n_nationkey[i] = Float64(n_nationkey_arr[i])
    #     n_regionkey[i] = Float64(n_regionkey_arr[i])
    #     n_name[i] = Float64(n_name_arr[i])
    
    # n_nationkey.data.tofile(Path("../Data/tpch_med/n_nationkey_tensor"))
    # n_regionkey.data.tofile(Path("../Data/tpch_med/n_regionkey_tensor"))
    # n_name.data.tofile(Path("../Data/tpch_med/n_name_tensor"))
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_regionkey, n_name)
    
    var nation_col_names = List[String]("nationkey", "regionkey", "n_name")

    var df_nation = DataFrameF64(nation_col_data, nation_col_names)


    # var file_path_region = '../..//data/tpch_3gb/region.csv'
    # var df_reg = pd.read_csv(file_path_region)
    # print(df_reg.head())
    # print(df_reg.shape)

    # var r_regionkey_arr = df_reg['r_regionkey']
    # var r_name_arr = df_reg['r_name']

    # var r_regionkey = Float64Array(5)
    # var r_name = Float64Array(5)

    var r_regionkey = Float64Array("../Data/tpch_med/r_regionkey_tensor")
    var r_name = Float64Array("../Data/tpch_med/r_name_tensor")

    # for i in range(5):
    #     r_regionkey[i] = Float64(r_regionkey_arr[i])
    #     r_name[i] = Float64(r_name_arr[i])
    
    # r_regionkey.data.tofile(Path("../Data/tpch_med/r_regionkey_tensor"))
    # r_name.data.tofile(Path("../Data/tpch_med/r_name_tensor"))

    print(r_regionkey.size)

    var region_col_data = List[Float64Array](r_regionkey, r_name)
    
    var region_col_names = List[String]("regionkey", "r_name")

    var df_region = DataFrameF64(region_col_data, region_col_names)


    var start_time = monotonic()
    # filter first then join
    # and r_name = 'ASIA'
	# and o_orderdate >= date '1994-01-01'
	# and o_orderdate < date '1994-01-01' + interval '1' year


    # 43715.0 is the float representation of ASIA
    df_region.select("r_name", "r_name", EQPredF64(), EQPredF64(), 43715.0, 43715.0, "")
    df_orders.select("o_orderdate", "o_orderdate", GTEPredF64(), LTPredF64(), 757382400.0, 788918400.0, "AND")

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("filter time:", execution_time_seconds)

    # c_custkey = o_custkey
	# and l_orderkey = o_orderkey
	# and l_suppkey = s_suppkey
	# and c_nationkey = s_nationkey
	# and s_nationkey = n_nationkey
	# and n_regionkey = r_regionkey
    var joined_co_df = inner_join_f64(df_customer, df_orders, "custkey")
    var joined_l_df = inner_join_f64(df_lineitem, joined_co_df, "orderkey")
    var joined_supp_df = inner_join_f64(joined_l_df, df_supplier, "suppkey")
    joined_supp_df.select("c_nationkey", "nationkey", EQPredF64(), EQPredF64(), 0, 0, "COL")

    var joined_nation_df = inner_join_f64(joined_supp_df, df_nation, "nationkey")
    var joined_df_final = inner_join_f64(joined_nation_df, df_region, "regionkey")
    
    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("join time:", execution_time_seconds)

    var aggregated_col_names = List[String]("n_name", "orderkey", "extendedprice",
                                            "discount", "revenue", "suppkey", 
                                            "custkey", "c_nationkey", "o_orderdate",
                                            "nationkey", "regionkey", "r_name")
    joined_df_final.groupby("n_name", "sum", aggregated_col_names)
    

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("aggregation time: ", execution_time_seconds)

    joined_df_final.sort_by(List[String]("revenue"))

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

        
    print("exec time: ", execution_time_seconds)
   
    for i in range(joined_df_final.columns.size):
        print(joined_df_final.column_names[i])
        print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 1])

    print()

    for i in range(joined_df_final.columns.size):
        print(joined_df_final.column_names[i])
        print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 2])

    print()

    for i in range(joined_df_final.columns.size):
        print(joined_df_final.column_names[i])
        print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 3])
    

    print()

    for i in range(joined_df_final.columns.size):
        print(joined_df_final.column_names[i])
        print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 4])

    print()
    for i in range(joined_df_final.columns.size):
        print(joined_df_final.column_names[i])
        print(joined_df_final[joined_df_final.column_names[i]][joined_df_final.columns[0].size - 5])


fn test_query_2() raises:
    var pd = Python.import_module("pandas")

    var start_time = monotonic()

    var ps_partkey = Float64Array("../Data/tpch_med/ps_partkey_tensor")
    var ps_suppkey = Float64Array("../Data/tpch_med/ps_suppkey_tensor")
    var ps_supplycost = Float64Array("../Data/tpch_med/ps_supplycost_tensor")

    print(ps_partkey.size)

    var ps_col_data = List[Float64Array](ps_partkey, ps_suppkey, ps_supplycost)
    
    var ps_col_names = List[String]("partkey", "suppkey", "ps_supplycost")

    var df_partsupp = DataFrameF64(ps_col_data, ps_col_names)
    # var df_partsupp_outer = DataFrameF64(ps_col_data, ps_col_names)

    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    var p_type_arr = df_pt['p_type']

    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_size = Float64Array("../Data/tpch_med/p_size_tensor")
    var p_type = List[String]()
    p_type.resize(600000, "")

    for i in range(600000):
        p_type[i] = (p_type_arr[i].__str__())
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey, p_size)
    
    var part_col_names = List[String]("partkey", "p_size")

    var df_part = DataFrameF64(part_col_data, part_col_names)
    # var df_part_outer = DataFrameF64(part_col_data, part_col_names)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")
    var s_acctbal = Float64Array("../Data/tpch_med/s_acctbal_tensor")
    var s_name = Float64Array("../Data/tpch_med/s_name_tensor")

    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey, s_acctbal, s_name)
    
    var supp_col_names = List[String]("suppkey", "nationkey", "s_acctbal", "s_name")


    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)
    # var df_supplier_outer = DataFrameF64(supp_col_data, supp_col_names)


    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_regionkey = Float64Array("../Data/tpch_med/n_regionkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_regionkey, n_name)
    
    var nation_col_names = List[String]("nationkey", "regionkey", "n_name")

    var df_nation = DataFrameF64(nation_col_data, nation_col_names)
    # var df_nation_outer = DataFrameF64(nation_col_data, nation_col_names)

    var r_regionkey = Float64Array("../Data/tpch_med/r_regionkey_tensor")
    var r_name = Float64Array("../Data/tpch_med/r_name_tensor")

    print(r_regionkey.size)

    var region_col_data = List[Float64Array](r_regionkey, r_name)

    
    var region_col_names = List[String]("regionkey", "r_name")

    var df_region = DataFrameF64(region_col_data, region_col_names)
    # var df_region_outer = DataFrameF64(region_col_data, region_col_names)


    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("Data loading time: ", execution_time_seconds)

    start_time = monotonic()
    # get min supp cost for each part
    # ps_supplycost = (
	# 	select
	# 		min(ps_supplycost)
	# 	from
	# 		partsupp,
	# 		supplier,
	# 		nation,
	# 		region
	# 	where
	# 		p_partkey = ps_partkey
	# 		and s_suppkey = ps_suppkey
	# 		and s_nationkey = n_nationkey
	# 		and n_regionkey = r_regionkey
	# 		and r_name = 'EUROPE'
	# )

    # EUROPE
    df_region.select("r_name", "r_name", EQPredF64(), EQPredF64(), 35796.0, 35796.0, "")
    # filter by p_type like '%BRASS' and p_size = 15 before join
    filter_string_endwith(df_part, p_type, "BRASS")
    df_part.select("p_size", "p_size", EQPredF64(), EQPredF64(), 15.0, 15.0, "")

    # inner query to find min supp cost
    var joined_nr_df = inner_join_f64(df_nation, df_region, "regionkey")
    var joined_sn_df = inner_join_f64(df_supplier, joined_nr_df, "nationkey")
    var joined_pss_df = inner_join_f64(df_partsupp, joined_sn_df, "suppkey")
    var joined_psp_df = inner_join_f64(df_part, joined_pss_df, "partkey")
   
    var min_supplycost = inner_join_f64(df_part, joined_pss_df, "partkey")
    
    var aggregated_col_names = List[String]("partkey", "p_size_min", "suppkey_min",
                                            "ps_supplycost_min", "nationkey_min", "s_acctbal_min", 
                                            "s_name_min", "regionkey_min", "n_name_min", "r_name_min")

    # min supply cost for each part
    min_supplycost.groupby("partkey", "min", aggregated_col_names)

    # min supply cost for each part, with other information, such as supplier and nation
    var min_supplycost_final = inner_join_f64(joined_psp_df, min_supplycost, "partkey")

    # retain only the rows where the supply cost for a part is equal to the minimum supply cost for that part
    min_supplycost_final.select("ps_supplycost", "ps_supplycost_min", EQPredF64(), EQPredF64(), 0, 0, "COL")

    min_supplycost_final.sort_by(List[String]("s_acctbal", "n_name", "s_name", "partkey"))

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    
    print("final df: ", min_supplycost_final.columns[0].size)

    print("exec time: ", execution_time_seconds)

    for i in range(min_supplycost_final.columns.size):
        if (min_supplycost_final.column_names[i] == "partkey"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["partkey"]][min_supplycost_final.columns[0].size - 1])
        elif (min_supplycost_final.column_names[i] == "ps_supplycost_min"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["ps_supplycost_min"]][min_supplycost_final.columns[0].size - 1])
        elif (min_supplycost_final.column_names[i] == "s_name"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["s_name"]][min_supplycost_final.columns[0].size - 1])
        elif (min_supplycost_final.column_names[i] == "n_name"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["n_name"]][min_supplycost_final.columns[0].size - 1])
        elif (min_supplycost_final.column_names[i] == "s_acctbal"): 
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["s_acctbal"]][min_supplycost_final.columns[0].size - 1])
    print()
    for i in range(min_supplycost_final.columns.size):
        if (min_supplycost_final.column_names[i] == "partkey"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["partkey"]][min_supplycost_final.columns[0].size - 2])
        elif (min_supplycost_final.column_names[i] == "ps_supplycost_min"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["ps_supplycost_min"]][min_supplycost_final.columns[0].size - 2])
        elif (min_supplycost_final.column_names[i] == "s_name"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["s_name"]][min_supplycost_final.columns[0].size - 2])
        elif (min_supplycost_final.column_names[i] == "n_name"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["n_name"]][min_supplycost_final.columns[0].size - 2])
        elif (min_supplycost_final.column_names[i] == "s_acctbal"): 
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["s_acctbal"]][min_supplycost_final.columns[0].size - 2])
    print()
    for i in range(min_supplycost_final.columns.size):
        if (min_supplycost_final.column_names[i] == "partkey"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["partkey"]][min_supplycost_final.columns[0].size - 3])
        elif (min_supplycost_final.column_names[i] == "ps_supplycost_min"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["ps_supplycost_min"]][min_supplycost_final.columns[0].size - 3])
        elif (min_supplycost_final.column_names[i] == "s_name"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["s_name"]][min_supplycost_final.columns[0].size - 3])
        elif (min_supplycost_final.column_names[i] == "n_name"):
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["n_name"]][min_supplycost_final.columns[0].size - 3])
        elif (min_supplycost_final.column_names[i] == "s_acctbal"): 
            print(min_supplycost_final.column_names[i])
            print(min_supplycost_final[min_supplycost_final.col_name_to_idx["s_acctbal"]][min_supplycost_final.columns[0].size - 3])
    

fn test_query_7() raises:
    # var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_suppkey_arr = df['l_suppkey'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()


    # var l_orderkey = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_volume = Float64Array(17996609)
    # var l_suppkey = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_volume = Float64Array("../Data/tpch_med/l_discprice_tensor")
    var l_suppkey = Float64Array("../Data/tpch_med/l_suppkey_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")

    # for i in range(17996609):
    #     l_orderkey[i] = l_orderkey_arr[i].to_float64()
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     l_discount[i] = l_discount_arr[i].to_float64()
    #     l_volume[i] = l_extendedprice[i] * (1 - l_discount[i])
    #     l_suppkey[i] = l_suppkey_arr[i].to_float64()
    #     l_shipdate[i] = l_shipdate_arr[i].to_float64()
        
    print(l_orderkey.size)

    var col_data = List[Float64Array](l_orderkey, l_extendedprice, l_discount, l_volume, l_suppkey, l_shipdate)

    var col_names = List[String]("orderkey", "l_extendedprice", "l_discount", "l_volume", "suppkey", "l_shipdate")

    var df_lineitem = DataFrameF64(col_data, col_names)

    

    # var file_path_supp = 'supplier.csv'
    # var df_supp = pd.read_csv(file_path_supp)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # var s_nationkey_arr = df_supp['s_nationkey'].to_numpy()

    # var s_suppkey = Float64Array(30000)
    # var s_nationkey = Float64Array(30000)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")

    # for i in range(30000):
    #     s_suppkey[i] = s_suppkey_arr[i].to_float64()
    #     s_nationkey[i] = s_nationkey_arr[i].to_float64()

    
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey)
    
    var supp_col_names = List[String]("suppkey", "nationkey1")


    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)



    # var file_path_nation = 'nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # var n_name_arr = df_nat['n_name'].to_numpy()

    # var n_nationkey = Float64Array(25)
    # var n_name = Float64Array(25)
    
    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # for i in range(25):
    #     n_nationkey[i] = n_nationkey_arr[i].to_float64()
    #     n_name[i] = n_name_arr[i].to_float64()
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name)
    
    var nation_col_names = List[String]("nationkey1", "n_name1")

    var nation2_col_names = List[String]("nationkey2", "n_name2")

    var df_nation1 = DataFrameF64(nation_col_data, nation_col_names)
    var df_nation2 = DataFrameF64(nation_col_data, nation2_col_names)
    

    # var file_path_customer = 'customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()
    # var c_nationkey_arr = df_cust['c_nationkey'].to_numpy()

    # var c_custkey = Float64Array(450000)
    # var c_nationkey = Float64Array(450000)

    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")
    var c_nationkey = Float64Array("../Data/tpch_med/c_nationkey_tensor")

    # for i in range(450000):
    #     c_custkey[i] = c_custkey_arr[i].to_float64()
    #     c_nationkey[i] = c_nationkey_arr[i].to_float64()
    
    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey, c_nationkey)
    
    var cust_col_names = List[String]("custkey", "nationkey2")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    # var file_path_orders = 'orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_custkey_arr = df_ord['o_custkey'].to_numpy()
    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()

    # var o_custkey = Float64Array(4500000)
    # var o_orderkey = Float64Array(4500000)


    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    

    # for i in range(4500000):
    #     o_custkey[i] = o_custkey_arr[i].to_float64()
    #     o_orderkey[i] = o_orderkey_arr[i].to_float64()
       
    
    print(o_custkey.size)

    var orders_col_data = List[Float64Array](o_custkey, o_orderkey)

    var orders_col_names = List[String]("custkey", "orderkey")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    var start_time = monotonic()

    var start = perf_counter()
    # Subquery to get shipping table
    df_lineitem.select("l_shipdate", "l_shipdate", GTEPredF64(), LEPredF64(), 788918400.0, 852076800.0, "AND")
    var end = perf_counter()
    print("filter time:", end - start)


    # inner query to find min supp cost
    start = perf_counter()
    var joined_ls_df = inner_join_f64(df_supplier, df_lineitem, "suppkey")
    var joined_o_df = inner_join_f64(joined_ls_df, df_orders, "orderkey")
    var joined_c_df = inner_join_f64(joined_o_df, df_customer, "custkey")
    var joined_n1_df = inner_join_f64(joined_c_df, df_nation1, "nationkey1")
    var shipping = inner_join_f64(joined_n1_df, df_nation2, "nationkey2")
    end = perf_counter()
    print("join time:", end - start)

    start = perf_counter()
    var masks = List[List[Bool]]()
    var france_germany_mask = shipping.select_mask("n_name1", "n_name2", EQPredF64(), EQPredF64(), 38075.0, 52342.0, "AND")
    var germany_france_mask = shipping.select_mask("n_name1", "n_name2", EQPredF64(), EQPredF64(), 52342.0, 38075.0, "AND")
    masks.append(france_germany_mask)
    masks.append(germany_france_mask)
    shipping.select_complex(masks, "OR")
    
    end = perf_counter()
    print("filter time complex:", end - start)
    
    # print("Shipping df size: ", shipping.columns[0].size)
    # extract(year from l_shipdate) as l_year
    for i in range(shipping["l_shipdate"].size):
        var unix_time = shipping["l_shipdate"][i]
        shipping.columns[shipping.col_name_to_idx["l_shipdate"]][i] = (SIMD[DType.float64, 1](1970.0 + (unix_time / 31536000.0))).roundeven()
        

    # group by
	# supp_nation,
	# cust_nation,
	# l_year
    var group_by_cols = List[String]("n_name1", "n_name2", "l_shipdate")

    var aggregated_col_names = List[String]("supp_nation", "cust_nation", "l_year", "sum_suppkey",
                                            "sum_nationkey1", "sum_orderkey", "sum_l_extendedprice",
                                            "sum_l_discount", "revenue", "sum_custkey",
                                            "sum_nationkey2")

    shipping.groupby_multicol(group_by_cols, "sum", aggregated_col_names)
    shipping.sort_by(List[String]("supp_nation", "cust_nation", "l_year"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)

    
    print("final df size: ", shipping.columns[0].size)

    for i in range(shipping.columns.size):
        print(shipping.column_names[i])
        print(shipping.columns[i][0])
    print()
    for i in range(shipping.columns.size):
        print(shipping.column_names[i])
        print(shipping.columns[i][1])
    print()
    for i in range(shipping.columns.size):
        print(shipping.column_names[i])
        print(shipping.columns[i][2])
    print()
    for i in range(shipping.columns.size):
        print(shipping.column_names[i])
        print(shipping.columns[i][3])
    print()
    for i in range(shipping.columns.size):
        print(shipping.column_names[i])
        print(shipping.columns[i][4])
    print()
    for i in range(shipping.columns.size):
        print(shipping.column_names[i])
        print(shipping.columns[i][5])


fn test_query_8() raises:
    var pd = Python.import_module("pandas")
    pd.set_option('display.max_columns', None)

    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    var p_type_arr = df_pt['p_type']

    #var p_partkey = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_type = List[String]()
    p_type.resize(600000, "")

    for i in range(600000):
        # p_partkey[i] = p_partkey_arr[i].to_float64()
        p_type[i] = p_type_arr[i].__str__()
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey)
    
    var part_col_names = List[String]("partkey")

    var df_part = DataFrameF64(part_col_data, part_col_names)


    # var file_path = '../../data/tpch_3gb/lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_suppkey_arr = df['l_suppkey'].to_numpy()
    # var l_partkey_arr = df['l_partkey'].to_numpy()


    # var l_orderkey = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_volume = Float64Array(17996609)
    # var l_suppkey = Float64Array(17996609)
    # var l_partkey = Float64Array(17996609)

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_volume = Float64Array("../Data/tpch_med/l_discprice_tensor")
    var l_suppkey = Float64Array("../Data/tpch_med/l_suppkey_tensor")
    var l_partkey = Float64Array("../Data/tpch_med/l_partkey_tensor")

    # for i in range(17996609):
    #     l_orderkey[i] = l_orderkey_arr[i].to_float64()
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     l_discount[i] = l_discount_arr[i].to_float64()
    #     l_volume[i] = l_extendedprice[i] * (1 - l_discount[i])
    #     l_suppkey[i] = l_suppkey_arr[i].to_float64()
    #     l_partkey[i] = l_partkey_arr[i].to_float64()
        
    print(l_orderkey.size)

    var col_data = List[Float64Array](l_orderkey, l_extendedprice, l_discount, l_volume, l_suppkey, l_partkey)

    var col_names = List[String]("orderkey", "l_extendedprice", "l_discount", "l_volume", "suppkey", "partkey")

    var df_lineitem = DataFrameF64(col_data, col_names)

    

    # var file_path_supp = '../../data/tpch_3gb/supplier.csv'
    # var df_supp = pd.read_csv(file_path_supp)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # var s_nationkey_arr = df_supp['s_nationkey'].to_numpy()

    # var s_suppkey = Float64Array(30000)
    # var s_nationkey = Float64Array(30000)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")

    # for i in range(30000):
    #     s_suppkey[i] = s_suppkey_arr[i].to_float64()
    #     s_nationkey[i] = s_nationkey_arr[i].to_float64()

    
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey)
     
    var supp_col_names = List[String]("suppkey", "nationkey2")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)


    # var file_path_nation = '../../data/tpch_3gb/nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # var n_name_arr = df_nat['n_name'].to_numpy()
    # var n_regionkey_arr = df_nat['n_regionkey'].to_numpy()

    # var n_nationkey = Float64Array(25)
    # var n_name = Float64Array(25)
    # var n_regionkey = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")
    var n_regionkey = Float64Array("../Data/tpch_med/n_regionkey_tensor")

    # for i in range(25):
    #     n_nationkey[i] = n_nationkey_arr[i].to_float64()
    #     n_name[i] = n_name_arr[i].to_float64()
    #     n_regionkey[i] = n_regionkey_arr[i].to_float64()
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name, n_regionkey)
    
    var nation_col_names = List[String]("nationkey1", "n_name1", "regionkey1")
    
    var nation2_col_names = List[String]("nationkey2", "n_name2", "regionkey2")

    var df_nation1 = DataFrameF64(nation_col_data, nation_col_names)
    var df_nation2 = DataFrameF64(nation_col_data, nation2_col_names)
    

    # var file_path_customer = '../../data/tpch_3gb/customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()
    # var c_nationkey_arr = df_cust['c_nationkey'].to_numpy()

    # var c_custkey = Float64Array(450000)
    # var c_nationkey = Float64Array(450000)

    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")
    var c_nationkey = Float64Array("../Data/tpch_med/c_nationkey_tensor")

    # for i in range(450000):
    #     c_custkey[i] = c_custkey_arr[i].to_float64()
    #     c_nationkey[i] = c_nationkey_arr[i].to_float64()
    
    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey, c_nationkey)
    
    var cust_col_names = List[String]("custkey", "nationkey1")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    # var file_path_orders = '../../data/tpch_3gb/orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_custkey_arr = df_ord['o_custkey'].to_numpy()
    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_orderdate_arr = df_ord['o_orderdate'].to_numpy()

    # var o_custkey = Float64Array(4500000)
    # var o_orderkey = Float64Array(4500000)
    # var o_orderdate = Float64Array(4500000)

    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")
    

    # for i in range(4500000):
    #     o_custkey[i] = o_custkey_arr[i].to_float64()
    #     o_orderkey[i] = o_orderkey_arr[i].to_float64()
    #     o_orderdate[i] = o_orderdate_arr[i].to_float64()
       
    
    print(o_custkey.size)

    var orders_col_data = List[Float64Array](o_custkey, o_orderkey, o_orderdate)
    
    var orders_col_names = List[String]("custkey", "orderkey", "o_orderdate")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    # var file_path_region = '../../data/tpch_3gb/region.csv'
    # var df_reg = pd.read_csv(file_path_region)
    # print(df_reg.head())
    # print(df_reg.shape)

    # var r_regionkey_arr = df_reg['r_regionkey'].to_numpy()
    # var r_name_arr = df_reg['r_name'].to_numpy()

    # var r_regionkey = Float64Array(5)
    # var r_name = Float64Array(5)

    var r_regionkey = Float64Array("../Data/tpch_med/r_regionkey_tensor")
    var r_name = Float64Array("../Data/tpch_med/r_name_tensor")


    # for i in range(5):
    #     r_regionkey[i] = r_regionkey_arr[i].to_float64()
    #     r_name[i] = r_name_arr[i].to_float64()
    
    print(r_regionkey.size)

    var region_col_data = List[Float64Array](r_regionkey, r_name)
    
    var region_col_names = List[String]("regionkey1", "r_name")

    var df_region = DataFrameF64(region_col_data, region_col_names)

    var start_time = monotonic()
    # Subquery to get shipping table
    df_orders.select("o_orderdate", "o_orderdate", GTEPredF64(), LEPredF64(), 788918400.0, 852076800.0, "AND")
    filter_string_equal(df_part, p_type, "ECONOMY ANODIZED STEEL")
    df_region.select("r_name", "r_name", EQPredF64(), EQPredF64(), 3070.0, 3070.0, "")

    
    # inner query to create all_nations table
    var joined_pl_df = inner_join_f64(df_part, df_lineitem, "partkey")
    var joined_s_df = inner_join_f64(df_supplier, joined_pl_df, "suppkey")
    var joined_o_df = inner_join_f64(joined_s_df, df_orders, "orderkey")
    var joined_c_df = inner_join_f64(joined_o_df, df_customer, "custkey")
    var joined_n1_df = inner_join_f64(joined_c_df, df_nation1, "nationkey1")
    var joined_n1_r_df = inner_join_f64(joined_n1_df, df_region, "regionkey1")

    var all_nations = inner_join_f64(joined_n1_r_df, df_nation2, "nationkey2")
    var brazil = inner_join_f64(joined_n1_r_df, df_nation2, "nationkey2")

    
    # extract(year from o_orderdate) as o_year
    for i in range(all_nations["o_orderdate"].size):
        var unix_time = all_nations["o_orderdate"][i]
        all_nations.columns[all_nations.col_name_to_idx["o_orderdate"]][i] = (SIMD[DType.float64, 1](1970.0 + (unix_time / 31536000.0))).roundeven()
        brazil.columns[brazil.col_name_to_idx["o_orderdate"]][i] = (SIMD[DType.float64, 1](1970.0 + (unix_time / 31536000.0))).roundeven()
    
    var brazil_mask = all_nations.select_mask("n_name2", "n_name2", EQPredF64(), EQPredF64(), 62514.0, 62514.0, "")

    var aggregated_col_names = List[String]("o_year", "suppkey", "nationkey2", "partkey", 
                                            "orderkey", "l_extendedprice", "l_discount", 
                                            "mkt_share", "custkey", "nationkey1", "n_name1", "regionkey1", 
                                            "n_name2", "regionkey2", "r_name")


    
    all_nations.groupby("o_orderdate", "sum", aggregated_col_names)
    
    brazil.groupby_conditional("o_orderdate", "sum", brazil_mask, aggregated_col_names)

    for i in range(brazil[brazil.col_name_to_idx["mkt_share"]].size):
        brazil[brazil.col_name_to_idx["mkt_share"]][i] /= all_nations[all_nations.col_name_to_idx["mkt_share"]][i]

    brazil.sort_by(List[String]("o_year"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)

    for i in range(brazil.columns.size):
        if (brazil.column_names[i] == "o_year"):
            print(brazil.column_names[i])
            print(brazil[brazil.col_name_to_idx["o_year"]][0])
        elif (brazil.column_names[i] == "mkt_share"):
            print(brazil.column_names[i])
            print(brazil[brazil.col_name_to_idx["mkt_share"]][0])
    print()
    for i in range(brazil.columns.size):
        if (brazil.column_names[i] == "o_year"):
            print(brazil.column_names[i])
            print(brazil[brazil.col_name_to_idx["o_year"]][1])
        elif (brazil.column_names[i] == "mkt_share"):
            print(brazil.column_names[i])
            print(brazil[brazil.col_name_to_idx["mkt_share"]][1])
    print()
    for i in range(brazil.columns.size):
        if (brazil.column_names[i] == "o_year"):
            print(brazil.column_names[i])
            print(brazil[brazil.col_name_to_idx["o_year"]][2])
        elif (brazil.column_names[i] == "mkt_share"):
            print(brazil.column_names[i])
            print(brazil[brazil.col_name_to_idx["mkt_share"]][2])
       


fn test_query_9() raises:
    var pd = Python.import_module("pandas")
    pd.set_option('display.max_columns', None)

    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    var p_name_arr = df_pt['p_name']

    # var p_partkey = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_name = List[String]()
    p_name.resize(600000, "")

    for i in range(600000):
        #p_partkey[i] = p_partkey_arr[i].to_float64()
        p_name[i] = (p_name_arr[i].__str__())
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey)
    
    var part_col_names = List[String]("partkey")

    var df_part = DataFrameF64(part_col_data, part_col_names)


    # var file_path = '../../data/tpch_3gb/lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_quantity_arr = df['l_quantity'].to_numpy()
    # var l_suppkey_arr = df['l_suppkey'].to_numpy()
    # var l_partkey_arr = df['l_partkey']


    # var l_orderkey = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)
    # var l_suppkey = Float64Array(17996609)
    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")
    var l_suppkey = Float64Array("../Data/tpch_med/l_suppkey_tensor")
    var l_partkey = Float64Array("../Data/tpch_med/l_partkey_tensor")

    # for i in range(17996609):
        # l_orderkey[i] = l_orderkey_arr[i].to_float64()
        # l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
        # l_discount[i] = l_discount_arr[i].to_float64()
        # l_quantity[i] = l_quantity_arr[i].to_float64()
        # l_suppkey[i] = l_suppkey_arr[i].to_float64()
        # l_partkey[i] = Float64(l_partkey_arr[i])
    
    #l_partkey.data.tofile(Path("../Data/tpch_med/l_partkey_tensor"))

    print(l_partkey.size)

    var col_data = List[Float64Array](l_orderkey, l_extendedprice, l_discount, l_quantity, l_suppkey, l_partkey)

    # var col1_name = "orderkey"
    # var col2_name = "l_extendedprice"
    # var col3_name = "l_discount"
    # var col4_name = "l_quantity"
    # var col5_name = "suppkey"
    # var col6_name = "partkey"

    var col_names = List[String]("orderkey", "l_extendedprice", "l_discount", "l_quantity", "suppkey", "partkey")

    var df_lineitem = DataFrameF64(col_data, col_names)

    
    # var file_path_psupp = '../../data/tpch_3gb/partsupp.csv'
    # var df_psupp = pd.read_csv(file_path_psupp)
    # print(df_psupp.head())
    # print(df_psupp.shape)

    # var ps_partkey_arr = df_psupp['ps_partkey'].to_numpy()
    # var ps_suppkey_arr = df_psupp['ps_suppkey'].to_numpy()
    # var ps_supplycost_arr = df_psupp['ps_supplycost'].to_numpy()

    # var ps_partkey = Float64Array(2400000)
    # var ps_suppkey = Float64Array(2400000)
    # var ps_supplycost = Float64Array(2400000)

    var ps_partkey = Float64Array("../Data/tpch_med/ps_partkey_tensor")
    var ps_suppkey = Float64Array("../Data/tpch_med/ps_suppkey_tensor")
    var ps_supplycost = Float64Array("../Data/tpch_med/ps_supplycost_tensor")

    # for i in range(2400000):
    #     ps_partkey[i] = ps_partkey_arr[i].to_float64()
    #     ps_suppkey[i] = ps_suppkey_arr[i].to_float64()
    #     ps_supplycost[i] = ps_supplycost_arr[i].to_float64()
    
    print(ps_partkey.size)

    var ps_col_data = List[Float64Array](ps_partkey, ps_suppkey, ps_supplycost)
    
    var ps_col_names = List[String]("partkey", "ps_suppkey", "ps_supplycost")

    var df_partsupp = DataFrameF64(ps_col_data, ps_col_names)


    # var file_path_supp = 'supplier.csv'
    # var df_supp = pd.read_csv(file_path_supp)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # var s_nationkey_arr = df_supp['s_nationkey'].to_numpy()

    # var s_suppkey = Float64Array(30000)
    # var s_nationkey = Float64Array(30000)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")

    # for i in range(30000):
    #     s_suppkey[i] = s_suppkey_arr[i].to_float64()
    #     s_nationkey[i] = s_nationkey_arr[i].to_float64()

    
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey)
    
    var supp_col_names = List[String]("suppkey", "nationkey")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)


    # var file_path_nation = 'nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # var n_name_arr = df_nat['n_name'].to_numpy()

    # var n_nationkey = Float64Array(25)
    # var n_name = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # for i in range(25):
    #     n_nationkey[i] = n_nationkey_arr[i].to_float64()
    #     n_name[i] = n_name_arr[i].to_float64()
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name)
    
    var nation_col_names = List[String]("nationkey", "n_name")

    var df_nation = DataFrameF64(nation_col_data, nation_col_names)
    

    # var file_path_orders = 'orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_orderdate_arr = df_ord['o_orderdate'].to_numpy()

    # var o_orderkey = Float64Array(4500000)
    # var o_orderdate = Float64Array(4500000)

    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")
    

    # for i in range(4500000):
    #     o_orderkey[i] = o_orderkey_arr[i].to_float64()
    #     o_orderdate[i] = o_orderdate_arr[i].to_float64()
       
    
    print(o_orderkey.size)

    var orders_col_data = List[Float64Array](o_orderkey, o_orderdate)
    
    var orders_col_names = List[String]("orderkey", "o_orderdate")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    var start_time = monotonic()

    var filter_start_time = monotonic()

    filter_string_contains(df_part, p_name, "green")

    var filter_end_time = monotonic()

    print("filter exec time: ", (filter_end_time - filter_start_time) / 1000000000)

    var joined_sn_df = inner_join_f64(df_supplier, df_nation, "nationkey")
    var joined_ls_df = inner_join_f64(df_lineitem, joined_sn_df, "suppkey")
    var joined_lp_df = inner_join_f64(joined_ls_df, df_part, "partkey")
    var joined_lps_df = inner_join_f64(joined_lp_df, df_partsupp, "partkey")
    joined_lps_df.select("suppkey", "ps_suppkey", EQPredF64(), EQPredF64(), 0, 0, "COL")

    var joined_final = inner_join_f64(joined_lps_df, df_orders, "orderkey")

    var extendedprice = joined_final["l_extendedprice"]
    var discount = joined_final["l_discount"]
    var supplycost = joined_final["ps_supplycost"]
    var quantity = joined_final["l_quantity"]
    var unix_time = joined_final["o_orderdate"]

    var amount = Float64Array(joined_final.columns[0].size)

    for i in range(amount.size):
        amount[i] = (extendedprice[i] * (1 - discount[i])) - (supplycost[i] * quantity[i])
        joined_final.columns[joined_final.col_name_to_idx["o_orderdate"]][i] = (SIMD[DType.float64, 1](1970.0 + (unix_time[i] / 31536000.0))).roundeven()
        
    joined_final.append_column(amount^, "amount")

    var aggregated_col_names = List[String]("nation", "o_year", "orderkey", "l_extendedprice", 
                                            "l_discount", "l_quantity", "suppkey", 
                                            "partkey", "nationkey", "ps_suppkey", "ps_supplycost", "sum_profit")

       
    var group_by_cols = List[String]("n_name", "o_orderdate")
    joined_final.groupby_multicol(group_by_cols, "sum", aggregated_col_names)
    

    joined_final.sort_by(List[String]("nation", "o_year"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("final joined size: ", joined_final.columns[0].size)

    for i in range(joined_final.columns.size):
        if (joined_final.column_names[i] == "o_year"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["o_year"]][0])
        elif (joined_final.column_names[i] == "nation"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["nation"]][0])
        elif (joined_final.column_names[i] == "sum_profit"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["sum_profit"]][0])
    print()
    for i in range(joined_final.columns.size):
        if (joined_final.column_names[i] == "o_year"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["o_year"]][1])
        elif (joined_final.column_names[i] == "nation"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["nation"]][1])
        elif (joined_final.column_names[i] == "sum_profit"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["sum_profit"]][1])
    print()
    for i in range(joined_final.columns.size):
        if (joined_final.column_names[i] == "o_year"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["o_year"]][2])
        elif (joined_final.column_names[i] == "nation"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["nation"]][2])
        elif (joined_final.column_names[i] == "sum_profit"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["sum_profit"]][2])


fn test_query_10() raises:
    # var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)


    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_returnflag_arr = df['l_returnflag'].to_numpy()


    # var l_orderkey = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_returnflag = Float64Array(17996609)
    # # var revenue = Float64Array(17996609)
    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_returnflag = Float64Array("../Data/tpch_med/l_returnflag_tensor")

    # for i in range(17996609):
    #     l_orderkey[i] = l_orderkey_arr[i].to_float64()
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     l_discount[i] = l_discount_arr[i].to_float64()
    #     l_returnflag[i] = l_returnflag_arr[i].to_float64()
    #     # revenue[i] = l_extendedprice[i] * (1 - l_discount[i])
        
    print(l_orderkey.size)

    var col_data = List[Float64Array](l_orderkey, l_extendedprice, l_discount, l_returnflag)

    var col_names = List[String]("orderkey", "l_extendedprice", "l_discount", "l_returnflag")

    var df_lineitem = DataFrameF64(col_data, col_names)

    

    # var file_path_nation = 'nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # var n_name_arr = df_nat['n_name'].to_numpy()

    # var n_nationkey = Float64Array(25)
    # var n_name = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # for i in range(25):
    #     n_nationkey[i] = n_nationkey_arr[i].to_float64()
    #     n_name[i] = n_name_arr[i].to_float64()
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name)

    var nation_col1_name = "nationkey"
    var nation_col2_name = "n_name"
    
    var nation_col_names = List[String]("nationkey", "n_name")

    var df_nation = DataFrameF64(nation_col_data, nation_col_names)
    


    # var file_path_orders = 'orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_custkey_arr = df_ord['o_custkey'].to_numpy()
    # var o_orderdate_arr = df_ord['o_orderdate'].to_numpy()
    

    # var o_orderkey = Float64Array(4500000)
    # var o_custkey = Float64Array(4500000)
    # var o_orderdate = Float64Array(4500000)
    
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")
    

    # for i in range(4500000):
    #     o_orderkey[i] = o_orderkey_arr[i].to_float64()
    #     o_custkey[i] = o_custkey_arr[i].to_float64()
    #     o_orderdate[i] = o_orderdate_arr[i].to_float64()
       
    
    print(o_orderkey.size)

    var orders_col_data = List[Float64Array](o_orderkey, o_custkey, o_orderdate) 
    
    var orders_col_names = List[String]("orderkey", "custkey", "o_orderdate")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    # var file_path_customer = 'customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()
    # var c_nationkey_arr = df_cust['c_nationkey'].to_numpy()
    # var c_acctbal_arr = df_cust['c_acctbal'].to_numpy()

    # var c_custkey = Float64Array(450000)
    # var c_nationkey = Float64Array(450000)
    # var c_acctbal = Float64Array(450000)

    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")
    var c_nationkey = Float64Array("../Data/tpch_med/c_nationkey_tensor")
    var c_acctbal = Float64Array("../Data/tpch_med/c_acctbal_tensor")

    # for i in range(450000):
    #     c_custkey[i] = c_custkey_arr[i].to_float64()
    #     c_nationkey[i] = c_nationkey_arr[i].to_float64()
    #     c_acctbal[i] = c_acctbal_arr[i].to_float64()
    
    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey, c_nationkey, c_acctbal)

    var cust_col1_name = "custkey"
    var cust_col2_name = "nationkey"
    var cust_col3_name = "c_acctbal"
    
    var cust_col_names = List[String]("custkey", "nationkey", "c_acctbal")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    var start_time = monotonic()

    df_lineitem.select("l_returnflag", "l_returnflag", EQPredF64(), EQPredF64(), 82.0, 82.0, "")
    df_orders.select("o_orderdate", "o_orderdate", GTEPredF64(), LTPredF64(), 750643200.0, 757382400.0, "AND")

    var joined_cn_df = inner_join_f64(df_customer, df_nation, "nationkey")
    var joined_oc_df = inner_join_f64(df_orders, joined_cn_df, "custkey")
    var joined_final = inner_join_f64(df_lineitem, joined_oc_df, "orderkey")
    
    var extendedprice = joined_final["l_extendedprice"]
    var discount = joined_final["l_discount"]

    var revenue = Float64Array(joined_final.columns[0].size)

    for i in range(revenue.size):
        revenue[i] = (extendedprice[i] * (1 - discount[i]))
    
    joined_final.append_column(revenue^, "revenue")
    
    var aggregated_col_names = List[String]("custkey", "c_acctbal", "n_name", "orderkey",
                                            "l_extendedprice", "l_discount", "l_returnflag", 
                                            "o_orderdate", "nationkey", "revenue")

       
    var group_by_cols = List[String]("custkey", "c_acctbal", "n_name")
    joined_final.groupby_multicol(group_by_cols, "sum", aggregated_col_names)
    
    joined_final.sort_by(List[String]("revenue"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("final joined size: ", joined_final.columns[0].size)


    for i in range(joined_final.columns.size):
        if (joined_final.column_names[i] == "custkey"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["custkey"]][joined_final.columns[0].size - 1])
        elif (joined_final.column_names[i] == "c_acctbal"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["c_acctbal"]][joined_final.columns[0].size - 1])
        elif (joined_final.column_names[i] == "n_name"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["n_name"]][joined_final.columns[0].size - 1])
        elif (joined_final.column_names[i] == "revenue"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["revenue"]][joined_final.columns[0].size - 1])
    print()
    for i in range(joined_final.columns.size):
        if (joined_final.column_names[i] == "custkey"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["custkey"]][joined_final.columns[0].size - 2])
        elif (joined_final.column_names[i] == "c_acctbal"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["c_acctbal"]][joined_final.columns[0].size - 2])
        elif (joined_final.column_names[i] == "n_name"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["n_name"]][joined_final.columns[0].size - 2])
        elif (joined_final.column_names[i] == "revenue"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["revenue"]][joined_final.columns[0].size - 2])
    print()
    for i in range(joined_final.columns.size):
        if (joined_final.column_names[i] == "custkey"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["custkey"]][joined_final.columns[0].size - 3])
        elif (joined_final.column_names[i] == "c_acctbal"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["c_acctbal"]][joined_final.columns[0].size - 3])
        elif (joined_final.column_names[i] == "n_name"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["n_name"]][joined_final.columns[0].size - 3])
        elif (joined_final.column_names[i] == "revenue"):
            print(joined_final.column_names[i])
            print(joined_final[joined_final.col_name_to_idx["revenue"]][joined_final.columns[0].size - 3])
   

fn test_query_11() raises:
    # var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    # var file_path_psupp = '../../data/tpch_3gb/partsupp.csv'
    # var df_psupp = pd.read_csv(file_path_psupp)
    # print(df_psupp.head())
    # print(df_psupp.shape)

    # var ps_partkey_arr = df_psupp['ps_partkey'].to_numpy()
    # var ps_suppkey_arr = df_psupp['ps_suppkey'].to_numpy()
    # var ps_supplycost_arr = df_psupp['ps_supplycost'].to_numpy()
    # var ps_availqty_arr = df_psupp['ps_availqty']

    # var ps_partkey = Float64Array(2400000)
    # var ps_suppkey = Float64Array(2400000)
    # var ps_supplycost = Float64Array(2400000)
    # var ps_availqty = Float64Array(2400000)

    var ps_partkey = Float64Array("../Data/tpch_med/ps_partkey_tensor")
    var ps_suppkey = Float64Array("../Data/tpch_med/ps_suppkey_tensor")
    var ps_supplycost = Float64Array("../Data/tpch_med/ps_supplycost_tensor")
    var ps_availqty = Float64Array("../Data/tpch_med/ps_availqty_tensor")

    # for i in range(2400000):
    #     # ps_partkey[i] = ps_partkey_arr[i].to_float64()
    #     # ps_suppkey[i] = ps_suppkey_arr[i].to_float64()
    #     # ps_supplycost[i] = ps_supplycost_arr[i].to_float64()
    #     ps_availqty[i] = ps_availqty_arr[i].to_float64()
    
    print(ps_availqty.size)
    # ps_availqty.data.tofile(Path("../Data/tpch_med/ps_availqty_tensor"))

    var ps_col_data = List[Float64Array](ps_partkey, ps_suppkey, ps_supplycost, ps_availqty)

    # var ps_col1_name = "ps_partkey"
    # var ps_col2_name = "suppkey"
    # var ps_col3_name = "ps_supplycost"
    # var ps_col4_name = "ps_availqty"
    
    var ps_col_names = List[String]("ps_partkey", "suppkey", "ps_supplycost", "ps_availqty")

    var df_partsupp = DataFrameF64(ps_col_data, ps_col_names)


    # # var file_path_supp = 'supplier.csv'
    # # var df_supp = pd.read_csv(file_path_supp)
    # # print(df_supp.head())
    # # print(df_supp.shape)

    # # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # # var s_nationkey_arr = df_supp['s_nationkey'].to_numpy()

    # # var s_suppkey = Float64Array(30000)
    # # var s_nationkey = Float64Array(30000)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")

    # # for i in range(30000):
    # #     s_suppkey[i] = s_suppkey_arr[i].to_float64()
    # #     s_nationkey[i] = s_nationkey_arr[i].to_float64()

    
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey)
   
    var supp_col_names = List[String]("suppkey", "nationkey")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)



    # # var file_path_nation = 'nation.csv'
    # # var df_nat = pd.read_csv(file_path_nation)
    # # print(df_nat.head())
    # # print(df_nat.shape)

    # # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # # var n_name_arr = df_nat['n_name'].to_numpy()

    # # var n_nationkey = Float64Array(25)
    # # var n_name = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # # for i in range(25):
    # #     n_nationkey[i] = n_nationkey_arr[i].to_float64()
    # #     n_name[i] = n_name_arr[i].to_float64()
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name)
    
    var nation_col_names = List[String]("nationkey", "n_name")


    var df_nation = DataFrameF64(nation_col_data, nation_col_names)

    var start_time = monotonic()

    # n_name = 'GERMANY'
    df_nation.select("n_name", "n_name", EQPredF64(), EQPredF64(), 52342.0, 52342.0, "")
    
    var joined_sn_df = inner_join_f64(df_supplier, df_nation, "nationkey")
    var joined_pss_df = inner_join_f64(df_partsupp, joined_sn_df, "suppkey")

    var value = Float64Array(joined_pss_df.columns[0].size)
    var value_percent = Float64Array(joined_pss_df.columns[0].size)

    var supplycost = joined_pss_df["ps_supplycost"]
    var availqty = joined_pss_df["ps_availqty"]


    for i in range(value_percent.size):
        value[i] = (supplycost[i] * availqty[i])
        value_percent[i] = (value[i] * 0.00001)
    
    var value_percent_sum = pairwise_sum_f64(value_percent, value_percent.size, 0, value_percent.size)

    joined_pss_df.append_column(value^, "value")
    joined_pss_df.append_column(value_percent^, "value_percent")


    var aggregated_col_names = List[String]("ps_partkey", "suppkey", "ps_supplycost", "ps_availqty",
                                            "nationkey", "n_name", "value_sum", "value_percent_sum")


    joined_pss_df.groupby("ps_partkey", "sum", aggregated_col_names)
    joined_pss_df.select("value_sum", "value_sum", GTPredF64(), GTPredF64(), value_percent_sum, value_percent_sum, "")
    
    joined_pss_df.sort_by(List[String]("value_sum"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("final joined size: ", joined_pss_df.columns[0].size)
    print("value sum percent: ", value_percent_sum)

    print()

    for i in range(joined_pss_df.columns.size):
        if (joined_pss_df.column_names[i] == "ps_partkey"):
                print(joined_pss_df.column_names[i])
                print(joined_pss_df[joined_pss_df.col_name_to_idx["ps_partkey"]][joined_pss_df.columns[0].size - 1])
        if (joined_pss_df.column_names[i] == "value_sum"):
                print(joined_pss_df.column_names[i])
                print(joined_pss_df[joined_pss_df.col_name_to_idx["value_sum"]][joined_pss_df.columns[0].size - 1])
            
    print()

    for i in range(joined_pss_df.columns.size):
        if (joined_pss_df.column_names[i] == "ps_partkey"):
                print(joined_pss_df.column_names[i])
                print(joined_pss_df[joined_pss_df.col_name_to_idx["ps_partkey"]][joined_pss_df.columns[0].size - 2])
        if (joined_pss_df.column_names[i] == "value_sum"):
                print(joined_pss_df.column_names[i])
                print(joined_pss_df[joined_pss_df.col_name_to_idx["value_sum"]][joined_pss_df.columns[0].size - 2])
    
    print()

    for i in range(joined_pss_df.columns.size):
        if (joined_pss_df.column_names[i] == "ps_partkey"):
                print(joined_pss_df.column_names[i])
                print(joined_pss_df[joined_pss_df.col_name_to_idx["ps_partkey"]][joined_pss_df.columns[0].size - 3])
        if (joined_pss_df.column_names[i] == "value_sum"):
                print(joined_pss_df.column_names[i])
                print(joined_pss_df[joined_pss_df.col_name_to_idx["value_sum"]][joined_pss_df.columns[0].size - 3])



fn test_query_12() raises:
    # var pd = Python.import_module("pandas")
    # # var np = Python.import_module("numpy")

    # pd.set_option('display.max_columns', None)
    # var file_path = '../../data/tpch_3gb/lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()
    # var l_commitdate_arr = df['l_commitdate'].to_numpy()
    # var l_receiptdate_arr = df['l_receiptdate'].to_numpy()
    # var l_shipmode_arr = df['l_shipmode']

    # var l_orderkey = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)
    # var l_commitdate = Float64Array(17996609)
    # var l_receiptdate = Float64Array(17996609)
    # var l_shipmode = Float64Array(17996609)

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")
    var l_commitdate = Float64Array("../Data/tpch_med/l_commitdate_tensor")
    var l_receiptdate = Float64Array("../Data/tpch_med/l_receiptdate_tensor")
    var l_shipmode = Float64Array("../Data/tpch_med/l_shipmode_tensor")

    # for i in range(17996609):
    #     # l_orderkey[i] = l_orderkey_arr[i].to_float64()
    #     # l_shipdate[i] = l_shipdate_arr[i].to_float64()
    #     # l_commitdate[i] = l_commitdate_arr[i].to_float64()
    #     # l_receiptdate[i] = l_receiptdate_arr[i].to_float64()
    #     l_shipmode[i] = l_shipmode_arr[i].to_float64()
        
    print(l_shipmode.size)
    # l_shipmode.data.tofile(Path("../Data/tpch_med/l_shipmode_tensor"))

    var col_data = List[Float64Array](l_orderkey, l_shipdate, l_commitdate, l_receiptdate, l_shipmode)

    var col_names = List[String]("orderkey", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipmode")

    var df_lineitem = DataFrameF64(col_data, col_names)


    # var file_path_orders = 'orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_orderpriority_arr = df_ord['o_orderpriority'].to_numpy()

    # var o_orderkey = Float64Array(4500000)
    # var o_orderpriority = Float64Array(4500000)

    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderpriority = Float64Array("../Data/tpch_med/o_orderpriority_tensor")

    # for i in range(4500000):
    #     o_orderkey[i] = o_orderkey_arr[i].to_float64()
    #     o_orderpriority[i] = o_orderpriority_arr[i].to_float64()
    
    print(o_orderkey.size)

    var orders_col_data = List[Float64Array](o_orderkey, o_orderpriority)
    
    var orders_col_names = List[String]("orderkey", "o_orderpriority")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)

    # shipmode_mapping = {
    #     'TRUCK': 1,
    #     'AIR': 2,
    #     'FOB': 3,
    #     'REG AIR': 4,
    #     'MAIL': 5,
    #     'SHIP': 6,
    #     'RAIL': 7
    # }

    var start_time = monotonic()

    # l_shipmode in ('MAIL', 'SHIP')
    # l_shipmode in (5, 6)
    df_lineitem.select("l_shipmode", "l_shipmode", EQPredF64(), EQPredF64(), 5.0, 6.0, "OR")
    df_lineitem.select("l_commitdate", "l_receiptdate", LTPredF64(), LTPredF64(), 0.0, 0.0, "COL")
    df_lineitem.select("l_shipdate", "l_commitdate", LTPredF64(), LTPredF64(), 0.0, 0.0, "COL")
    df_lineitem.select("l_receiptdate", "l_receiptdate", GTEPredF64(), LTPredF64(), 757382400.0, 788918400.0, "AND")

    var joined_lo_df = inner_join_f64(df_lineitem, df_orders, "orderkey")

    # get two masks, then append two columns for high and low line count
    var high_line_mask = joined_lo_df.select_mask("o_orderpriority", "o_orderpriority", EQPredF64(), EQPredF64(), 1.0, 2.0, "OR")
    var low_line_mask = joined_lo_df.select_mask("o_orderpriority", "o_orderpriority", NEQPredF64(), NEQPredF64(), 1.0, 2.0, "AND")

    var high_line = Float64Array(joined_lo_df.columns[0].size)
    var low_line = Float64Array(joined_lo_df.columns[0].size)

    for i in range(high_line_mask.size):
        if high_line_mask[i]:
            high_line[i] = 1.0
        else:
            high_line[i] = 0.0
        
        if low_line_mask[i]:
            low_line[i] = 1.0
        else:
            low_line[i] = 0.0
    
    joined_lo_df.append_column(high_line^, "high_line")
    joined_lo_df.append_column(low_line^, "low_line")

    var aggregated_col_names = List[String]("l_shipmode", "orderkey", "l_shipdate", "l_commitdate",
                                            "l_receiptdate", "o_orderpriority", "high_line_count", "low_line_count")

    # print("final joined size: ", joined_lo_df.columns[0].size)
    joined_lo_df.groupby("l_shipmode", "sum", aggregated_col_names)
    
    joined_lo_df.sort_by(List[String]("l_shipmode"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("grouped size: ", joined_lo_df.columns[0].size)

    print()

    for i in range(joined_lo_df.columns.size):
        if (joined_lo_df.column_names[i] == "l_shipmode"):
                print(joined_lo_df.column_names[i])
                print(joined_lo_df[joined_lo_df.col_name_to_idx["l_shipmode"]][0])
        if (joined_lo_df.column_names[i] == "high_line_count"):
                print(joined_lo_df.column_names[i])
                print(joined_lo_df[joined_lo_df.col_name_to_idx["high_line_count"]][0])
        if (joined_lo_df.column_names[i] == "low_line_count"):
                print(joined_lo_df.column_names[i])
                print(joined_lo_df[joined_lo_df.col_name_to_idx["low_line_count"]][0])
            
    print()

    for i in range(joined_lo_df.columns.size):
        if (joined_lo_df.column_names[i] == "l_shipmode"):
                    print(joined_lo_df.column_names[i])
                    print(joined_lo_df[joined_lo_df.col_name_to_idx["l_shipmode"]][1])
        if (joined_lo_df.column_names[i] == "high_line_count"):
                print(joined_lo_df.column_names[i])
                print(joined_lo_df[joined_lo_df.col_name_to_idx["high_line_count"]][1])
        if (joined_lo_df.column_names[i] == "low_line_count"):
                print(joined_lo_df.column_names[i])
                print(joined_lo_df[joined_lo_df.col_name_to_idx["low_line_count"]][1])


fn test_query_13() raises:
    var pd = Python.import_module("pandas")
    pd.set_option('display.max_columns', None)

    # var file_path_customer = 'customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()

    # var c_custkey = Float64Array(450000)
    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")

    # for i in range(450000):
    #     c_custkey[i] = c_custkey_arr[i].to_float64()
    
    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey)
    
    var cust_col_names = List[String]("custkey")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    var file_path_orders = '../../data/tpch_3gb/orders.csv'
    var df_ord = pd.read_csv(file_path_orders)
    print(df_ord.head())
    print(df_ord.shape)

    #var o_custkey_arr = df_ord['o_custkey'].to_numpy()
    var o_comment_arr = df_ord['o_comment']

    #var o_custkey = Float64Array(4500000)
    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_comment = List[String]()
    o_comment.resize(4500000, "")

    for i in range(4500000):
        #o_custkey[i] = o_custkey_arr[i].to_float64()
        o_comment[i] = (o_comment_arr[i].__str__())

    print(o_custkey.size)

    var orders_col_data = List[Float64Array](o_custkey)
    
    var orders_col_names = List[String]("custkey")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)
    
    var start_time = monotonic()

    filter_not_string_exists_before(df_orders, o_comment, "special", "requests")

    var joined_co_df = inner_join_f64(df_customer, df_orders, "custkey")
    var aggregated_col_names = List[String]("custkey", "customer_order_count")

    print("final joined size: ", joined_co_df.columns[0].size)

    joined_co_df.groupby("custkey", "count", aggregated_col_names)

    var aggregated_col_names2 = List[String]("c_count", "custdist")

    joined_co_df.groupby("customer_order_count", "count", aggregated_col_names2)
    
    joined_co_df.sort_by(List[String]("custdist", "c_count"))

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("grouped size: ", joined_co_df.columns[0].size)

    for i in range(joined_co_df.columns.size):
        if (joined_co_df.column_names[i] == "c_count"):
                    print(joined_co_df.column_names[i])
                    print(joined_co_df[joined_co_df.col_name_to_idx["c_count"]][joined_co_df.columns[0].size - 1])
        if (joined_co_df.column_names[i] == "custdist"):
                print(joined_co_df.column_names[i])
                print(joined_co_df[joined_co_df.col_name_to_idx["custdist"]][joined_co_df.columns[0].size - 1])
    
    print()

    for i in range(joined_co_df.columns.size):
        if (joined_co_df.column_names[i] == "c_count"):
                    print(joined_co_df.column_names[i])
                    print(joined_co_df[joined_co_df.col_name_to_idx["c_count"]][joined_co_df.columns[0].size - 2])
        if (joined_co_df.column_names[i] == "custdist"):
                print(joined_co_df.column_names[i])
                print(joined_co_df[joined_co_df.col_name_to_idx["custdist"]][joined_co_df.columns[0].size - 2])
    
    print()

    for i in range(joined_co_df.columns.size):
        if (joined_co_df.column_names[i] == "c_count"):
                    print(joined_co_df.column_names[i])
                    print(joined_co_df[joined_co_df.col_name_to_idx["c_count"]][joined_co_df.columns[0].size - 3])
        if (joined_co_df.column_names[i] == "custdist"):
                print(joined_co_df.column_names[i])
                print(joined_co_df[joined_co_df.col_name_to_idx["custdist"]][joined_co_df.columns[0].size - 3])
    

fn test_query_14() raises:
    var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_partkey_arr = df['l_partkey'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()


    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_partkey = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)

    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_partkey = Float64Array("../Data/tpch_med/l_partkey_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")

    # for i in range(17996609):
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     l_discount[i] = l_discount_arr[i].to_float64()
    #     l_partkey[i] = l_partkey_arr[i].to_float64()
    #     l_shipdate[i] = l_shipdate_arr[i].to_float64()
        
    print(l_partkey.size)

    var col_data = List[Float64Array](l_extendedprice, l_discount, l_partkey, l_shipdate)

    var col_names = List[String]("l_extendedprice", "l_discount", "partkey", "l_shipdate")

    var df_lineitem = DataFrameF64(col_data, col_names)


    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    var p_type_arr = df_pt['p_type']

    # var p_partkey = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_type = List[String]()
    p_type.resize(600000, "")

    for i in range(600000):
        # p_partkey[i] = p_partkey_arr[i].to_float64()
        p_type[i] = p_type_arr[i].__str__()
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey)
    
    var part_col_names = List[String]("partkey")

    var df_part = DataFrameF64(part_col_data, part_col_names)

    var df_part_promo = DataFrameF64(part_col_data, part_col_names)


    var start_time = monotonic()

    df_lineitem.select("l_shipdate", "l_shipdate", GTEPredF64(), LTPredF64(), 809913600.0, 812505600.0, "AND")
    filter_string_startwith(df_part_promo, p_type, "PROMO")

    var joined_lp_df = inner_join_f64(df_lineitem, df_part, "partkey")
    var joined_lp_promo_df = inner_join_f64(df_lineitem, df_part_promo, "partkey")

    # used to sum for revenue: promo / total
    var denominator_promo_revenue = Float64Array(joined_lp_df.columns[0].size)
    var numerator_promo_revenue = Float64Array(joined_lp_promo_df.columns[0].size)

    var extendedprice_denom = joined_lp_df["l_extendedprice"]
    var discount_denom = joined_lp_df["l_discount"]

    var extendedprice_nume = joined_lp_promo_df["l_extendedprice"]
    var discount_nume = joined_lp_promo_df["l_discount"]

    for i in range(denominator_promo_revenue.size):
        denominator_promo_revenue[i] = (extendedprice_denom[i] * (1 - discount_denom[i]))
    
    for i in range(numerator_promo_revenue.size):
        numerator_promo_revenue[i] = (extendedprice_nume[i] * (1 - discount_nume[i]))

    
    var promo_revenue = pairwise_sum_f64(numerator_promo_revenue, numerator_promo_revenue.size, 0, numerator_promo_revenue.size)
    var total_revenue = pairwise_sum_f64(denominator_promo_revenue, denominator_promo_revenue.size, 0, denominator_promo_revenue.size)

    var promo_revenue_percentage = (promo_revenue / total_revenue) * 100.0

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("total_revenue: ", total_revenue)
    print("promo_revenue: ", promo_revenue)
    print("promo_revenue_percentage: ", promo_revenue_percentage)
    


fn test_query_15() raises:
    # var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    # # Load and prepare the `lineitem` data
    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_suppkey_arr = df['l_suppkey'].to_numpy()
    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()

    # var l_suppkey = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)

    var l_suppkey = Float64Array("../Data/tpch_med/l_suppkey_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")

    # for i in range(17996609):
    #     l_suppkey[i] = l_suppkey_arr[i].to_float64()
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
    #     l_discount[i] = l_discount_arr[i].to_float64()
    #     l_shipdate[i] = l_shipdate_arr[i].to_float64()
        
    print(l_shipdate.size)

    # Prepare `lineitem` columns
    var col_data = List[Float64Array](l_suppkey, l_extendedprice, l_discount, l_shipdate)

    var col_names = List[String]("suppkey", "l_extendedprice", "l_discount", "l_shipdate")

    var df_lineitem = DataFrameF64(col_data, col_names)

    # Load and prepare the `supplier` data
    # var file_path_supplier = 'supplier.csv'
    # var df_supp = pd.read_csv(file_path_supplier)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # var s_name_arr = df_supp['s_name'].to_numpy()

    # var s_suppkey = Float64Array(30000)
    # var s_name = Float64Array(30000)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_name = Float64Array("../Data/tpch_med/s_name_tensor")

    # for i in range(30000):
    #     s_suppkey[i] = s_suppkey_arr[i].to_float64()
    #     s_name[i] = s_name_arr[i].to_float64()

    # Prepare `supplier` columns
    var supplier_col_data = List[Float64Array](s_suppkey, s_name)

    var supplier_col_names = List[String]("suppkey", "s_name")

    var df_supplier = DataFrameF64(supplier_col_data, supplier_col_names)


    var start_time = monotonic()

    # Filter `lineitem` by `l_shipdate` between January 1, 1996, and April 1, 1996
    df_lineitem.select("l_shipdate", "l_shipdate", GTEPredF64(), LTPredF64(), 820454400.0, 828230400.0, "AND")

    var extendedprice = df_lineitem["l_extendedprice"]
    var discount = df_lineitem["l_discount"]

    # Calculate `total_revenue` as `l_extendedprice * (1 - l_discount)`
    var total_revenue = Float64Array(df_lineitem.columns[0].size)
    for i in range(total_revenue.size):
        total_revenue[i] = extendedprice[i] * (1 - discount[i])
    
    # Append `total_revenue` column to `lineitem` DataFrame
    df_lineitem.append_column(total_revenue^, "total_revenue")

    var joined_ls_df = inner_join_f64(df_lineitem, df_supplier, "suppkey")


    # Group by `l_suppkey` and calculate the sum of `total_revenue`
    var aggregated_col_names = List[String]("suppkey", "s_name", "l_extendedprice",
                                            "l_discount", "l_shipdate", "total_revenue")

    joined_ls_df.groupby_multicol(List[String]("suppkey", "s_name"), "sum", aggregated_col_names)
    
    print("joined_ls_df size after groupby: ", joined_ls_df.columns[0].size)

    # Find the maximum `total_revenue`
    var group_revenue = joined_ls_df["total_revenue"]
    var max_revenue = group_revenue[0]
    for i in range(1, group_revenue.size):
        if group_revenue[i] > max_revenue:
            max_revenue = group_revenue[i]

    # Filter `joined_df` where `total_revenue` equals `max_revenue`
    joined_ls_df.select("total_revenue", "total_revenue", EQPredF64(), EQPredF64(), max_revenue, max_revenue, "")

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("Execution time: ", execution_time_seconds)
    print("Max revenue: ", max_revenue)
    print("Result size: ", joined_ls_df.columns[0].size)

    for i in range(joined_ls_df.columns.size):
        if (joined_ls_df.column_names[i] == "suppkey"):
                print(joined_ls_df.column_names[i])
                print(joined_ls_df[joined_ls_df.col_name_to_idx["suppkey"]][0])
        if (joined_ls_df.column_names[i] == "s_name"):
                print(joined_ls_df.column_names[i])
                print(joined_ls_df[joined_ls_df.col_name_to_idx["s_name"]][0])
        if (joined_ls_df.column_names[i] == "total_revenue"):
                print(joined_ls_df.column_names[i])
                print(joined_ls_df[joined_ls_df.col_name_to_idx["total_revenue"]][0])




fn test_query_16() raises:
    var pd = Python.import_module("pandas")
    pd.set_option('display.max_columns', None)

    var file_path_supp = '../../data/tpch_3gb/supplier.csv'
    var df_supp = pd.read_csv(file_path_supp)
    print(df_supp.head())
    print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    var s_comment_arr = df_supp['s_comment']

    # var s_suppkey = Float64Array(30000)
    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_comment = List[String]()
    s_comment.resize(30000, "")

    for i in range(30000):
        # s_suppkey[i] = s_suppkey_arr[i].to_float64()
        s_comment[i] = s_comment_arr[i].__str__()

    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey)
  
    var supp_col_names = List[String]("suppkey")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)


    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    # var p_size_arr = df_pt['p_size'].to_numpy()
    var p_type_arr = df_pt['p_type']
    var p_brand_arr = df_pt['p_brand']

    # var p_partkey = Float64Array(600000)
    # var p_size = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_size = Float64Array("../Data/tpch_med/p_size_tensor")
    var p_type = List[String]()
    p_type.resize(600000, "")
    var p_brand = List[String]()
    p_brand.resize(600000, "")

    for i in range(600000):
        # p_partkey[i] = p_partkey_arr[i].to_float64()
        # p_size[i] = p_size_arr[i].to_float64()
        p_type[i] = p_type_arr[i].__str__()
        p_brand[i] = p_brand_arr[i].__str__()
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey, p_size)
    
    var part_col_names = List[String]("partkey", "p_size")

    var df_part = DataFrameF64(part_col_data, part_col_names)



    # var file_path_psupp = 'partsupp.csv'
    # var df_psupp = pd.read_csv(file_path_psupp)
    # print(df_psupp.head())
    # print(df_psupp.shape)

    # var ps_partkey_arr = df_psupp['ps_partkey'].to_numpy()
    # var ps_suppkey_arr = df_psupp['ps_suppkey'].to_numpy()

    # var ps_partkey = Float64Array(2400000)
    # var ps_suppkey = Float64Array(2400000)

    var ps_partkey = Float64Array("../Data/tpch_med/ps_partkey_tensor")
    var ps_suppkey = Float64Array("../Data/tpch_med/ps_suppkey_tensor")

    # for i in range(2400000):
    #     ps_partkey[i] = ps_partkey_arr[i].to_float64()
    #     ps_suppkey[i] = ps_suppkey_arr[i].to_float64()
    
    print(ps_partkey.size)

    var ps_col_data = List[Float64Array](ps_partkey, ps_suppkey)
    
    var ps_col_names = List[String]("partkey", "suppkey")

    var df_partsupp = DataFrameF64(ps_col_data, ps_col_names)

    var start_time = monotonic()

    # filter by p_brand and p_type and p_size first
    # filter out invalid suppliers, where s_comment has customer complaint
    # join tables, then perform groupby aggregation
    var p_brand_mask = filter_string_not_equal_mask(p_brand, "Brand#45")
    var p_type_mask = filter_string_not_startwith_mask(p_type, "MEDIUM POLISHED")
    var p_size_filter_list = Float64Array(8)
    p_size_filter_list[0] = 49.0
    p_size_filter_list[1] = 14.0
    p_size_filter_list[2] = 23.0
    p_size_filter_list[3] = 45.0
    p_size_filter_list[4] = 19.0
    p_size_filter_list[5] = 3.0
    p_size_filter_list[6] = 36.0
    p_size_filter_list[7] = 9.0
    var p_size_mask = filter_f64_IN_mask(df_part["p_size"], p_size_filter_list)

    var part_masks = List[List[Bool]]()
    
    part_masks.append(p_brand_mask)
    part_masks.append(p_type_mask)
    part_masks.append(p_size_mask)

    df_part.select_complex(part_masks, "AND")

    # filter out invalid suppliers with customer complaints
    filter_not_string_exists_before(df_supplier, s_comment, "Customer", "Complaints")

    var joined_pss_df = inner_join_f64(df_partsupp, df_supplier, "suppkey")
    var joined_psp_df = inner_join_f64(joined_pss_df, df_part, "partkey")


    var aggregated_col_names = List[String]("p_size", "suppkey")
    joined_psp_df.groupby("p_size", "count_distinct", aggregated_col_names)

    joined_psp_df.sort_by(List[String]("suppkey", "p_size"))


    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)

    for i in range(joined_psp_df.columns.size):
        if (joined_psp_df.column_names[i] == "p_size"):
                print(joined_psp_df.column_names[i])
                print(joined_psp_df[joined_psp_df.col_name_to_idx["p_size"]][joined_psp_df.columns[0].size - 1])
        if (joined_psp_df.column_names[i] == "suppkey"):
                print(joined_psp_df.column_names[i])
                print(joined_psp_df[joined_psp_df.col_name_to_idx["suppkey"]][joined_psp_df.columns[0].size - 1])
    print()

    for i in range(joined_psp_df.columns.size):
        if (joined_psp_df.column_names[i] == "p_size"):
                print(joined_psp_df.column_names[i])
                print(joined_psp_df[joined_psp_df.col_name_to_idx["p_size"]][joined_psp_df.columns[0].size - 2])
        if (joined_psp_df.column_names[i] == "suppkey"):
                print(joined_psp_df.column_names[i])
                print(joined_psp_df[joined_psp_df.col_name_to_idx["suppkey"]][joined_psp_df.columns[0].size - 2])
    print()

    for i in range(joined_psp_df.columns.size):
        if (joined_psp_df.column_names[i] == "p_size"):
                print(joined_psp_df.column_names[i])
                print(joined_psp_df[joined_psp_df.col_name_to_idx["p_size"]][joined_psp_df.columns[0].size - 3])
        if (joined_psp_df.column_names[i] == "suppkey"):
                print(joined_psp_df.column_names[i])
                print(joined_psp_df[joined_psp_df.col_name_to_idx["suppkey"]][joined_psp_df.columns[0].size - 3])
    

fn test_query_17() raises:
    var pd = Python.import_module("pandas")
    pd.set_option('display.max_columns', None)

    # var file_path_lineitem = 'lineitem-med.csv'
    # var df_line = pd.read_csv(file_path_lineitem)
    # print(df_line.head())
    # print(df_line.shape)

    # var l_partkey_arr = df_line['l_partkey'].to_numpy()
    # var l_quantity_arr = df_line['l_quantity'].to_numpy()
    # var l_extendedprice_arr = df_line['l_extendedprice'].to_numpy()

    # var l_partkey = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)
    # var l_extendedprice = Float64Array(17996609)
    var l_partkey = Float64Array("../Data/tpch_med/l_partkey_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")
    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")

    # for i in range(17996609):
    #     l_partkey[i] = l_partkey_arr[i].to_float64()
    #     l_quantity[i] = l_quantity_arr[i].to_float64()
    #     l_extendedprice[i] = l_extendedprice_arr[i].to_float64()

    print(l_partkey.size)

    var lineitem_col_data = List[Float64Array](l_partkey, l_quantity, l_extendedprice)

    var lineitem_col_names = List[String]("partkey", "l_quantity", "l_extendedprice")

    var df_lineitem = DataFrameF64(lineitem_col_data, lineitem_col_names)


    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    var p_brand_arr = df_pt['p_brand']
    var p_container_arr = df_pt['p_container']

    # var p_partkey = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_brand = List[String]()
    p_brand.resize(600000, "")
    var p_container = List[String]()
    p_container.resize(600000, "")

    for i in range(600000):
        # p_partkey[i] = p_partkey_arr[i].to_float64()
        p_brand[i] = p_brand_arr[i].__str__()
        p_container[i] = p_container_arr[i].__str__()

    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey)

    var part_col_names = List[String]("partkey")

    var df_part = DataFrameF64(part_col_data, part_col_names)

    var start_time = monotonic()

    # Filter `part` table by `p_brand` and `p_container`
    var p_brand_mask = filter_string_equal_mask(p_brand, "Brand#23")
    var p_container_mask = filter_string_equal_mask(p_container, "MED BOX")
    
    var masks = List[List[Bool]]()
    masks.append(p_brand_mask)
    masks.append(p_container_mask)
    df_part.select_complex(masks, "AND")

    var joined_lp_df = inner_join_f64(df_lineitem, df_part, "partkey")
    var lp_df_grouped = inner_join_f64(df_lineitem, df_part, "partkey")

    # get group avg for l_quantity, grouped by l_partkey
   
    # sum up extended_price and divide by 7

    var aggregated_col_names = List[String]("partkey", "l_quantity_avg", "l_extendedprice_avg")
    lp_df_grouped.groupby("partkey", "mean", aggregated_col_names)
    
    print("grouped size: ", lp_df_grouped.columns[0].size)

    var l_quantity_avg = lp_df_grouped["l_quantity_avg"]
    var scaled_avg = Float64Array(lp_df_grouped.columns[0].size)

    for i in range(scaled_avg.size):
        scaled_avg[i] = 0.2 * l_quantity_avg[i]
        
    lp_df_grouped.append_column(scaled_avg^, "l_quantity_avg_scaled")

    # join back with joined_lp_df
    # filter where l_quantity < l_quantity_average
    var joined_ll_df = inner_join_f64(joined_lp_df, lp_df_grouped, "partkey")
    joined_ll_df.select("l_quantity", "l_quantity_avg_scaled", LTPredF64(), LTPredF64(), 0.0, 0.0, "COL")

    var ll_df_extendedprice = joined_ll_df["l_extendedprice"]
    var avg_yearly = pairwise_sum_f64(ll_df_extendedprice, ll_df_extendedprice.size, 0, ll_df_extendedprice.size) / 7.0

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("joined size: ", joined_ll_df.columns[0].size)
    print("Yearly Average: ", avg_yearly)

fn test_query_18() raises:
    # var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_quantity_arr = df['l_quantity'].to_numpy()
    
    # var l_orderkey = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")

    # for i in range(17996609):
    #     l_orderkey[i] = l_orderkey_arr[i].to_float64()
    #     l_quantity[i] = l_quantity_arr[i].to_float64()

        
    print(l_orderkey.size)

    var col_data = List[Float64Array](l_orderkey, l_quantity)

    var col_names = List[String]("orderkey", "l_quantity")

    var df_lineitem = DataFrameF64(col_data, col_names)


    # var file_path_orders = '../../data/tpch_3gb/orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()
    # var o_orderdate_arr = df_ord['o_orderdate'].to_numpy()
    # var o_totalprice_arr = df_ord['o_totalprice']
    # var o_custkey_arr = df_ord['o_custkey'].to_numpy()

    # var o_orderkey = Float64Array(4500000)
    # var o_orderdate = Float64Array(4500000)
    # var o_totalprice = Float64Array(4500000)
    # var o_custkey = Float64Array(4500000)

    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    var o_orderdate = Float64Array("../Data/tpch_med/o_orderdate_tensor")
    var o_totalprice = Float64Array("../Data/tpch_med/o_totalprice_tensor")
    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")


    # for i in range(4500000):
        #  o_orderkey[i] = o_orderkey_arr[i].to_float64()
        # o_orderdate[i] = o_orderdate_arr[i].to_float64()
        # o_totalprice[i] = o_totalprice_arr[i].to_float64()
        # o_custkey[i] = o_custkey_arr[i].to_float64()
    
    print(o_totalprice.size)

    # o_totalprice.data.tofile(Path("../Data/tpch_med/o_totalprice_tensor"))

    var orders_col_data = List[Float64Array](o_orderkey, o_orderdate, o_totalprice, o_custkey)
    
    var orders_col_names = List[String]("orderkey", "o_orderdate", "o_totalprice", "custkey")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    # var file_path_customer = 'customer.csv'
    # var df_cust = pd.read_csv(file_path_customer)
    # print(df_cust.head())
    # print(df_cust.shape)

    # var c_custkey_arr = df_cust['c_custkey'].to_numpy()

    # var c_custkey = Float64Array(450000)
    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")

    # for i in range(450000):
    #     c_custkey[i] = c_custkey_arr[i].to_float64()
    
    print(c_custkey.size)

    var cust_col_data = List[Float64Array](c_custkey)
    
    var cust_col_names = List[String]("custkey")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)


    var start_time = monotonic()

    var joined_lo_df = inner_join_f64(df_lineitem, df_orders, "orderkey")

    # groupby l_orderkey, sum l_quantity, filter, then join back with the lineitem orders df
    # var group_start = perf_counter()
    var aggregated_col_names = List[String]("orderkey", "l_quantity_sum")
    df_lineitem.groupby("orderkey", "sum", aggregated_col_names)
    # var group_end = perf_counter()
    #print("groupby time: ", group_end - group_start)
    
    df_lineitem.select("l_quantity_sum", "l_quantity_sum", GTPredF64(), GTPredF64(), 300.0, 300.0, "")

    var filtered_lo_df = inner_join_f64(df_lineitem, joined_lo_df, "orderkey")

    var joined_oc_df = inner_join_f64(filtered_lo_df, df_customer, "custkey")


    var group_by_cols = List[String]("custkey", "orderkey", "o_orderdate", "o_totalprice")

    var aggregated_col_names2 = List[String]("custkey", "orderkey", "o_orderdate", "o_totalprice",
                                             "l_quantity_sum", "quantity_sum_grouped")

    joined_oc_df.groupby_multicol(group_by_cols, "sum", aggregated_col_names2)
    joined_oc_df.sort_by(List[String]("o_totalprice", "o_orderdate"))


    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("final grouped size: ", joined_oc_df.columns[0].size)
    

    for i in range(joined_oc_df.columns.size):
        if (joined_oc_df.column_names[i] == "custkey"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["custkey"]][joined_oc_df.columns[0].size - 1])
        if (joined_oc_df.column_names[i] == "orderkey"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["orderkey"]][joined_oc_df.columns[0].size - 1])
        if (joined_oc_df.column_names[i] == "o_orderdate"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["o_orderdate"]][joined_oc_df.columns[0].size - 1])
        if (joined_oc_df.column_names[i] == "o_totalprice"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["o_totalprice"]][joined_oc_df.columns[0].size - 1])
        if (joined_oc_df.column_names[i] == "quantity_sum_grouped"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["quantity_sum_grouped"]][joined_oc_df.columns[0].size - 1])
    print()

    for i in range(joined_oc_df.columns.size):
        if (joined_oc_df.column_names[i] == "custkey"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["custkey"]][joined_oc_df.columns[0].size - 2])
        if (joined_oc_df.column_names[i] == "orderkey"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["orderkey"]][joined_oc_df.columns[0].size - 2])
        if (joined_oc_df.column_names[i] == "o_orderdate"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["o_orderdate"]][joined_oc_df.columns[0].size - 2])
        if (joined_oc_df.column_names[i] == "o_totalprice"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["o_totalprice"]][joined_oc_df.columns[0].size - 2])
        if (joined_oc_df.column_names[i] == "quantity_sum_grouped"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["quantity_sum_grouped"]][joined_oc_df.columns[0].size - 2])
    print()

    for i in range(joined_oc_df.columns.size):
        if (joined_oc_df.column_names[i] == "custkey"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["custkey"]][joined_oc_df.columns[0].size - 3])
        if (joined_oc_df.column_names[i] == "orderkey"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["orderkey"]][joined_oc_df.columns[0].size - 3])
        if (joined_oc_df.column_names[i] == "o_orderdate"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["o_orderdate"]][joined_oc_df.columns[0].size - 3])
        if (joined_oc_df.column_names[i] == "o_totalprice"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["o_totalprice"]][joined_oc_df.columns[0].size - 3])
        if (joined_oc_df.column_names[i] == "quantity_sum_grouped"):
                print(joined_oc_df.column_names[i])
                print(joined_oc_df[joined_oc_df.col_name_to_idx["quantity_sum_grouped"]][joined_oc_df.columns[0].size - 3])



fn test_query_19() raises:
    var pd = Python.import_module("pandas")
    pd.set_option('display.max_columns', None)

    # predicate pushdown-> filter lineitem by shipmode and shipstruct first
    # join lineitem_filtered and part table on partkey-> return the indexer for part table
    # use the indexer for part table to create the re-indexed p_brand and p_container

    # get masks for p_brand, p_container, l_quantity, and p_size
    # use the AND to get comined mask for this first group of filters

    var file_path = '../../data/tpch_3gb/lineitem-med.csv'
    var df = pd.read_csv(file_path)
    print(df.head())
    print(df.shape)

    # var l_extendedprice_arr = df['l_extendedprice'].to_numpy()
    # var l_discount_arr = df['l_discount'].to_numpy()
    # var l_quantity_arr = df['l_quantity'].to_numpy()
    # var l_partkey_arr = df['l_partkey'].to_numpy()
    # var l_shipmode_arr = df['l_shipmode'].to_numpy()
    var l_shipinstruct_arr = df['l_shipinstruct']


    # var l_extendedprice = Float64Array(17996609)
    # var l_discount = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)
    # var l_partkey = Float64Array(17996609)
    # var l_shipmode = Float64Array(17996609)

    var l_extendedprice = Float64Array("../Data/tpch_med/l_extendedprice_tensor")
    var l_discount = Float64Array("../Data/tpch_med/l_discount_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")
    var l_partkey = Float64Array("../Data/tpch_med/l_partkey_tensor")
    var l_shipmode = Float64Array("../Data/tpch_med/l_shipmode_tensor")

    var l_shipinstruct = List[String]()
    l_shipinstruct.resize(17996609, "")

    for i in range(17996609):
        # l_extendedprice[i] = l_extendedprice_arr[i].to_float64()
        # l_discount[i] = l_discount_arr[i].to_float64()
        # l_quantity[i] = l_quantity_arr[i].to_float64()
        # l_partkey[i] = l_partkey_arr[i].to_float64()
        # l_shipmode[i] = l_shipmode_arr[i].to_float64()
        l_shipinstruct[i] = l_shipinstruct_arr[i].__str__()
        
    print(l_partkey.size)

    var col_data = List[Float64Array](l_extendedprice, l_discount, l_quantity, l_partkey, l_shipmode)

    var col_names = List[String]("l_extendedprice", "l_discount", "l_quantity", "partkey", "l_shipmode")

    var df_lineitem = DataFrameF64(col_data, col_names)

    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    # var p_size_arr = df_pt['p_size'].to_numpy()
    var p_brand_arr = df_pt['p_brand']
    var p_container_arr = df_pt['p_container']

    # var p_partkey = Float64Array(600000)
    # var p_size = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_size = Float64Array("../Data/tpch_med/p_size_tensor")
    var p_brand = List[String]()
    p_brand.resize(600000, "")
    var p_container = List[String]()
    p_container.resize(600000, "")

    for i in range(600000):
        # p_partkey[i] = p_partkey_arr[i].to_float64()
        # p_size[i] = p_size_arr[i].to_float64()
        p_brand[i] = p_brand_arr[i].__str__()
        p_container[i] = p_container_arr[i].__str__()
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey, p_size)
    
    var part_col_names = List[String]("partkey", "p_size")

    var df_part = DataFrameF64(part_col_data, part_col_names)

    var start_time = monotonic()

    var ship_mode_mask = df_lineitem.select_mask("l_shipmode", "l_shipmode", EQPredF64(), EQPredF64(), 2.0, 4.0, "OR")
    var shipinstruct_mask = filter_string_equal_mask(l_shipinstruct, "DELIVER IN PERSON")
    
    var ship_masks = List[List[Bool]](ship_mode_mask, shipinstruct_mask)

    df_lineitem.select_complex(ship_masks, "AND")

    #print("lineitem after pred pushdown: ", df_lineitem.columns[0].size)

    var joined_lp_df_with_indexers = inner_join_f64_reindex(df_lineitem, df_part, "partkey")
    var part_indexers = joined_lp_df_with_indexers.indexers[1]
    
    # reindex p_brand and p_container using the indexers returned from inner join
    

    var p_container_reindexed = reindex_string_column(p_container, part_indexers)
    var p_brand_reindexed = reindex_string_column(p_brand, part_indexers)
    
    var p_brand12_mask = filter_string_equal_mask(p_brand_reindexed, "Brand#12")
    var p_container_small_mask = filter_string_IN_mask(p_container_reindexed, List[String]("SM CASE", "SM BOX", "SM PACK", "SM PKG"))
    var l_quantity_mask1 = joined_lp_df_with_indexers.df.select_mask("l_quantity", "l_quantity", GTEPredF64(), LEPredF64(), 1.0, 11.0, "AND")
    var p_size_mask1 = joined_lp_df_with_indexers.df.select_mask("p_size", "p_size", GTEPredF64(), LEPredF64(), 1.0, 5.0, "AND")
    var combined_mask1 = combine_masks(List[List[Bool]](p_brand12_mask, p_container_small_mask, l_quantity_mask1, p_size_mask1), "AND")

    var p_brand23_mask = filter_string_equal_mask(p_brand_reindexed, "Brand#23")
    var p_container_med_mask = filter_string_IN_mask(p_container_reindexed, List[String]("MED BAG", "MED BOX", "MED PKG", "MED PACK"))
    var l_quantity_mask2 = joined_lp_df_with_indexers.df.select_mask("l_quantity", "l_quantity", GTEPredF64(), LEPredF64(), 10.0, 20.0, "AND")
    var p_size_mask2 = joined_lp_df_with_indexers.df.select_mask("p_size", "p_size", GTEPredF64(), LEPredF64(), 1.0, 10.0, "AND")
    var combined_mask2 = combine_masks(List[List[Bool]](p_brand23_mask, p_container_med_mask, l_quantity_mask2, p_size_mask2), "AND")

    var p_brand34_mask = filter_string_equal_mask(p_brand_reindexed, "Brand#34")
    var p_container_large_mask = filter_string_IN_mask(p_container_reindexed, List[String]("LG CASE", "LG BOX", "LG PACK", "LG PKG"))
    var l_quantity_mask3 = joined_lp_df_with_indexers.df.select_mask("l_quantity", "l_quantity", GTEPredF64(), LEPredF64(), 20.0, 30.0, "AND")
    var p_size_mask3 = joined_lp_df_with_indexers.df.select_mask("p_size", "p_size", GTEPredF64(), LEPredF64(), 1.0, 15.0, "AND")
    var combined_mask3 = combine_masks(List[List[Bool]](p_brand34_mask, p_container_large_mask, l_quantity_mask3, p_size_mask3), "AND")

    joined_lp_df_with_indexers.df.select_complex(List[List[Bool]](combined_mask1, combined_mask2, combined_mask3),"OR")


    var extended_price = joined_lp_df_with_indexers.df["l_extendedprice"]
    var discount = joined_lp_df_with_indexers.df["l_discount"]
    var disc_price = Float64Array(extended_price.size)

    for i in range(disc_price.size):
        disc_price[i] = (-1.0 * discount[i] + 1.0) * extended_price[i]

    var revenue = pairwise_sum_f64(disc_price, disc_price.size, 0, disc_price.size)

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("final joined size after all filter: ", joined_lp_df_with_indexers.df.columns[0].size)
    print("indexer size: ", joined_lp_df_with_indexers.indexers[1].size)
    print("REVENUE: ", revenue)


fn test_query_20() raises:
    var pd = Python.import_module("pandas")
    # pd.set_option('display.max_columns', None)

    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_partkey_arr = df['l_partkey'].to_numpy()
    # var l_quantity_arr = df['l_quantity'].to_numpy()
    # var l_suppkey_arr = df['l_suppkey'].to_numpy()
    # var l_shipdate_arr = df['l_shipdate'].to_numpy()


    # var l_partkey = Float64Array(17996609)
    # var l_quantity = Float64Array(17996609)
    # var l_suppkey = Float64Array(17996609)
    # var l_shipdate = Float64Array(17996609)
    var l_partkey = Float64Array("../Data/tpch_med/l_partkey_tensor")
    var l_quantity = Float64Array("../Data/tpch_med/l_quantity_tensor")
    var l_suppkey = Float64Array("../Data/tpch_med/l_suppkey_tensor")
    var l_shipdate = Float64Array("../Data/tpch_med/l_shipdate_tensor")
    # for i in range(17996609):
    #     l_partkey[i] = l_partkey_arr[i].to_float64()
    #     l_quantity[i] = l_quantity_arr[i].to_float64()
    #     l_suppkey[i] = l_suppkey_arr[i].to_float64()
    #     l_shipdate[i] = l_shipdate_arr[i].to_float64()
        
    print(l_partkey.size)

    var col_data = List[Float64Array](l_partkey, l_quantity, l_suppkey, l_shipdate)

    var col_names = List[String]("partkey", "l_quantity", "l_suppkey", "l_shipdate")

    var df_lineitem = DataFrameF64(col_data, col_names)


    var file_path_part = '../../data/tpch_3gb/part.csv'
    var df_pt = pd.read_csv(file_path_part)
    print(df_pt.head())
    print(df_pt.shape)

    # var p_partkey_arr = df_pt['p_partkey'].to_numpy()
    var p_name_arr = df_pt['p_name']

    # var p_partkey = Float64Array(600000)
    var p_partkey = Float64Array("../Data/tpch_med/p_partkey_tensor")
    var p_name = List[String]()
    p_name.resize(600000, "")

    for i in range(600000):
        # p_partkey[i] = p_partkey_arr[i].to_float64()
        p_name[i] = p_name_arr[i].__str__()
    
    print(p_partkey.size)

    var part_col_data = List[Float64Array](p_partkey)
    
    var part_col_names = List[String]("partkey")

    var df_part = DataFrameF64(part_col_data, part_col_names)


    # var file_path_supp = 'supplier.csv'
    # var df_supp = pd.read_csv(file_path_supp)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # var s_nationkey_arr = df_supp['s_nationkey'].to_numpy()
    # var s_name_arr = df_supp['s_name'].to_numpy()

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")
    var s_name = Float64Array("../Data/tpch_med/s_name_tensor")

    # var s_suppkey = Float64Array(30000)
    # var s_nationkey = Float64Array(30000)
    # var s_name = Float64Array(30000)

    # for i in range(30000):
    #     s_suppkey[i] = s_suppkey_arr[i].to_float64()
    #     s_nationkey[i] = s_nationkey_arr[i].to_float64()
    #     s_name[i] = s_name_arr[i].to_float64()
    
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey, s_name)
    
    var supp_col_names = List[String]("s_suppkey", "nationkey", "s_name")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)


    # var file_path_psupp = 'partsupp.csv'
    # var df_psupp = pd.read_csv(file_path_psupp)
    # print(df_psupp.head())
    # print(df_psupp.shape)

    # var ps_partkey_arr = df_psupp['ps_partkey'].to_numpy()
    # var ps_suppkey_arr = df_psupp['ps_suppkey'].to_numpy()
    # var ps_availqty_arr = df_psupp['ps_availqty'].to_numpy()

    var ps_partkey = Float64Array("../Data/tpch_med/ps_partkey_tensor")
    var ps_suppkey = Float64Array("../Data/tpch_med/ps_suppkey_tensor")
    var ps_availqty = Float64Array("../Data/tpch_med/ps_availqty_tensor")

    # var ps_partkey = Float64Array(2400000)
    # var ps_suppkey = Float64Array(2400000)
    # var ps_availqty = Float64Array(2400000)

    # for i in range(2400000):
    #     ps_partkey[i] = ps_partkey_arr[i].to_float64()
    #     ps_suppkey[i] = ps_suppkey_arr[i].to_float64()
    #     ps_availqty[i] = ps_availqty_arr[i].to_float64()
    
    print(ps_partkey.size)

    var ps_col_data = List[Float64Array](ps_partkey, ps_suppkey, ps_availqty)
    
    var ps_col_names = List[String]("partkey", "s_suppkey", "ps_availqty")

    var df_partsupp = DataFrameF64(ps_col_data, ps_col_names)


    # var file_path_nation = 'nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # var n_name_arr = df_nat['n_name'].to_numpy()

    # var n_nationkey = Float64Array(25)
    # var n_name = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # for i in range(25):
    #     n_nationkey[i] = n_nationkey_arr[i].to_float64()
    #     n_name[i] = n_name_arr[i].to_float64()
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name)
    
    var nation_col_names = List[String]("nationkey", "n_name")

    var df_nation = DataFrameF64(nation_col_data, nation_col_names)

    var start_time = monotonic()

    filter_string_startwith(df_part, p_name, "forest")
    df_lineitem.select("l_shipdate", "l_shipdate", GTEPredF64(), LTPredF64(), 757382400.0, 788918400.0, "AND")

    print("lineitem filtered size: ", df_lineitem.columns[0].size)

    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time after string filter: ", execution_time_seconds)


    df_nation.select("n_name", "n_name", EQPredF64(), EQPredF64(), 35480.0, 35480.0, "")

    # lineitem groupby sum
    df_lineitem.groupby_multicol(List[String]("partkey", "l_suppkey"), "sum", List[String]("partkey", "l_suppkey", "l_quantity_sum", "l_shipdate_sum"))

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time after lineitem groupby: ", execution_time_seconds)

    var joined_sn_df = inner_join_f64(df_supplier, df_nation, "nationkey")
    var joined_psp_df = inner_join_f64(df_partsupp, df_part, "partkey")
    var joined_pss_df = inner_join_f64(joined_psp_df, joined_sn_df, "s_suppkey")

    var joined_lpss_df = inner_join_f64(df_lineitem, joined_pss_df, "partkey")
    joined_lpss_df.select("l_suppkey", "s_suppkey", EQPredF64(), EQPredF64(), 0.0, 0.0, "COL")

    # add the column for 0.5 * sum(l_quantity)
    var l_quantity_sum = joined_lpss_df["l_quantity_sum"]
    var scaled_l_quantity_sum = Float64Array(joined_lpss_df.columns[0].size)

    for i in range(scaled_l_quantity_sum.size):
        scaled_l_quantity_sum[i] = l_quantity_sum[i] * 0.5
    
    joined_lpss_df.append_column(scaled_l_quantity_sum^, "scaled_l_quantity_sum")

    joined_lpss_df.select("ps_availqty", "scaled_l_quantity_sum", GTPredF64(), GTPredF64(), 0.0, 0.0, "COL")

    joined_lpss_df.sort_by(List[String]("s_name"))


    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("joined_lpss_df size: ", joined_lpss_df.columns[0].size)
    print()

    for i in range(joined_lpss_df.columns.size):
        if (joined_lpss_df.column_names[i] == "s_name"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["s_name"]][0])
        if (joined_lpss_df.column_names[i] == "partkey"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["partkey"]][0])
        if (joined_lpss_df.column_names[i] == "ps_availqty"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["ps_availqty"]][0])
            
    print()

    for i in range(joined_lpss_df.columns.size):
        if (joined_lpss_df.column_names[i] == "s_name"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["s_name"]][1])
        if (joined_lpss_df.column_names[i] == "partkey"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["partkey"]][1])
        if (joined_lpss_df.column_names[i] == "ps_availqty"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["ps_availqty"]][1])
    
    print()

    for i in range(joined_lpss_df.columns.size):
        if (joined_lpss_df.column_names[i] == "s_name"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["s_name"]][2])
        if (joined_lpss_df.column_names[i] == "partkey"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["partkey"]][2])
        if (joined_lpss_df.column_names[i] == "ps_availqty"):
                print(joined_lpss_df.column_names[i])
                print(joined_lpss_df[joined_lpss_df.col_name_to_idx["ps_availqty"]][2])




fn test_query_21() raises:
    # var pd = Python.import_module("pandas")
    # var np = Python.import_module("numpy")

    # pd.set_option('display.max_columns', None)
    # var file_path = 'lineitem-med.csv'
    # var df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.shape)

    # var l_orderkey_arr = df['l_orderkey'].to_numpy()
    # var l_suppkey_arr = df['l_suppkey'].to_numpy()
    # var l_commitdate_arr = df['l_commitdate'].to_numpy()
    # var l_receiptdate_arr = df['l_receiptdate'].to_numpy()

    # var l_orderkey = Float64Array(17996609)
    # var l_suppkey = Float64Array(17996609)
    # var l_commitdate = Float64Array(17996609)
    # var l_receiptdate = Float64Array(17996609)

    var l_orderkey = Float64Array("../Data/tpch_med/l_orderkey_tensor")
    var l_suppkey = Float64Array("../Data/tpch_med/l_suppkey_tensor")
    var l_commitdate = Float64Array("../Data/tpch_med/l_commitdate_tensor")
    var l_receiptdate = Float64Array("../Data/tpch_med/l_receiptdate_tensor")
    
    # for i in range(17996609):
    #     l_orderkey[i] = float(l_orderkey_arr[i])
    #     l_suppkey[i] = float(l_suppkey_arr[i])
    #     l_commitdate[i] = float(l_commitdate_arr[i])
    #     l_receiptdate[i] = float(l_receiptdate_arr[i])
    
    print(l_receiptdate.size)

    var col_data = List[Float64Array](l_orderkey, l_suppkey, l_commitdate, l_receiptdate)

    var col_names = List[String]("orderkey", "suppkey", "l_commitdate", "l_receiptdate")

    var df_lineitem = DataFrameF64(col_data, col_names)
    var df_lineitem_late = DataFrameF64(col_data, List[String]("orderkey", "suppkey_late", "l_commitdate", "l_receiptdate"))


    # var file_path_supp = 'supplier.csv'
    # var df_supp = pd.read_csv(file_path_supp)
    # print(df_supp.head())
    # print(df_supp.shape)

    # var s_suppkey_arr = df_supp['s_suppkey'].to_numpy()
    # var s_nationkey_arr = df_supp['s_nationkey'].to_numpy()
    # var s_name_arr = df_supp['s_name'].to_numpy()

    # var s_suppkey = Float64Array(30000)
    # var s_nationkey = Float64Array(30000)
    # var s_name = Float64Array(30000)

    var s_suppkey = Float64Array("../Data/tpch_med/s_suppkey_tensor")
    var s_nationkey = Float64Array("../Data/tpch_med/s_nationkey_tensor")
    var s_name = Float64Array("../Data/tpch_med/s_name_tensor")

    # for i in range(30000):
    #     s_suppkey[i] = float(s_suppkey_arr[i])
    #     s_nationkey[i] = float(s_nationkey_arr[i])
    #     s_name[i] = float(s_name_arr[i])
    
    print(s_suppkey.size)

    var supp_col_data = List[Float64Array](s_suppkey, s_nationkey, s_name)
    
    var supp_col_names = List[String]("suppkey", "nationkey", "s_name")

    var df_supplier = DataFrameF64(supp_col_data, supp_col_names)


    # var file_path_orders = '../../data/tpch_3gb/orders.csv'
    # var df_ord = pd.read_csv(file_path_orders)
    # print(df_ord.head())
    # print(df_ord.shape)

    # var o_orderstatus_arr = df_ord['o_orderstatus']
    # # var o_orderkey_arr = df_ord['o_orderkey'].to_numpy()

    # var o_orderstatus = Float64Array(4500000)
    # var o_orderkey = Float64Array(4500000)

    var o_orderstatus = Float64Array("../Data/tpch_med/o_orderstatus_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")
    

    # for i in range(4500000):
    #     o_orderstatus[i] = (o_orderstatus_arr[i].to_float64())
        #o_orderkey[i] = float(o_orderkey_arr[i])
    
    
    print(o_orderstatus.size)
    # o_orderstatus.data.tofile(Path("../Data/tpch_med/o_orderstatus_tensor"))

    var orders_col_data = List[Float64Array](o_orderstatus, o_orderkey)
    
    var orders_col_names = List[String]("o_orderstatus", "orderkey")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)


    # var file_path_nation = 'nation.csv'
    # var df_nat = pd.read_csv(file_path_nation)
    # print(df_nat.head())
    # print(df_nat.shape)

    # var n_nationkey_arr = df_nat['n_nationkey'].to_numpy()
    # var n_name_arr = df_nat['n_name'].to_numpy()

    # var n_nationkey = Float64Array(25)
    # var n_name = Float64Array(25)

    var n_nationkey = Float64Array("../Data/tpch_med/n_nationkey_tensor")
    var n_name = Float64Array("../Data/tpch_med/n_name_tensor")

    # # for i in range(25):
    # #     n_nationkey[i] = float(n_nationkey_arr[i])
    # #     n_name[i] = float(n_name_arr[i])
    
    print(n_nationkey.size)

    var nation_col_data = List[Float64Array](n_nationkey, n_name)
    
    var nation_col_names = List[String]("nationkey", "n_name")

    var df_nation = DataFrameF64(nation_col_data, nation_col_names)

    var start_time = monotonic()

    # o_orderstatus = 'F', 70 is the integer ASCII value of F
    df_orders.select("o_orderstatus", "o_orderstatus", EQPredF64(), EQPredF64(), 70.0, 70.0, "")

    var lineitem_orders_late = inner_join_f64(df_lineitem, df_orders, "orderkey")
    var suppliers_per_order = inner_join_f64(df_lineitem, df_orders, "orderkey")
    var late_suppliers_per_order = inner_join_f64(df_lineitem_late, df_orders, "orderkey")
    
    # late_lineitems
    lineitem_orders_late.select("l_receiptdate", "l_commitdate", GTPredF64(), GTPredF64(), 0.0, 0.0, "COL")
    late_suppliers_per_order.select("l_receiptdate", "l_commitdate", GTPredF64(), GTPredF64(), 0.0, 0.0, "COL")

    # compute the number of distinct suppliers per order
    suppliers_per_order.groupby("orderkey", "count_distinct", List[String]("orderkey", "suppkey"))
    suppliers_per_order.rename_column("suppkey", "num_total_suppliers")

    print("suppliers_per_order size:", suppliers_per_order.columns[0].size)

    # compute the number of distinct faulty suppliers per order
    late_suppliers_per_order.groupby("orderkey", "count_distinct", List[String]("orderkey", "suppkey_late"))
    print("late_suppliers_per_order size:", late_suppliers_per_order.columns[0].size)
    late_suppliers_per_order.rename_column("suppkey_late", "num_faulty_supplier")

    # filter orders with > 1 supplier and exactly 1 late supplier, after joining to get order stats
    var valid_orders = inner_join_f64(suppliers_per_order, late_suppliers_per_order, "orderkey")

    valid_orders.select("num_total_suppliers", "num_faulty_supplier", GTPredF64(), EQPredF64(), 1.0, 1.0, "AND")

    # nation SAUDI ARABIA
    df_nation.select("n_name", "n_name", EQPredF64(), EQPredF64(), 54189.0, 54189.0, "")

    var supplier_sa = inner_join_f64(df_nation, df_supplier, "nationkey")

    var late_lineitems_sa = inner_join_f64(lineitem_orders_late, supplier_sa, "suppkey")

    var valid_lineitems_sa = inner_join_f64(valid_orders, late_lineitems_sa, "orderkey")

    print("valid_lineitems_sa size:", valid_lineitems_sa.columns[0].size)

    # for i in range(valid_lineitems_sa.columns.size):
    #     print(valid_lineitems_sa.column_names[i])

    # compute number of late lineitems per order for each supplier
    valid_lineitems_sa.groupby_multicol(List[String]("s_name", "orderkey"), "count", List[String]("s_name", "orderkey", "count_per_order"))
    print("count per order size:", valid_lineitems_sa.columns[0].size)

    # # compute total number of late lineitems per supplier
    valid_lineitems_sa.groupby("s_name", "sum", List[String]("s_name", "orderkey", "numwait"))
    print("total num per supp size:", valid_lineitems_sa.columns[0].size)
    
    valid_lineitems_sa.sort_by(List[String]("numwait", "s_name"))


    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)

    for i in range(valid_lineitems_sa.columns.size):
        if (valid_lineitems_sa.column_names[i] == "s_name"):
            print(valid_lineitems_sa.column_names[i])
            print(valid_lineitems_sa[valid_lineitems_sa.col_name_to_idx["s_name"]][0])
        if (valid_lineitems_sa.column_names[i] == "numwait"):
            print(valid_lineitems_sa.column_names[i])
            print(valid_lineitems_sa[valid_lineitems_sa.col_name_to_idx["numwait"]][0])
    
    for i in range(valid_lineitems_sa.columns.size):
        if (valid_lineitems_sa.column_names[i] == "s_name"):
            print(valid_lineitems_sa.column_names[i])
            print(valid_lineitems_sa[valid_lineitems_sa.col_name_to_idx["s_name"]][valid_lineitems_sa.columns[0].size - 1])
        if (valid_lineitems_sa.column_names[i] == "numwait"):
            print(valid_lineitems_sa.column_names[i])
            print(valid_lineitems_sa[valid_lineitems_sa.col_name_to_idx["numwait"]][valid_lineitems_sa.columns[0].size - 1])


fn test_query_22() raises:
    var pd = Python.import_module("pandas")
    # var tm = Python.import_module("time")

    # print("data load start time: ", monotonic())

    var file_path_customer = '../../data/tpch_3gb/customer.csv'
    var df_cust = pd.read_csv(file_path_customer)

    var c_phone_arr = df_cust['c_phone']
    var c_phone = List[String]()
    c_phone.resize(450000, "")

    for i in range(450000):
        c_phone[i] = (c_phone_arr[i].__str__())
    
    var c_custkey = Float64Array("../Data/tpch_med/c_custkey_tensor")
    var c_nationkey = Float64Array("../Data/tpch_med/c_nationkey_tensor")
    var c_acctbal = Float64Array("../Data/tpch_med/c_acctbal_tensor")

    var cust_col_data = List[Float64Array](c_custkey, c_nationkey, c_acctbal)
    
    var cust_col_names = List[String]("custkey", "nationkey", "c_acctbal")

    var df_customer = DataFrameF64(cust_col_data, cust_col_names)
    var df_customer2 = DataFrameF64(cust_col_data, cust_col_names)


    var o_custkey = Float64Array("../Data/tpch_med/o_custkey_tensor")
    var o_orderkey = Float64Array("../Data/tpch_med/o_orderkey_tensor")

    var orders_col_data = List[Float64Array](o_custkey, o_orderkey)
    
    var orders_col_names = List[String]("custkey", "o_orderkey")

    var df_orders = DataFrameF64(orders_col_data, orders_col_names)

    print("data load end time: ", monotonic())

    var start_time = monotonic()

    var null_value = neg_inf[DType.float64]()
    var valid_country_code = Float64Array(7)
    valid_country_code[0] = 13.0
    valid_country_code[1] = 31.0
    valid_country_code[2] = 23.0
    valid_country_code[3] = 29.0
    valid_country_code[4] = 30.0
    valid_country_code[5] = 18.0
    valid_country_code[6] = 17.0

    var country_code = cast_as_float64(c_phone, 0, 2)

    var valid_country_code_mask = filter_f64_IN_mask(country_code, valid_country_code)
    var acctbal_mask = df_customer2.select_mask("c_acctbal", "c_acctbal", GTPredF64(), GTPredF64(), 0.0, 0.0, "")

    df_customer2.select_complex(List[List[Bool]](acctbal_mask, valid_country_code_mask), "AND")

    print("df_customer2 size: ", df_customer2.columns[0].size)

    var filtered_df_size = df_customer2.columns[0].size

    # compute average c_acctbal for customers whose phone prefix is in valid_country_code and c_acctbal > 0
    var acctbal_col = df_customer2["c_acctbal"]
    var avg_c_acctbal = pairwise_sum_f64(acctbal_col, filtered_df_size, 0, filtered_df_size) / filtered_df_size

    print("avg_c_acctbal: ", avg_c_acctbal)

    # add country_code column to df_customer
    df_customer.append_column(country_code^, "cntrycode")

    # filter df_customer by country code and acctbal > avg_c_acctbal
    var acctbal_avg_mask = df_customer.select_mask("c_acctbal", "c_acctbal", GTPredF64(), GTPredF64(), avg_c_acctbal, avg_c_acctbal, "")

    # df_customer.select("c_acctbal", "c_acctbal", GTPredF64(), GTPredF64(), avg_c_acctbal, avg_c_acctbal, "")

    df_customer.select_complex(List[List[Bool]](acctbal_avg_mask, valid_country_code_mask), "AND")

    print("df_customer size: ", df_customer.columns[0].size)


    var joined_co_df = left_join_f64(df_customer, df_orders, "custkey")

    print("joined df size:", joined_co_df.columns[0].size)

    # keep rows where there are no matching orders
    joined_co_df.select("o_orderkey", "o_orderkey", EQPredF64(), EQPredF64(), null_value, null_value, "")

    print("no orders df size:", joined_co_df.columns[0].size)

    joined_co_df.groupby("cntrycode", "all", List[String]("cntrycode", "custkey_sum", "nationkey_sum", "c_acctbal_sum",
                                                          "o_orderkey_sum", "custkey_avg", "nationkey_avg", "c_acctbal_avg",
                                                          "o_orderkey_avg", "numcust"))
    
    joined_co_df.sort_by(List[String]("cntrycode"))

    
    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("exec time: ", execution_time_seconds)
    print("final shape:", joined_co_df.columns[0].size)


    for i in range(joined_co_df.columns.size):
        if (joined_co_df.column_names[i] == "cntrycode"):
            print(joined_co_df.column_names[i])
            print(joined_co_df[joined_co_df.col_name_to_idx["cntrycode"]][0])
        if (joined_co_df.column_names[i] == "c_acctbal_sum"):
            print(joined_co_df.column_names[i])
            print(joined_co_df[joined_co_df.col_name_to_idx["c_acctbal_sum"]][0])
        if (joined_co_df.column_names[i] == "numcust"):
            print(joined_co_df.column_names[i])
            print(joined_co_df[joined_co_df.col_name_to_idx["numcust"]][0])
    
    print()

    for i in range(joined_co_df.columns.size):
        if (joined_co_df.column_names[i] == "cntrycode"):
            print(joined_co_df.column_names[i])
            print(joined_co_df[joined_co_df.col_name_to_idx["cntrycode"]][joined_co_df.columns[0].size - 1])
        if (joined_co_df.column_names[i] == "c_acctbal_sum"):
            print(joined_co_df.column_names[i])
            print(joined_co_df[joined_co_df.col_name_to_idx["c_acctbal_sum"]][joined_co_df.columns[0].size - 1])
        if (joined_co_df.column_names[i] == "numcust"):
            print(joined_co_df.column_names[i])
            print(joined_co_df[joined_co_df.col_name_to_idx["numcust"]][joined_co_df.columns[0].size - 1])

