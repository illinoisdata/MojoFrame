from core.DataFrame import DataFrameF64, DataFrameF32, DataFrameI32, SetElement
from collections.dict import Dict, KeyElement
from collections import Set
from utils.index import Index
from tensor import Tensor
from math import isclose
from utils.numerics import neg_inf
from time import monotonic, perf_counter
from algorithm import parallelize, vectorize
from utils.lock import BlockingSpinLock
from math import math
from sys.info import simdwidthof
from core.dict import CompactDict
from core.keys_container import KeysBuilder, KeyRef, Keyable
from hashlib.hash import _hash_simd

fn array_max_f64(read arr: Float64Array) raises -> SIMD[DType.float64, 1]:
    var cur_max = arr[0]

    for i in range(1, arr.size):
        if arr[i] > cur_max:
            cur_max = arr[i]

    return cur_max

fn element_mult_f64(mut arr1: Float64Array, mut arr2: Float64Array) raises -> Float64Array:
    var unroll_factor = 8
    var length = arr1.size
    var remainder = length % unroll_factor
    var result_arr = Float64Array(length)

    for i in range(0, length - remainder, unroll_factor):
        result_arr[i] = arr1[i] * arr2[i]
        result_arr[i+1] = arr1[i+1] * arr2[i+1]
        result_arr[i+2] = arr1[i+2] * arr2[i+2]
        result_arr[i+3] = arr1[i+3] * arr2[i+3]
        result_arr[i+4] = arr1[i+4] * arr2[i+4]
        result_arr[i+5] = arr1[i+5] * arr2[i+5]
        result_arr[i+6] = arr1[i+6] * arr2[i+6]
        result_arr[i+7] = arr1[i+7] * arr2[i+7]
    
    # process remainder of the array
    for i in range(length - remainder, length):
        result_arr[i] = arr1[i] * arr2[i]
    
    return result_arr

fn pairwise_sum_f64(mut arr: Float64Array, n: Int, start: Int, stop: Int) -> SIMD[DType.float64, 1]:
    if n < 8:
        var res = SIMD[DType.float64, 1](0)
        for i in range(start, stop):
            res += arr[i]
        return res
    
    elif n <= 128:
        var r0 = arr[start+0]
        var r1 = arr[start+1]
        var r2 = arr[start+2]
        var r3 = arr[start+3]
        var r4 = arr[start+4]
        var r5 = arr[start+5]
        var r6 = arr[start+6]
        var r7 = arr[start+7]

        var m = stop - (stop % 8)
        for i in range(start + 8, m, 8):
            r0 += arr[i]
            r1 += arr[i+1]
            r2 += arr[i+2]
            r3 += arr[i+3]
            r4 += arr[i+4]
            r5 += arr[i+5]
            r6 += arr[i+6]
            r7 += arr[i+7]
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

        for i in range(m, stop):
            res += arr[i]

        return res
    else:
        var n2 = n // 2
        n2 -= (n2 % 8)
        var middle = start + n2
        return (pairwise_sum_f64(arr, n2, start, middle)
                + pairwise_sum_f64(arr, n - n2, middle, stop))
                

fn pairwise_sum_f32(mut arr: Float32Array, n: Int, start: Int, stop: Int) -> SIMD[DType.float32, 1]:
    if n < 8:
        var res = SIMD[DType.float32, 1](0)
        for i in range(start, stop):
            res += arr[i]
        return res
    
    elif n <= 128:
        var r0 = arr[start+0]
        var r1 = arr[start+1]
        var r2 = arr[start+2]
        var r3 = arr[start+3]
        var r4 = arr[start+4]
        var r5 = arr[start+5]
        var r6 = arr[start+6]
        var r7 = arr[start+7]

        var m = stop - (stop % 8)
        for i in range(start + 8, m, 8):
            r0 += arr[i]
            r1 += arr[i+1]
            r2 += arr[i+2]
            r3 += arr[i+3]
            r4 += arr[i+4]
            r5 += arr[i+5]
            r6 += arr[i+6]
            r7 += arr[i+7]
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

        for i in range(m, stop):
            res += arr[i]

        return res
    else:
        var n2 = n // 2
        n2 -= (n2 % 8)
        var middle = start + n2
        return (pairwise_sum_f32(arr, n2, start, middle)
                + pairwise_sum_f32(arr, n - n2, middle, stop))

fn pairwise_sum_i32(mut arr: Int32Array, n: Int, start: Int, stop: Int) -> SIMD[DType.int32, 1]:
    if n < 8:
        var res = SIMD[DType.int32, 1](0)
        for i in range(start, stop):
            res += arr[i]
        return res
    
    elif n <= 128:
        var r0 = arr[start+0]
        var r1 = arr[start+1]
        var r2 = arr[start+2]
        var r3 = arr[start+3]
        var r4 = arr[start+4]
        var r5 = arr[start+5]
        var r6 = arr[start+6]
        var r7 = arr[start+7]

        var m = stop - (stop % 8)
        for i in range(start + 8, m, 8):
            r0 += arr[i]
            r1 += arr[i+1]
            r2 += arr[i+2]
            r3 += arr[i+3]
            r4 += arr[i+4]
            r5 += arr[i+5]
            r6 += arr[i+6]
            r7 += arr[i+7]
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

        for i in range(m, stop):
            res += arr[i]

        return res
    else:
        var n2 = n // 2
        n2 -= (n2 % 8)
        var middle = start + n2
        return (pairwise_sum_i32(arr, n2, start, middle)
                + pairwise_sum_i32(arr, n - n2, middle, stop))

fn column_wise_mult_f64(mut arr1: Float64Array, mut arr2: Float64Array) raises -> Float64Array:
    var result_arr = Float64Array(arr1.size)
    for i in range(arr1.size):
        result_arr[i] = (arr1[i] * arr2[i])
    return result_arr

fn aggregation_sum_i32(mut columns: List[Int32Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Int32Array]:
    # Iterate through each row of the DF (stored in columnar format)
    # For the groupby column, group all the rows by the different keys

    # The groupby sum table in Pandas is a 2d array
    # Dict that maps group to sums takes 14s to run for 10M records
    # var groupby_table = Dict[Int, Int32Array]()
    var start_time = monotonic()

    var groupby_table = List[Int32Array]()
    var groups_vec = List[IntKey]()
    var num_cols = columns.size
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[IntKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = IntKey(group_by_col[row_i])
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0

    # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        groupby_table.append(Int32Array(num_cols - 1))

    # for row_i in range(num_rows):
    #     # Each row in the groupby column belongs to a group
    #     var agg_i = 0
    #     var cur_row_group = (group_by_col[row_i])
    #     if groupby_table.__contains__(cur_row_group):
    #         # For each row/group, iterate through the columns at this row index
    #         for col_i in range(num_cols):
    #             if col_i != groupby_col_idx:
    #                 var cur_col = columns[col_i]
    #                 groupby_table[cur_row_group][agg_i] += cur_col[row_i]
    #                 agg_i += 1
    #     else:
    #         groups_vec.append(cur_row_group)
    #         groupby_table[cur_row_group] = Int32Array(num_cols)
    #         for col_i in range(num_cols):
    #             if col_i != groupby_col_idx:
    #                 var cur_col = columns[col_i]
    #                 groupby_table[cur_row_group][agg_i] = cur_col[row_i]
    #                 agg_i += 1
    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[IntKey(group_by_col[row_i])]
                # if groupby_table.__contains__(cur_row_group):
                #     groupby_table[cur_row_group][agg_i] += cur_col[row_i]
                # else:
                #     groups_vec.append(cur_row_group)
                #     groupby_table[cur_row_group] = Int32Array(num_cols - 1)
                #     groupby_table[cur_row_group][agg_i] = cur_col[row_i]
                groupby_table[cur_row_group_idx][agg_i] += cur_col[row_i]
            agg_i += 1
    var end_time = monotonic()
    print((end_time - start_time) / 1000000000)
    # After building the groupby dict, return the result as a DF
    # var summed_data = List[Int32Array]()
    # var group_vec_size = groups_vec.size
    # for i in range(group_vec_size):
    #     summed_data.append(groupby_table[groups_vec[i]])
    
    return groupby_table

# fn aggregation_sum_i32_alt(mut columns: List[Int32Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Int32Array]:
#     # Dict that stores group and their number of records
#     # For each group, map it to a vector of arrays
#     # For each row/group, store all the elements across columns into the vector of arrays
#     # Then for each group, for each vector of arrays, sum them
#     #var groupby_table = Dict[Int, List[Int32Array]]()
#     var group_to_size = Dict[IntKey, Int]()
#     var groups_vec = List[SIMD[DType.int32, 1]]()
#     var group_to_idx = Dict[IntKey, Int]()
   

#     var latest_group_index = Dict[IntKey, Int]()

#     var num_cols = columns.size
#     var num_rows = columns[groupby_col_idx].size
    

#     var group_by_col = columns[groupby_col_idx]
#     var max_group_size = 0
#     for row_i in range(num_rows):
#         # Each row in the groupby column belongs to a group
#         var cur_row_group = group_by_col[row_i] 
#         if group_to_size.__contains__(cur_row_group):
#             group_to_size[cur_row_group] += 1
#             var gts = group_to_size[cur_row_group]
#             if gts > max_group_size:
#                 max_group_size = gts
#         else:
#             groups_vec.append(cur_row_group)
#             latest_group_index[cur_row_group] = 0
#             group_to_size[cur_row_group] = 1

#     for i in range(groups_vec.size):
#         group_to_idx[groups_vec[i]] = i
#     # print(groups_vec.size)

#     # Tensor -> N x C columns, R rows
#     var groupby_table = Tensor[DType.int32](groups_vec.size * (num_cols-1), max_group_size)

    
#     var col_idx = 0
#     for col in range(num_cols):
#         if col != groupby_col_idx:
#             var cur_col = columns[col]
#             for ele_i in range(num_rows):
#                 var cur_group = group_by_col[ele_i]
#                 # group 5 maps to index 0, each group has 2 columns
#                 groupby_table[group_to_idx[cur_group] * (num_cols-1) + col_idx][latest_group_index[cur_group]] = cur_col[ele_i]
#                 #groupby_table[Index(group_to_idx[cur_group] * (num_cols-1) + col_idx, latest_group_index[cur_group])] = cur_col[ele_i]
#                 latest_group_index[cur_group] += 1
#                 if (latest_group_index[cur_group]) >= max_group_size:
#                     latest_group_index[cur_group] = 0
#                 #print(group_to_idx[group_by_col[ele_i]] * (num_cols-1) + col_idx)
#             col_idx += 1
        
    #print(groupby_table)
    #print(groupby_table.shape())
    # for i in range(groups_vec.size * (num_cols - 1)):
    #     var grp_sum = groupby_table.simd_load[4](i*max_group_size).reduce_add[1]()
    #     print(grp_sum)
    # for i in range(groups_vec.size):
    #     var cur_group = groups_vec[i]
    #     groupby_table[cur_group] = List[Int32Array]()
    #     # col index without groupby col
    #     var col_idx = 0
    #     for col in range(num_cols):
    #         if col == groupby_col_idx:
    #             continue
    #         # After calculating how many rows of data belong to each group, allocate memory for each group
    #         # Each group should contain a vector of Arrays
    #         var group_size = group_to_size[cur_group]
    #         var alloc_array = Int32Array(group_size)
    #         groupby_table[cur_group].append(alloc_array)

    #         # var group_row_idx = 0
    #         # var cur_col = columns[col]
    #         # for ele_i in range(num_rows):
    #         #     if group_by_col[ele_i] == cur_group:
    #         #         groupby_table[group_by_col[ele_i]][col_idx].append(cur_col[ele_i])
    #         #         group_row_idx += 1

    #         col_idx += 1

    # var agg_i = 0
    # for col_i in range(num_cols):
    #     if col_i != groupby_col_idx:
    #         var cur_col = columns[col_i]
    #         for row_i in range(num_rows):
    #             var cur_row_group = group_by_col[row_i]
    #             var latest_group_idx = latest_group_index[cur_row_group]
    #             #print(cur_row_group, agg_i, row_i)
    #             #groupby_table[cur_row_group][agg_i][latest_group_idx] = 1
    #             #print(latest_group_index[cur_row_group])
    #             latest_group_index[cur_row_group] += 1
    #         agg_i += 1


    # return columns

fn aggregation_sum_f64_multicol(mut df: DataFrameF64, groupby_cols: List[String]) raises -> List[Float64Array]:
    if groupby_cols.size == 2:
        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[DoubleTupleKey, Int]()
        #var key_to_index = Dict[TupleKey, List[Int]]()
        var groups_vec = List[DoubleTupleKey]()
        var num_cols = df.columns.size
        # append the columns of keys into a flat list
        # then create compound keys
        # [1,2,3,8001,8002,8003,121,122,123]
        
        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

        for i in range(groupby_cols.size):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True

        # need to map tuple to list of rows
        # tuple -> [0,4,7,8]
        # later iterate dict keys for aggregation
    
        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            # record row index for the corresponding key
            if not(compound_key in group_to_idx):
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                group_to_idx[compound_key] = 0
                #key_to_index[compound_key] = List[Int](i)

        for i in range(groups_vec.size):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            key_to_index[i] = group_to_idx[compound_key]

        for _ in range(num_cols - groupby_cols.size):
            groupby_table.append(Float64Array(groups_vec.size))

        ######## right now have groups to summed up values for each column
        ######## need list of columns by N groups
        var agg_i = 0
        # for every col, there are rows belonging to a group, aggregate values for groups
        for col_i in range(num_cols):
            if not (df.column_names[col_i] in groupby_cols_dict):
                var cur_col = (df.columns[col_i])
                for row_i in range(key_col_len):
                    groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
                agg_i += 1
        
        var group_col1 = Float64Array(groups_vec.size)
        var group_col2 = Float64Array(groups_vec.size)
        
        for key_i in range(groups_vec.size):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]

        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        return groupby_table

    elif groupby_cols.size == 3:

        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[TupleKey, Int]()
        #var key_to_index = Dict[TupleKey, List[Int]]()
        var groups_vec = List[TupleKey]()
        var num_cols = df.columns.size
        # append the columns of keys into a flat list
        # then create compound keys
        # [1,2,3,8001,8002,8003,121,122,123]
        
        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

        for i in range(groupby_cols.size):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True

        # need to map tuple to list of rows
        # tuple -> [0,4,7,8]
        # later iterate dict keys for aggregation
    
        for i in range(key_col_len):
            var compound_key = TupleKey(TripleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)])))
            # record row index for the corresponding key

            if not (compound_key in group_to_idx):
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                group_to_idx[compound_key] = 0
                #key_to_index[compound_key] = List[Int](i)

        for i in range(groups_vec.size):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = TupleKey(TripleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)])))
            key_to_index[i] = group_to_idx[compound_key]

        for _ in range(num_cols - groupby_cols.size):
            groupby_table.append(Float64Array(groups_vec.size))

        ######## right now have groups to summed up values for each column
        ######## need list of columns by N groups
        var agg_i = 0
        # for every col, there are rows belonging to a group, aggregate values for groups
        for col_i in range(num_cols):
            if not groupby_cols_dict.__contains__(df.column_names[col_i]):
                var cur_col = (df.columns[col_i])
                # if there is a compound key, there is a row where the values need to be aggregated
                # for key_i in range(groups_vec.size):
                #     var cur_key = groups_vec[key_i]
                #     var group_index = group_to_idx[cur_key]
                #     var cur_group_rows = key_to_index[cur_key]
                #     for row_i in range (cur_group_rows.size):
                #         groupby_table[agg_i][group_index] += (cur_col[cur_group_rows[row_i]])
                for row_i in range(key_col_len):
                    groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
                agg_i += 1
        
        var group_col1 = Float64Array(groups_vec.size)
        var group_col2 = Float64Array(groups_vec.size)
        var group_col3 = Float64Array(groups_vec.size)
        for key_i in range(groups_vec.size):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]
            group_col3[key_i] = cur_key.i.data[2]

        groupby_table.insert(0, group_col3)
        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)
        # print("groups vec:", groups_vec.size)
        # print("group 0 summed vals:")
        # for i in range(groupby_table[0].size):
        #     print(groupby_table[0][i])

        return groupby_table
    
    elif groupby_cols.size == 4:
        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[QuadTupleKey, Int]()
        #var key_to_index = Dict[TupleKey, List[Int]]()
        var groups_vec = List[QuadTupleKey]()
        var num_cols = df.columns.size
        # append the columns of keys into a flat list
        # then create compound keys
        # [1,2,3,8001,8002,8003,121,122,123]
        
        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

        for i in range(groupby_cols.size):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True

        # need to map tuple to list of rows
        # tuple -> [0,4,7,8]
        # later iterate dict keys for aggregation
    
        for i in range(key_col_len):
            var compound_key = QuadTupleKey(QuadTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)], flat_keys[i+(key_col_len*3)])))
            # record row index for the corresponding key
            if not (compound_key in group_to_idx):
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                group_to_idx[compound_key] = 0
                #key_to_index[compound_key] = List[Int](i)

        for i in range(groups_vec.size):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = QuadTupleKey(QuadTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)], flat_keys[i+(key_col_len*3)])))
            key_to_index[i] = group_to_idx[compound_key]

        for _ in range(num_cols - groupby_cols.size):
            groupby_table.append(Float64Array(groups_vec.size))

        ######## right now have groups to summed up values for each column
        ######## need list of columns by N groups
        var agg_i = 0
        # for every col, there are rows belonging to a group, aggregate values for groups
        for col_i in range(num_cols):
            if not (df.column_names[col_i] in groupby_cols_dict):
                var cur_col = (df.columns[col_i])
                # if there is a compound key, there is a row where the values need to be aggregated
                # for key_i in range(groups_vec.size):
                #     var cur_key = groups_vec[key_i]
                #     var group_index = group_to_idx[cur_key]
                #     var cur_group_rows = key_to_index[cur_key]
                #     for row_i in range (cur_group_rows.size):
                #         groupby_table[agg_i][group_index] += (cur_col[cur_group_rows[row_i]])
                for row_i in range(key_col_len):
                    groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
                agg_i += 1
        
        var group_col1 = Float64Array(groups_vec.size)
        var group_col2 = Float64Array(groups_vec.size)
        var group_col3 = Float64Array(groups_vec.size)
        var group_col4 = Float64Array(groups_vec.size)
        for key_i in range(groups_vec.size):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]
            group_col3[key_i] = cur_key.i.data[2]
            group_col4[key_i] = cur_key.i.data[3]

        groupby_table.insert(0, group_col4)
        groupby_table.insert(0, group_col3)
        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        return groupby_table

    return List[Float64Array]()


fn aggregation_count_f64_multicol(mut df: DataFrameF64, groupby_cols: List[String]) raises -> List[Float64Array]:
    if groupby_cols.size == 2:
        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[DoubleTupleKey, Int]()
        #var key_to_index = Dict[TupleKey, List[Int]]()
        var groups_vec = List[DoubleTupleKey]()
        var num_rows = df.columns[0].size
        # append the columns of keys into a flat list
        # then create compound keys
        # [1,2,3,8001,8002,8003,121,122,123]
        
        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

        for i in range(groupby_cols.size):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True

        # need to map tuple to list of rows
        # tuple -> [0,4,7,8]
        # later iterate dict keys for aggregation
    
        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            # record row index for the corresponding key
            if not(compound_key in group_to_idx):
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                group_to_idx[compound_key] = 0
                #key_to_index[compound_key] = List[Int](i)

        for i in range(groups_vec.size):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            key_to_index[i] = group_to_idx[compound_key]

        groupby_table.append(Float64Array(groups_vec.size))

        for row_i in range(num_rows):
            var cur_row_group_idx = key_to_index[row_i]
            groupby_table[0][cur_row_group_idx] += 1

        
        var group_col1 = Float64Array(groups_vec.size)
        var group_col2 = Float64Array(groups_vec.size)
        
        for key_i in range(groups_vec.size):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]

        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        return groupby_table
    
    return List[Float64Array]()


fn aggregation_all_f64_multicol(mut df: DataFrameF64, groupby_cols: List[String]) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var group_to_idx = Dict[DoubleTupleKey, Int]()
    # var key_to_index = Dict[DoubleTupleKey, List[Int]]()
   
    var groups_vec = List[DoubleTupleKey]()
    var num_cols_df = df.columns.size
    # append the columns of keys into a flat list
    # then create compound keys
    # [1,2,3,8001,8002,8003,121,122,123]
    var num_cols = (df.columns.size - groupby_cols.size) * 2 + 1
    
    var groupby_cols_dict = Dict[String, Bool]()
    var flat_keys = List[SIMD[DType.float64, 1]]()
    var key_col_len = df[0].size
    
    # use to keep track of which group the current row belongs to
    var key_to_index = List[Int]()
    key_to_index.resize(key_col_len, 0)

    for i in range(groupby_cols.size):
        var col = df[groupby_cols[i]]
        for row_i in range(key_col_len):
            flat_keys.append(col[row_i])
        groupby_cols_dict[groupby_cols[i]] = True

   
    for i in range(key_col_len):
        var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
        # record row index for the corresponding key
        if not (compound_key in group_to_idx):
            groups_vec.append(compound_key)
            # create the groups for the aggregated result
            group_to_idx[compound_key] = 0
            # key_to_index[compound_key] = List[Int](i)
    print("num of double keys: ", groups_vec.size)
    
    # use to keep track of which group the current row belongs to
    var key_to_count = List[Int]()
    key_to_count.resize(groups_vec.size, 0)

    for i in range(groups_vec.size):
        # map a compound key to an integer group
        group_to_idx[groups_vec[i]] = i

    for i in range(key_col_len):
        var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
        key_to_index[i] = group_to_idx[compound_key]
        key_to_count[group_to_idx[compound_key]] += 1

    for _ in range(num_cols):
        groupby_table.append(Float64Array(groups_vec.size))

    print("created all cols for sums and avgs")
    
    # for each column, sum each row to the corresponding group
    var agg_i = 0
    # for every col, there are rows belonging to a group, aggregate values for groups
    for col_i in range(num_cols_df):
        if not groupby_cols_dict.__contains__(df.column_names[col_i]):
            var cur_col = (df.columns[col_i])
            # if there is a compound key, there is a row where the values need to be aggregated
            for row_i in range(key_col_len):
                groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
            
            for group in range(groups_vec.size):
                groupby_table[agg_i + num_cols_df - groupby_cols.size][group] = (groupby_table[agg_i][group]) / (key_to_count[group])
            
            agg_i += 1

    var group_col1 = Float64Array(groups_vec.size)
    var group_col2 = Float64Array(groups_vec.size)
    # record count for each group
    for group_index in range(groups_vec.size):
        var cur_key = groups_vec[group_index]
        group_col1[group_index] = cur_key.i.data[0]
        group_col2[group_index] = cur_key.i.data[1]

        groupby_table[groupby_table.size - 1][group_index] = key_to_count[group_index]

    groupby_table.insert(0, group_col2)
    groupby_table.insert(0, group_col1)
    # print("groups vec:", groups_vec.size)

    return groupby_table

fn aggregation_sum_f64_parallel(mut columns: List[Float64Array],
                                col_names: List[String],
                                groupby_col_idx: Int,
                                chunk_size: Int = 640) raises -> List[Float64Array]:

    # 1) Build the group->index mapping (same as single-threaded)
    var num_cols = columns.size
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]

    var group_to_idx = Dict[FloatKey, Int]()
    var groups_vec = List[FloatKey]()

    for row_i in range(num_rows):
        var g_val = FloatKey(group_by_col[row_i])
        if not (g_val in group_to_idx):
            groups_vec.append(g_val)
            group_to_idx[g_val] = 0

    for i in range(groups_vec.size):
        group_to_idx[groups_vec[i]] = i

    var total_groups = groups_vec.size

    # We skip the actual aggregator here (final) until after partials
    # The final aggregator has (num_cols - 1) columns + 1 column for group keys
    # That means (num_cols - 1) sum columns, each sized [total_groups].
    var final_table = List[Float64Array]()
    for _ in range(num_cols - 1):
        final_table.append(Float64Array(total_groups))

    # 2) Compute how many chunks to split
    var n_chunks = (num_rows + chunk_size - 1) // chunk_size

    print("Number of chunks: ", n_chunks)

    # We'll store partial aggregators from each chunk in a list
    # partials[chunk_id] = one aggregator with (num_cols-1) Float64Arrays
    var partials = List[List[Float64Array]]()
    partials.resize(n_chunks, List[Float64Array]())

    # 3) Worker function that processes [start_row .. end_row)
    @parameter
    fn worker(chunk_id: Int):
        var start_row = chunk_id * chunk_size
        var end_row = min(start_row + chunk_size, num_rows)

        # Create local aggregator (num_cols-1) columns
        var local_agg = List[Float64Array]()
        try:
            for _ in range(num_cols - 1):
                local_agg.append(Float64Array(total_groups))
        except:
            pass

        var local_agg_i = 0

        try:
            for col_i in range(num_cols):
                if col_i != groupby_col_idx:
                    var cur_col = columns[col_i]
                    for row_i in range(start_row, end_row):
                        var g_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                        local_agg[local_agg_i][g_idx] += cur_col[row_i]
                    local_agg_i += 1
        except:
            pass

        partials[chunk_id] = local_agg

    # 4) Launch parallel tasks
    parallelize[worker](n_chunks, n_chunks)

    # 5) Merge partials into final_table
    for chunk_id in range(n_chunks):
        var partial_agg = partials[chunk_id]
        for c in range(partial_agg.size):
            var partial_col = partial_agg[c]
            # var final_col   = final_table[c]
            for g in range(total_groups):
                final_table[c][g] += partial_col[g]

    # 6) Optionally insert the group-key column at the front
    var group_col = Float64Array(total_groups)
    for key_i in range(total_groups):
        group_col[key_i] = groups_vec[key_i].i
    final_table.insert(0, group_col)

    return final_table

fn aggregation_sum_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = columns.size
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        # var cur_row_group = group_by_col[row_i]
        var cur_row_group_key = FloatKey(group_by_col[row_i])
        if not (cur_row_group_key in group_to_idx):
            groups_vec.append(cur_row_group_key)
            group_to_idx[cur_row_group_key] = 0

        # if group_to_idx.__contains__(cur_row_group):
        #     continue
        # else:
        #     groups_vec.append(cur_row_group)
        #     group_to_idx[cur_row_group] = 0
        
    # table
    #  col1_sum col_2_sum col1_avg col2_avg
    # 0
    # 1
    # 2
    # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(groups_vec.size))

    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                # l_extendedprice * (1 - l_discount) sum this for each row if this condition is given
                groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            agg_i += 1
    
    var group_col = Float64Array(groups_vec.size)
   
    for key_i in range(groups_vec.size):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table

fn aggregation_sum_conditional_f64(mut columns: List[Float64Array], col_names: List[String], read mask: List[Bool], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = columns.size
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i])
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0
        
    # table
    #  col1_sum col_2_sum col1_avg col2_avg
    # 0
    # 1
    # 2
    # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(groups_vec.size))

    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                if mask[row_i]:
                    groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            agg_i += 1
    
    var group_col = Float64Array(groups_vec.size)
   
    for key_i in range(groups_vec.size):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table

fn aggregation_min_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = columns.size
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()
    var neg_inf = neg_inf[DType.float64]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i]) 
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0
    
    # table
    #  col1_sum col_2_sum col1_avg col2_avg
    # 0
    # 1
    # 2
    # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(groups_vec.size, True))

    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                if groupby_table[agg_i][cur_row_group_idx] == neg_inf:
                    groupby_table[agg_i][cur_row_group_idx] = cur_col[row_i]
                else:
                    groupby_table[agg_i][cur_row_group_idx] = min(cur_col[row_i], groupby_table[agg_i][cur_row_group_idx])
            agg_i += 1
    
    var group_col = Float64Array(groups_vec.size)
   
    for key_i in range(groups_vec.size):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table

fn aggregation_count_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    
    var group_by_col = columns[groupby_col_idx]
    var num_rows = group_by_col.size
    var group_to_idx = Dict[FloatKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i])
        if not(cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0
        
    # table
    #  col1_sum col_2_sum col1_avg col2_avg
    # 0
    # 1
    # 2
    # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))


    groupby_table.append(Float64Array(groups_vec.size))
    
    for row_i in range(num_rows):
        var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
        groupby_table[0][cur_row_group_idx] += 1
    
    var group_col = Float64Array(groups_vec.size)
   
    for key_i in range(groups_vec.size):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table

fn aggregation_count_distinct_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int, aggregation_col_idx: Int) raises -> List[Float64Array]:
    var start_time = monotonic()

    var groupby_table = List[Float64Array]()
    # var groups_vec = List[SIMD[DType.float64, 1]]()
    
    
    var group_by_col = columns[groupby_col_idx]
    var aggregation_col = columns[aggregation_col_idx]

    var num_rows = group_by_col.size

    var groups_vec = List[FloatKey](capacity=num_rows)
    var group_to_idx = Dict[FloatKey, Int](power_of_two_initial_capacity=4194304)
    # var existing_groups = Set[FloatKey]()


    var end_time = monotonic()
    var execution_time_nanoseconds = end_time - start_time
    var execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("variable creation time:", execution_time_seconds)

    # create a list of lists
    # each group is mapped to a list of bools, to keep track if what elements already appeared
    var group_to_distinct_elements = List[SetElement]()

    # var max_value_in_elements = array_max_f64(aggregation_col)

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i])
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            # existing_groups.add(cur_row_group)
            group_to_idx[cur_row_group] = 0
            # continue
    #######   instead of append, use inner join approach to speed up

    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("groups_vec creation time:", execution_time_seconds)
    #return groupby_table
    # table
    #    selected_col count unique
    # 0
    # 1
    # 2
    # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # var cur_group_distinct = List[Bool]()
        # cur_group_distinct.resize(max_value_in_elements.__int__() + 1, False)
        # group_to_distinct_elements.append(cur_group_distinct)

    group_to_distinct_elements.resize(groups_vec.size, SetElement())

    print("group_vec size:", groups_vec.size)
    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("table creation time:", execution_time_seconds)

    
    # groupby_table.append(Float64Array(groups_vec.size))

    var distinct_count_array = Float64Array(groups_vec.size)


    # var num_work_items = 64

    # compute chunk_size so each thread handles a portion of the rows
    # var chunk_size = (num_rows + num_work_items - 1) // num_work_items
   
    # # Each thread will build its own local sets to avoid contention.
    # # partial_sets[t] is a list of SetElement, one for each group.
    # var partial_sets = List[List[SetElement]]()
    
    # # for t in range(num_work_items):
    # var sets_for_this_thread = List[SetElement]()
    # sets_for_this_thread.resize(groups_vec.size, SetElement())
    
    # partial_sets.append(sets_for_this_thread)

    # var sets_for_other_thread = List[SetElement]()
    # sets_for_other_thread.resize(groups_vec.size, SetElement())

    # partial_sets.append(sets_for_other_thread)

    
    # end_time = monotonic()
    # execution_time_nanoseconds = end_time - start_time
    # execution_time_seconds = execution_time_nanoseconds / 1000000000

    # print("partial sets time:", execution_time_seconds)


    # @parameter
    # fn parallel_count_distinct(thread_id: Int):
    #     # Calculate the row range for this thread
    #     var start = thread_id * chunk_size
    #     var end = min(start + chunk_size, num_rows)

    #     # Grab this thread's local sets
    #     var local_sets = partial_sets[thread_id]

    #     try:
    #         for row_i in range(start, end):
    #             var g_val = FloatKey(group_by_col[row_i])
    #             var agg_val = FloatKey(aggregation_col[row_i])
                
    #             # Which group index does this row belong to
    #             var g_idx = group_to_idx[g_val]

    #             # If not in local set, add it
    #             if not (agg_val in local_sets[g_idx].distinct_elements):
    #                 local_sets[g_idx].distinct_elements.add(agg_val)
    #     except:
    #         pass


    # # Launch the parallel jobs
    # parallelize[parallel_count_distinct](num_work_items)

    # # merge partial results for 1st worker
    # var local_sets_worker1 = partial_sets[0]
    # var local_sets_worker2 = partial_sets[1]
    # for group_i in range(groups_vec.size):
    #     group_to_distinct_elements[group_i].distinct_elements.update(local_sets_worker1[group_i].distinct_elements)
    #     group_to_distinct_elements[group_i].distinct_elements.update(local_sets_worker2[group_i].distinct_elements)
    #     distinct_count_array[group_i] = group_to_distinct_elements[group_i].distinct_elements.__len__()
        

    # @parameter
    # fn parallel_count_distinct(thread_id: Int):
    #     # Calculate the row range for this thread
    #     var start = thread_id * chunk_size
    #     var end = min(start + chunk_size, num_rows)
    
    #     try:
    #         for row_i in range(start, end):
    #             var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
    #             var cur_element_for_count = FloatKey(aggregation_col[row_i])
            
    #             if not (cur_element_for_count in group_to_distinct_elements[cur_row_group_idx].distinct_elements):
    #                 group_to_distinct_elements[cur_row_group_idx].distinct_elements.add(cur_element_for_count)
    #     except:
    #         pass
    
    # parallelize[parallel_count_distinct](num_work_items, n_workers)

    # # process leftover rows
    # var end = (num_rows // chunk_size) * chunk_size

    # for i in range(end, num_rows):
    #     var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[i])]
    #     var cur_element_for_count = FloatKey(aggregation_col[i])
    
    #     if not (cur_element_for_count in group_to_distinct_elements[cur_row_group_idx].distinct_elements):
    #         group_to_distinct_elements[cur_row_group_idx].distinct_elements.add(cur_element_for_count)
    
    for row_i in range(num_rows):
        # check if unique element of aggregation column is in the set
        var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
        var cur_element_for_count = FloatKey(aggregation_col[row_i])
    
        if not (cur_element_for_count in group_to_distinct_elements[cur_row_group_idx].distinct_elements):
            group_to_distinct_elements[cur_row_group_idx].distinct_elements.add(cur_element_for_count)
            # groupby_table[0][cur_row_group_idx] += 1
            # distinct_count_array[cur_row_group_idx] += 1
    
    end_time = monotonic()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("distinct count time:", execution_time_seconds)
    
    var group_col = Float64Array(groups_vec.size)
   
    for key_i in range(groups_vec.size):
        distinct_count_array[key_i] = group_to_distinct_elements[key_i].distinct_elements.__len__()
        group_col[key_i] = groups_vec[key_i].i

    # groupby_table.insert(0, group_col)
    # groupby_table.insert(0, group_col)

    groupby_table.append(group_col)
    groupby_table.append(distinct_count_array)

    return groupby_table

fn aggregation_all_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]() 
    # sum, avg two agg methods, then one column for groups and one for counts
    var num_cols_df = columns.size
    var num_cols = (columns.size - 1) * 2 + 2
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i])
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0
    
    var num_groups = groups_vec.size
    var group_count = List[Int]()
    # # Map groups to index like 0, 1, 2
    # print(groups_vec.size)
    for i in range(groups_vec.size):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        group_count.append(0)
    
    for _ in range(num_cols):
        groupby_table.append(Float64Array(num_groups))
    
    var agg_i = 1

    for col_i in range(num_cols_df):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                group_count[cur_row_group_idx] += 1
                groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            agg_i += 1
    
    # from col1 to size - 1, the table stores sums
    for col in range(1, num_cols_df):
        var sums = groupby_table[col]
        for group in range(num_groups):
            var group_count = group_count[group] / (num_cols_df - 1)
            groupby_table[agg_i][group] = sums[group] / group_count
        agg_i += 1
    
    for group in range(num_groups):
        var group_count = group_count[group] / (num_cols_df - 1)
        groupby_table[num_cols - 1][group] = group_count
        groupby_table[0][group] = groups_vec[group].i

    return groupby_table

fn aggregation_mean_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    # has groupby column, map each group to an integer index
    # has a aggregation column
    # go through the values in the aggregation column, sum the values into the correct group
    # each group has several sums for each aggregation column

    # columns passed in should have all the aggregation columns
    # sum(l_quantity) as sum_qty,
	# sum(l_extendedprice) as sum_base_price,
	# avg(l_quantity) as avg_qty,
	# avg(l_extendedprice) as avg_price,
	# avg(l_discount) as avg_disc,
	# count(*) as count_order
    

    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = columns.size
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i])
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0
    
    var group_count = List[Int]()
    group_count.resize(groups_vec.size, 0)

    # # Map groups to index like 0, 1, 2
    for i in range(groups_vec.size):
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))
            
    for i in range(group_by_col.size):
        group_count[group_to_idx[FloatKey(group_by_col[i])]] += 1

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(groups_vec.size))

    
    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            
            for group in range(groups_vec.size):
                groupby_table[agg_i][group] = (groupby_table[agg_i][group]) / (group_count[group])

            agg_i += 1

    var group_col = Float64Array(groups_vec.size)
   
    for key_i in range(groups_vec.size):
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table


# fn aggregation_sum_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
#     var groupby_table = List[Float64Array]()
#     var groups_vec = List[SIMD[DType.float64, 1]]()
#     var num_cols = columns.size
#     var num_rows = columns[groupby_col_idx].size
#     var group_by_col = columns[groupby_col_idx]
#     var group_to_idx = Dict[FloatKey, Int]()

#     for row_i in range(num_rows):
#         # Each row in the groupby column belongs to a group
#         var cur_row_group = group_by_col[row_i] 
#         if group_to_idx.__contains__(cur_row_group):
#             continue
#         else:
#             groups_vec.append(cur_row_group)
#             group_to_idx[cur_row_group] = 0
#     # Map groups to index like 0, 1, 2
#     for i in range(groups_vec.size):
#         # print(groups_vec[i])
#         group_to_idx[groups_vec[i]] = i
#         groupby_table.append(Float64Array(num_cols - 1))

#     var agg_i = 0
    
#     for col_i in range(num_cols):
#         if col_i != groupby_col_idx:
#             var cur_col = columns[col_i]
#             for row_i in range(num_rows):
#                 var cur_row_group_idx = group_to_idx[group_by_col[row_i]]
#                 groupby_table[cur_row_group_idx][agg_i] += cur_col[row_i]
#             agg_i += 1

#     return groupby_table

fn combine_masks(read masks: List[List[Bool]], logical_operator: String) raises -> List[Bool]:
        var filtered_mask = masks[0]
        var mask_len = filtered_mask.size

        if logical_operator == "AND":
            for i in range(1, masks.size):
                var cur_mask = masks[i]
                for j in range(mask_len):
                    filtered_mask[j] = (filtered_mask[j] and cur_mask[j])
        elif logical_operator == "OR":
            for i in range(1, masks.size):
                var cur_mask = masks[i]
                for j in range(mask_len):
                    filtered_mask[j] = (filtered_mask[j] or cur_mask[j])
        
        return filtered_mask

fn filter_string_equal(mut df: DataFrameF64, read str_col: List[String], filter_str: String) raises:
    var selected_indices = List[Int]()
    for i in range(df.columns[0].size):
        if str_col[i] == filter_str:
            selected_indices.append(i)

    var filtered_data = List[Float64Array]()
    for col_i in range(df.columns.size):
        var col_to_fill = Float64Array(selected_indices.size)
        var original_col = df.columns[col_i]
        for row_i in range(selected_indices.size):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data

fn filter_string_equal_mask(read str_col: List[String], filter_str: String) ->  List[Bool]:
    var n = str_col.size
    # var chunk_size = 64000
    var num_work_items = 8
    var chunk_size = (n + num_work_items - 1) // num_work_items

    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(n, False)
   
    @parameter
    fn worker(thread_id: Int):
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, n)
        for i in range(start, end):
            if str_col[i] == filter_str:
                selected_indices_mask[i] = True

    parallelize[worker](num_work_items)

    return selected_indices_mask

fn filter_string_not_equal_mask(read str_col: List[String], filter_str: String) ->  List[Bool]:
    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(str_col.size, False)

    for i in range(str_col.size):
        if str_col[i] != filter_str:
            selected_indices_mask[i] = True

    return selected_indices_mask

fn filter_string_contains(mut df: DataFrameF64, read str_col: List[String], filter_str: String) raises:
    # var selected_indices = List[Int]()
    # for i in range(df.columns[0].size):
    #     if filter_str in str_col[i]:
    #         selected_indices.append(i)

    # var filtered_data = List[Float64Array]()
    # for col_i in range(df.columns.size):
    #     var col_to_fill = Float64Array(selected_indices.size)
    #     var original_col = df.columns[col_i]
    #     for row_i in range(selected_indices.size):
    #         col_to_fill[row_i] = original_col[selected_indices[row_i]]
    #     filtered_data.append(col_to_fill)
        
    # df.columns = filtered_data

    var n = df.columns[0].size
    var chunk_size = 64000
    var n_chunks   = (n + chunk_size - 1) // chunk_size

    # Each chunk will store matching row indices in its own list
    var partial_lists = List[List[Int]]()
    partial_lists.resize(n_chunks, List[Int]())

    @parameter
    fn filter_worker(chunk_id: Int):
        var start_i = chunk_id * chunk_size
        var end_i = min(start_i + chunk_size, n)

        var local_matches = List[Int]()

        for i in range(start_i, end_i):
            if filter_str in str_col[i]:
                local_matches.append(i)

        partial_lists[chunk_id] = local_matches

    # Launch parallel tasks to find matching rows
    parallelize[filter_worker](n_chunks)

    var selected_indices = List[Int]()
    for c in range(n_chunks):
        var local_matches = partial_lists[c]
        for idx in range(local_matches.size):
            selected_indices.append(local_matches[idx])


    var filtered_data = List[Float64Array]()
    for col_i in range(df.columns.size):
        var col_to_fill = Float64Array(selected_indices.size)
        var original_col = df.columns[col_i]
        for row_i in range(selected_indices.size):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data
    

fn filter_string_endwith(mut df: DataFrameF64, read str_col: List[String], filter_str: String) raises:
    var selected_indices = List[Int]()
    for i in range(df.columns[0].size):
        if str_col[i].endswith(filter_str):
            selected_indices.append(i)

    var filtered_data = List[Float64Array]()
    for col_i in range(df.columns.size):
        var col_to_fill = Float64Array(selected_indices.size)
        var original_col = df.columns[col_i]
        for row_i in range(selected_indices.size):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data

fn filter_string_startwith(mut df: DataFrameF64, read str_col: List[String], filter_str: String) raises:
    var selected_indices = List[Int]()
    # var n = df.columns[0].size

    # var chunk_size = 64000

    # var n_chunks = (n + chunk_size - 1) // chunk_size

    # var partial_results = List[List[Int]]()
    # partial_results.resize(n_chunks, List[Int]())


    # @parameter
    # fn filter_worker(chunk_id: Int):
    #     var start_i = chunk_id * chunk_size
    #     var end_i = min(start_i + chunk_size, n)

    #     var local_list = List[Int]()

    #     for i in range(start_i, end_i):
    #         if str_col[i].startswith(filter_str):
    #             local_list.append(i)

    #     partial_results[chunk_id] = local_list

    # # 6) Launch parallel tasks (each chunk in its own task)
    # parallelize[filter_worker](n_chunks)

    # for chunk_id in range(n_chunks):
    #     var local_list = partial_results[chunk_id]
    #     for idx in range(local_list.size):
    #         selected_indices.append(local_list[idx])

    for i in range(df.columns[0].size):
        if str_col[i].startswith(filter_str):
            selected_indices.append(i)

    var filtered_data = List[Float64Array]()
    for col_i in range(df.columns.size):
        var col_to_fill = Float64Array(selected_indices.size)
        var original_col = df.columns[col_i]
        for row_i in range(selected_indices.size):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data

fn filter_string_not_startwith_mask(read str_col: List[String], filter_str: String) -> List[Bool]:
    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(str_col.size, False)

    for i in range(str_col.size):
        if not (str_col[i].startswith(filter_str)):
            selected_indices_mask[i] = True

    return selected_indices_mask

fn filter_f64_IN_mask(read float_col: Float64Array, filter_list: Float64Array) -> List[Bool]:
    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(float_col.size, False)

    var float_list = Set[FloatKey]()

    for i in range(filter_list.size):
        float_list.add(FloatKey(filter_list[i]))

    for i in range(float_col.size):
        if FloatKey(float_col[i]) in float_list:
            selected_indices_mask[i] = True

    return selected_indices_mask

fn filter_string_IN_mask(read string_col: List[String], filter_list: List[String]) -> List[Bool]:
    var n = string_col.size
    var num_work_items = 8
    # var chunk_size = 64000
    var chunk_size = (n + num_work_items - 1) // num_work_items

    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(string_col.size, False)

    var string_list = Set[String]()

    for i in range(filter_list.size):
        string_list.add(filter_list[i])

    # for i in range(string_col.size):
    #     if string_col[i] in string_list:
    #         selected_indices_mask[i] = True

    # 4) Per-thread function
    @parameter
    fn worker(thread_id: Int):
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, n)
        for i in range(start, end):
            if string_col[i] in string_list:
                selected_indices_mask[i] = True

    parallelize[worker](num_work_items)

    return selected_indices_mask

fn filter_not_string_exists_before(mut df: DataFrameF64, read str_col: List[String], filter_str1: String, filter_str2: String) raises:
    # ".*str1.*str2.*" str2 appears after str1 at least once

    # var selected_indices = List[Int]()
    # for i in range(df.columns[0].size):
    #     # find index of the first occurrence of str1
    #     var first_filter_str1_pos = str_col[i].find(filter_str1)
    #     # find index of the first occurrence of str1
    #     var last_filter_str2_pos = str_col[i].rfind(filter_str2)
    #     if  not (first_filter_str1_pos != -1 and last_filter_str2_pos != -1 and first_filter_str1_pos < last_filter_str2_pos):
    #         selected_indices.append(i)

    var n = df.columns[0].size
    var chunk_size = 6400
    var n_chunks   = (n + chunk_size - 1) // chunk_size

    # Each chunk collects matching indices in local lists
    var partial_results = List[List[Int]]()
    partial_results.resize(n_chunks, List[Int]())

    @parameter
    fn filter_worker(chunk_id: Int):
        var start_i = chunk_id * chunk_size
        var end_i = min(start_i + chunk_size, n)

        var local_list = List[Int]()

        for i in range(start_i, end_i):
            # find index of the first occurrence of filter_str1
            var first_str1_pos = str_col[i].find(filter_str1)
            # find index of the last occurrence of filter_str2
            var last_str2_pos  = str_col[i].rfind(filter_str2)

            # If str1 and str2 both appear with str1 < str2, we EXCLUDE
            # i.e. "if not (first_str1_pos != -1 and last_str2_pos != -1 and first_str1_pos < last_str2_pos)"
            if not (first_str1_pos != -1 and last_str2_pos != -1 and first_str1_pos < last_str2_pos):
                local_list.append(i)

        partial_results[chunk_id] = local_list

    # Launch parallel filtering
    parallelize[filter_worker](n_chunks)

    # Merge partial results
    var selected_indices = List[Int]()
    for c in range(n_chunks):
        var local_list = partial_results[c]
        for idx in range(local_list.size):
            selected_indices.append(local_list[idx])

    var filtered_data = List[Float64Array]()
    for col_i in range(df.columns.size):
        var col_to_fill = Float64Array(selected_indices.size)
        var original_col = df.columns[col_i]
        for row_i in range(selected_indices.size):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data


fn cast_as_float64(read str_col: List[String], substr_start_index: Int, substr_end_index: Int) raises -> Float64Array:
    var float_col = Float64Array(str_col.size)

    for i in range(str_col.size):
        var substr = str_col[i][substr_start_index:substr_end_index]
        float_col[i] = float(substr)

    return float_col


fn evaluate_query6[T: PredicateF64, T2: PredicateF64,
                   T3: PredicateF64, T4: PredicateF64,
                   T5: PredicateF64](mut column_1: Float64Array, mut column_2: Float64Array, mut column_3: Float64Array,
                                    predicate_1: T, predicate_2: T2, predicate_3: T3, predicate_4: T4, predicate_5: T5,
                                    value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                                    value_cmp_3: SIMD[DType.float64, 1], value_cmp_4: SIMD[DType.float64, 1],
                                    value_cmp_5: SIMD[DType.float64, 1],
                                    logical_operator: String) raises -> List[Int]:
                                    
    var vector_of_index = List[Int]()

    if logical_operator == "AND":
        for row_i in range(column_1.size):
            if (predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_1[row_i], value_cmp_2))
                and (predicate_3.evaluate(column_2[row_i], value_cmp_3) and predicate_4.evaluate(column_2[row_i], value_cmp_4))
                and (predicate_5.evaluate(column_3[row_i], value_cmp_5)):
                vector_of_index.append(row_i)
    else:
        pass

    return vector_of_index

fn evaluate_f64_alt[T: PredicateF64, T2: PredicateF64](read column_1: Float64Array, read column_2: Float64Array,
                      predicate_1: T, predicate_2: T2,
                      value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                      logical_operator: String) raises -> Int32Array:
    
    var n = column_1.size
    var total_count = 0
    var num_threads = 2

    if logical_operator == "AND":
        var chunk_size = 64000
        var n_chunks   = (n + chunk_size - 1) // chunk_size

        # We'll store partial results in one list per chunk
        var partial_results = List[List[Int]]()
        partial_results.resize(n_chunks, List[Int]())

        var partial_count = Int32Array(n_chunks)
        # var start = perf_counter()
        @parameter
        fn worker_and(chunk_id: Int):
            var start_i = chunk_id * chunk_size
            var end_i   = min(start_i + chunk_size, n)

            var local_list = List[Int]()
            var local_count = 0
            # initialize local idxs to have (end_i - start_i) elements
            # SIMD load column data
            for row_i in range(start_i, end_i):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
                    local_list.append(row_i)
                    local_count += 1

            partial_results[chunk_id] = local_list
            partial_count[chunk_id] = local_count

        parallelize[worker_and](n_chunks, num_threads)

        
        for c in range(n_chunks):
            total_count += (partial_count[c].__int__())
        
        var filtered_idxs = Int32Array(total_count)
        var i = 0
        for c in range(n_chunks):
            var local_list = partial_results[c]
            for idx in range(local_list.size):
                filtered_idxs[i] = local_list[idx]
                i += 1
        
        #var end = perf_counter()
    
        return filtered_idxs
    elif logical_operator == "OR":
        var chunk_size = 64000
        var n_chunks   = (n + chunk_size - 1) // chunk_size

        # We'll store partial results in one list per chunk
        var partial_results = List[List[Int]]()
        partial_results.resize(n_chunks, List[Int]())

        var partial_count = Int32Array(n_chunks)
        # var start = perf_counter()
        @parameter
        fn worker_or(chunk_id: Int):
            var start_i = chunk_id * chunk_size
            var end_i   = min(start_i + chunk_size, n)

            var local_list = List[Int]()
            var local_count = 0
            # initialize local idxs to have (end_i - start_i) elements
            # SIMD load column data
            for row_i in range(start_i, end_i):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1) or predicate_2.evaluate(column_2[row_i], value_cmp_2):
                    local_list.append(row_i)
                    local_count += 1

            partial_results[chunk_id] = local_list
            partial_count[chunk_id] = local_count

        parallelize[worker_or](n_chunks, num_threads)

        
        for c in range(n_chunks):
            total_count += (partial_count[c].__int__())
        
        var filtered_idxs = Int32Array(total_count)
        var i = 0
        for c in range(n_chunks):
            var local_list = partial_results[c]
            for idx in range(local_list.size):
                filtered_idxs[i] = local_list[idx]
                i += 1
        
        return filtered_idxs
    else:
        if logical_operator == "COL":
            #var start = perf_counter()
            var chunk_size = 64000
            var n_chunks   = (n + chunk_size - 1) // chunk_size

            # We'll store partial results in one list per chunk
            var partial_results = List[List[Int]]()
            partial_results.resize(n_chunks, List[Int]())

            var partial_count = Int32Array(n_chunks)
            # var start = perf_counter()
            @parameter
            fn worker(chunk_id: Int):
                var start_i = chunk_id * chunk_size
                var end_i   = min(start_i + chunk_size, n)

                var local_list = List[Int]()
                var local_count = 0
                # initialize local idxs to have (end_i - start_i) elements
                # SIMD load column data
                for row_i in range(start_i, end_i):
                    if predicate_1.evaluate(column_1[row_i], column_2[row_i]):
                        local_list.append(row_i)
                        local_count += 1

                partial_results[chunk_id] = local_list
                partial_count[chunk_id] = local_count

            parallelize[worker](n_chunks, num_threads)

            
            for c in range(n_chunks):
                total_count += (partial_count[c].__int__())
            
            var filtered_idxs = Int32Array(total_count)
            var i = 0
            for c in range(n_chunks):
                var local_list = partial_results[c]
                for idx in range(local_list.size):
                    filtered_idxs[i] = local_list[idx]
                    i += 1
            
            #var end = perf_counter()
        
            return filtered_idxs
        else:
            #var start = perf_counter()
            var chunk_size = 64000
            var n_chunks   = (n + chunk_size - 1) // chunk_size

            # We'll store partial results in one list per chunk
            var partial_results = List[List[Int]]()
            partial_results.resize(n_chunks, List[Int]())

            var partial_count = Int32Array(n_chunks)
            # var start = perf_counter()
            @parameter
            fn col_worker(chunk_id: Int):
                var start_i = chunk_id * chunk_size
                var end_i   = min(start_i + chunk_size, n)

                var local_list = List[Int]()
                var local_count = 0
                # initialize local idxs to have (end_i - start_i) elements
                # SIMD load column data
                for row_i in range(start_i, end_i):
                    if predicate_1.evaluate(column_1[row_i], value_cmp_1):
                        local_list.append(row_i)
                        local_count += 1

                partial_results[chunk_id] = local_list
                partial_count[chunk_id] = local_count

            parallelize[col_worker](n_chunks, num_threads)

            
            for c in range(n_chunks):
                total_count += (partial_count[c].__int__())
            
            var filtered_idxs = Int32Array(total_count)
            var i = 0
            for c in range(n_chunks):
                var local_list = partial_results[c]
                for idx in range(local_list.size):
                    filtered_idxs[i] = local_list[idx]
                    i += 1
            
            #var end = perf_counter()
        
            return filtered_idxs
    
    return Int32Array(0)

fn evaluate_f64[T: PredicateF64, T2: PredicateF64](read column_1: Float64Array, read column_2: Float64Array,
                      predicate_1: T, predicate_2: T2,
                      value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                      logical_operator: String) raises -> List[Int]:
    
    var n = column_1.size
    var vector_of_index = List[Int]()
    var num_threads = 2

    if logical_operator == "AND":
        var chunk_size = 64000
        var n_chunks   = (n + chunk_size - 1) // chunk_size

        # We'll store partial results in one list per chunk
        var partial_results = List[List[Int]]()
        partial_results.resize(n_chunks, List[Int]())

        # var start = perf_counter()
        @parameter
        fn worker(chunk_id: Int):
            var start_i = chunk_id * chunk_size
            var end_i   = min(start_i + chunk_size, n)

            var local_list = List[Int]()

            # initialize local idxs to have (end_i - start_i) elements
            # SIMD load column data
            for row_i in range(start_i, end_i):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
                    local_list.append(row_i)

            partial_results[chunk_id] = local_list

        parallelize[worker](n_chunks, num_threads)

        # var end = perf_counter()
        # print("Time taken to filter parallel:", end - start)
        # for row_i in range(column_1.size):
        #     if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
        #         vector_of_index.append(row_i)

        for c in range(n_chunks):
            var local_list = partial_results[c]
            for idx in range(local_list.size):
                vector_of_index.append(local_list[idx])

    elif logical_operator == "OR":
        for row_i in range(column_1.size):
            if predicate_1.evaluate(column_1[row_i], value_cmp_1) or predicate_2.evaluate(column_2[row_i], value_cmp_2):
                vector_of_index.append(row_i)
    
    else:
        if logical_operator == "COL":
            # var chunk_size = 640000
            # var n_chunks   = (n + chunk_size - 1) // chunk_size

            # print("n_chunks", n_chunks)
            # # We'll store partial results in one list per chunk
            # var partial_results = List[List[Int]]()
            # partial_results.resize(n_chunks, List[Int]())
            for row_i in range(column_1.size):
                if predicate_1.evaluate(column_1[row_i], column_2[row_i]):
                    vector_of_index.append(row_i)

            # @parameter
            # fn worker2(chunk_id: Int):
            #     var start_i = chunk_id * chunk_size
            #     var end_i   = min(start_i + chunk_size, n)

            #     var local_list = List[Int]()

            #     for row_i in range(start_i, end_i):
            #         if predicate_1.evaluate(column_1[row_i], column_2[row_i]):
            #             local_list.append(row_i)

            #     partial_results[chunk_id] = local_list

            # parallelize[worker2](n_chunks, 8)
            
            # # # var i = 0
            # for c in range(n_chunks):
            #     var local_list = partial_results[c]
            #     for idx in range(local_list.size):
            #         vector_of_index.append(local_list[idx])

        else:
            var start = perf_counter()
            for row_i in range(column_1.size):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1):
                    vector_of_index.append(row_i)
            var end = perf_counter()
            print("Time taken to filter parallel:", end - start)
    
    return vector_of_index

fn evaluate_f64_mask[T: PredicateF64, T2: PredicateF64](read column_1: Float64Array, read column_2: Float64Array,
                      predicate_1: T, predicate_2: T2,
                      value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                      logical_operator: String) raises -> List[Bool]:
    var index_mask = List[Bool]()

    if logical_operator == "AND":
        for row_i in range(column_1.size):
            if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
                index_mask.append(True)
            else:
                index_mask.append(False)

    elif logical_operator == "OR":
        for row_i in range(column_1.size):
            if predicate_1.evaluate(column_1[row_i], value_cmp_1) or predicate_2.evaluate(column_2[row_i], value_cmp_2):
                index_mask.append(True)
            else:
                index_mask.append(False)
    
    else:
        if logical_operator == "COL":
            for row_i in range(column_1.size):
                if predicate_1.evaluate(column_1[row_i], column_2[row_i]):
                    index_mask.append(True)
                else:
                    index_mask.append(False)
        else:
            for row_i in range(column_1.size):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1):
                    index_mask.append(True)
                else:
                    index_mask.append(False)
    
    return index_mask

fn evaluate_f32(read column_1: Float32Array, read column_2: Float32Array,
                      value_cmp_1: SIMD[DType.float32, 1], value_cmp_2: SIMD[DType.float32, 1]) raises -> List[Int]:
    var vector_of_index = List[Int]()
    
    for row_i in range(column_1.size):
        if column_1[row_i] > value_cmp_1 and column_2[row_i] <= value_cmp_2:
            vector_of_index.append(row_i)
    
    return vector_of_index

fn evaluate_i32(mut column: Int32Array, operation: String, value_cmp: SIMD[DType.int32, 1]) raises -> List[Int]:
    # In pandas, evaluate is called on each row with a condition
    # Eventually, an array of Bools is used to select filtered data
    var vector = List[Int]()

    if operation == "<":
        for row_i in range(column.size):
            if column[row_i] < value_cmp:
                vector.append(row_i)
    elif operation == ">":
        for row_i in range(column.size):
            if column[row_i] > value_cmp:
                vector.append(row_i)
    else:
        for row_i in range(column.size):
            if column[row_i] == value_cmp:
                vector.append(row_i)
    
    return vector

# fn inner_join_i32_naive(mut df1: DataFrameI32, mut df2: DataFrameI32, key_column: String) raises:
    
#     var key_column1 = df1[key_column]
#     var key_column2 = df2[key_column]

#     # Initialize empty lists to hold the indices of matching entries
#     var indices_list1 = List[Int]()
#     var indices_list2 = List[Int]()
    
#     # Create a mapping from join_key values to their indices in list2 for quick lookup
#     var key_to_index = Dict[IntKey, List[Int]]()

#     for index in range(key_column2.size):
#         var key_val = key_column2[index]
#         if not key_to_index.__contains__(key_val):
#             key_to_index[key_val] = List[Int]()
#         key_to_index[key_val].append(index)

#     for index1 in range(key_column1.size):
#         var key_val = key_column1[index1]
#         if key_to_index.__contains__(key_val):
#             var key_idxs = key_to_index[key_val]
#             for index2 in range(key_idxs.__len__()):
#                 indices_list1.append(index1)
#                 indices_list2.append(key_idxs[index2])

fn inner_join_i32(mut df1: DataFrameI32, mut df2: DataFrameI32, key_column: String) raises -> DataFrameI32:
    # find the max number of distinct groups in both join columns
    var key_column1 = df1[key_column]
    var key_column2 = df2[key_column]

    var max1 = key_column1[0]
    var max2 = key_column2[0]

    for i in range(key_column1.size):
        max1 = max(max1, key_column1[i])

    for i in range(key_column2.size):
        max2 = max(max2, key_column2[i])
    
    var max_groups = max(max1, max2).__int__()
    var count = 0

    
    var left_count = List[Int]()
    var right_count = List[Int]()
    left_count.resize(max_groups + 1, 0)
    right_count.resize(max_groups + 1, 0)
    
    # # First pass to count frequencies of each group/key
    for i in range(key_column1.size):
        left_count[key_column1[i].__int__()] += 1

    for i in range(key_column2.size):
        right_count[key_column2[i].__int__()] += 1

    # Determine how many combinations will result from this group in the output
    # cartesion product
    # if the group has entries in both columns
    for i in range(1, max_groups+1):
        var lc = left_count[i]
        var rc = right_count[i]
        if lc > 0 and rc > 0:
            count += lc * rc

    var left_indexer = Int32Array(count)
    var right_indexer = Int32Array(count)

    var left_pos = left_count[0]
    var right_pos = right_count[0]
    var position = 0

    for i in range(1, max_groups+1):
        var lc = left_count[i]
        var rc = right_count[i]

        if rc > 0 and lc > 0:
            for j in range(lc):
               
                # Calculate the starting index in the result arrays
                # for group left[i] and all elements from right belonging to the same group.
                # 1,2,3      1,2,1,3,2
                # 0,0,1,1,2          0,1,2,3,4
                # Group 1 -> lc:1 rc:2

                # offset = 0 + 0 * 2
                var offset = position + j * rc
                # print("offset:", offset)
    
                for k in range(rc):
                    # left_indexer[0 + 0] = 0 + 0
                    # left_indexer[0 + 1] = 0 + 0
                    left_indexer[offset + k] = left_pos + j
                    # right_indexer[0 + 0] = 0 + 0
                    # right_indexer[0 + 1] = 0 + 1
                    right_indexer[offset + k] = right_pos + k
            # update pointer to skip to next location after all combinations in current group
            position += lc * rc

        # skip to next group
        left_pos += lc
        right_pos += rc
    

    # Use indexers to build DataFrame
    var col_data1 = List[Int32Array]()
    var col_names1 = List[String]()

    # [0,1,2,3,4] [1,1,2,2,3]
    col_names1 = df1.column_names
    
    for col_i in range (df1.column_names.__len__()):
        var row_input_idx = 0
        var col = df1[col_i]
        var col_to_append = Int32Array(left_indexer.size)

        for row_i in range(left_indexer.size):
            #print("get:", col[(left_indexer[row_i]).__int__()])
            col_to_append[row_input_idx] = col[(left_indexer[row_i]).__int__()]
            row_input_idx += 1
        col_data1.append(col_to_append)

    # var col_data2 = List[Int32Array]()
    # var col_names2 = List[String]()

    # [0,1,2,3,4] [1,1,2,2,3]
    
    for col_i in range (df2.column_names.__len__()):
        if df2.column_names[col_i] != key_column:
            col_names1.append(df2.column_names[col_i])
            var row_input_idx = 0
            var col = df2[col_i]
            var col_to_append = Int32Array(right_indexer.size)

            for row_i in range(right_indexer.size):
                #print("get:", col[(right_indexer[row_i]).__int__()])
                col_to_append[row_input_idx] = col[(right_indexer[row_i]).__int__()]
                row_input_idx += 1
            col_data1.append(col_to_append)

    # for i in range(left_indexer.size):
    #     print(left_indexer[i])
    
    # for i in range(right_indexer.size):
    #     print(right_indexer[i])
    
    return DataFrameI32(col_data1, col_names1)

fn left_join_f64(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameF64:
    var key_column1 = df1[key_column]
    var key_column2 = df2[key_column]

    # find the max key value for both dataframes
    var max1 = key_column1[0]
    var max2 = key_column2[0]

    for i in range(key_column1.size):
        max1 = max(max1, key_column1[i])

    for i in range(key_column2.size):
        max2 = max(max2, key_column2[i])
    
    var max_groups = max(max1, max2).__int__()

    # var left_count = List[Int]()
    # var right_count = List[Int]()

    var left_count = Int32Array(max_groups + 1)
    var right_count = Int32Array(max_groups + 1)
    # left_count.resize(max_groups + 1, 0)
    # right_count.resize(max_groups + 1, 0)

    for i in range(key_column1.size):
        var group_id = key_column1[i].__int__()
        left_count[group_id] += 1

    for i in range(key_column2.size):
        var group_id = key_column2[i].__int__()
        right_count[group_id] += 1
    

    # for each group i in [1, max_groups]
    # if right_count[i] > 0, need left_count[i] * right_count[i] rows
    # else, need left_count[i] rows -> right side is unmatched

    var total_rows = 0
    for i in range(1, max_groups + 1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())
        if lc > 0:
            if rc > 0:
                total_rows += lc * rc
            else:
                total_rows += lc
    

    var left_indexer = Int32Array(total_rows)
    var right_indexer = Int32Array(total_rows)

    var position = 0

    var running_left_pos = left_count[0]   
    var running_right_pos = right_count[0]


    for i in range(1, max_groups + 1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())

        if lc == 0:
            # No rows on the left for group i => nothing to do
            running_left_pos += lc
            running_right_pos += rc
            continue

        if rc > 0:
            # each left row in group i
            for j in range(lc):       
                var offset = position + j * rc
                # each right row in group i
                for k in range(rc):   
                    left_indexer[offset + k]  = running_left_pos + j
                    right_indexer[offset + k] = running_right_pos + k
            position += lc * rc
        else:
            # no matching right rows -> need lc rows, right_indexer = -1
            for j in range(lc):
                left_indexer[position + j]  = running_left_pos + j
                right_indexer[position + j] = -1

            position += lc

        running_left_pos  += lc
        running_right_pos += rc


    var new_left_indexer = groupsort_indexer(key_column1, left_indexer, left_count, max1.__int__())

    var temp_right_indexer = Int32Array(total_rows)
    for i in range(total_rows):
        var idx = right_indexer[i]
        if idx < 0:
            temp_right_indexer[i] = 0
        else:
            temp_right_indexer[i] = idx

    var new_right_indexer = groupsort_indexer(key_column2, temp_right_indexer, right_count, max2.__int__())

    # build joined dataframes using indexers
    var col_data  = List[Float64Array]()
    var col_names = List[String]()

    col_names = df1.column_names

    for col_i in range(df1.column_names.__len__()):
        var row_input_idx = 0
        var col = df1[col_i]
        var out_col = Float64Array(left_indexer.size)

        for row_i in range(left_indexer.size):
            var matched_idx = new_left_indexer[row_i].__int__()
            out_col[row_i] = col[matched_idx]
            row_input_idx += 1

        col_data.append(out_col)


    for col_i in range(df2.column_names.__len__()):
        if df2.column_names[col_i] != key_column:
            col_names.append(df2.column_names[col_i])
            var row_input_idx = 0
            var col = df2[col_i]
            var out_col = Float64Array(right_indexer.size)

            for row_i in range(right_indexer.size):
                var potential_matched_idx = new_right_indexer[row_i].__int__()
                # check if the original was -1, fill with default value
                if right_indexer[row_i] == -1:
                    out_col[row_input_idx] = neg_inf[DType.float64]()
                else:
                    out_col[row_input_idx] = col[potential_matched_idx]
                row_input_idx += 1
            
            col_data.append(out_col)

    return DataFrameF64(col_data, col_names)

fn inner_join_f64(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameF64:
    var start_time = monotonic()

    var num_threads = 2

    var key_column1 = df1[key_column]
    var key_column2 = df2[key_column]

    var max1 = key_column1[0]
    var max2 = key_column2[0]

    # var max1 = find_max_in_parallel(key_column1)

    # # Find max in parallel for key_column2
    # var max2 = find_max_in_parallel(key_column2)
    var max_limit1 = (key_column1.size // 8) * 8
    var max_limit2 = (key_column2.size // 8) * 8

    for i in range(0, max_limit1, 8):
        var maxs = key_column1.data.load[width=8](i)
        var temp_max = maxs.reduce_max()
        max1 = max(max1, temp_max)
    
    # take care of remaining elements for column 1
    for i in range(max_limit1, key_column1.size):
        max1 = max(max1, key_column1[i])

    for i in range(0, max_limit2, 8):
        var maxs = key_column2.data.load[width=8](i)
        var temp_max = maxs.reduce_max()
        max2 = max(max2, temp_max)
    
    # take care of remaining elements for column 2
    for i in range(max_limit2, key_column2.size):
        max2 = max(max2, key_column2[i])

    # var max1 = parallel_find_max(key_column1)
    # var max2 = parallel_find_max(key_column2)
    
    var max_groups = max(max1, max2).__int__()
    
    var count = 0
    
    var end_time = monotonic()

    print("find max time: ", (end_time - start_time) / 1000000000)

    # print("max grps")
    # print(max_groups)
    # var left_count = List[Int]()
    # var right_count = List[Int]()

    start_time = monotonic()

    var left_count = Int32Array(max_groups + 1)
    var right_count = Int32Array(max_groups + 1)
    # left_count.resize(max_groups + 1, 0)
    # right_count.resize(max_groups + 1, 0)
    
    # # First pass to count frequencies of each group/key
    for i in range(key_column1.size):
        left_count[key_column1[i].__int__()] += 1

    for i in range(key_column2.size):
        right_count[key_column2[i].__int__()] += 1

    # var left_count = count_group_freq_parallel(key_column1, max_groups)
    # var right_count = count_group_freq_parallel(key_column2, max_groups)

    end_time = monotonic()
    print("count time:", (end_time - start_time) / 1000000000)

    # Determine how many combinations will result from this group in the output
    # cartesion product
    # if the group has entries in both columns

    start_time = monotonic()

    for i in range(1, max_groups+1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())
        if lc > 0 and rc > 0:
            count += lc * rc

    var left_indexer = Int32Array(count)
    var right_indexer = Int32Array(count)

    var left_pos = left_count[0]
    var right_pos = right_count[0]
    var position = 0

    for i in range(1, max_groups+1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())

        if rc > 0 and lc > 0:
            for j in range(lc):
               
                # Calculate the starting index in the result arrays
                # for group left[i] and all elements from right belonging to the same group.
                # 1,2,3      1,2,1,3,2
                # 0,0,1,1,2          0,1,2,3,4
                # Group 1 -> lc:1 rc:2

                # offset = 0 + 0 * 2
                var offset = position + j * rc
                # print("offset:", offset)
    
                for k in range(rc):
                    # left_indexer[0 + 0] = 0 + 0
                    # left_indexer[0 + 1] = 0 + 0
                    left_indexer[offset + k] = left_pos + j
                    # right_indexer[0 + 0] = 0 + 0
                    # right_indexer[0 + 1] = 0 + 1
                    right_indexer[offset + k] = right_pos + k
            # update pointer to skip to next location after all combinations in current group
            position += lc * rc

        # skip to next group
        left_pos += lc
        right_pos += rc

    end_time = monotonic()
    print("indexer time:", (end_time - start_time) / 1000000000)

    # print("left indexer:")
    # for i in range(left_indexer.size):
    #     print(left_indexer[i])

    # print("right indexer:")
    # for i in range(right_indexer.size):
    #     print(right_indexer[i])

    #var start_time = monotonic()
    
    start_time = monotonic()

    var new_left_indexer = groupsort_indexer(key_column1, left_indexer, left_count, max1.__int__())
    var new_right_indexer = groupsort_indexer(key_column2, right_indexer, right_count, max2.__int__())

    end_time = monotonic()
    print("sort time:", (end_time - start_time) / 1000000000)
    
    # var plan = build_column_plan(df1, df2, key_column)

    start_time = monotonic()

    # Use indexers to build DataFrame
    var col_data1 = List[Float64Array]()
    var col_names1 = List[String]()
    
    # [0,1,2,3,4] [1,1,2,2,3]
    col_names1 = df1.column_names
    
    var chunk_size = 64000

    var num_rows = left_indexer.size
    # var main_limit = (num_rows // 8) * 8


    for col_i in range (df1.column_names.__len__()):
        # var row_input_idx = 0
        var col = df1[col_i]
        var col_to_append = Float64Array(num_rows)

        var start = perf_counter()

        # for row_i in range(left_indexer.size):
        #     #print("get:", col[(left_indexer[row_i]).__int__()])
        #     col_to_append[row_i] = col[(left_indexer[row_i]).__int__()]
        
        
        var n_chunks = (num_rows + chunk_size - 1) // chunk_size

        # Copy a slice of rows using 4 SIMD registers

        @parameter
        fn worker(chunk_id: Int):
            # var start_i = chunk_id * chunk_size
            # var end_i = min(start_i + chunk_size, left_indexer.size)
            # var limit = ((end_i - start_i) // 8) * 8 + start_i

            # # Copy a slice of rows
            # for row_i in range(start_i, limit, 8):
            #     var matched_idxs = new_left_indexer.data.load[width=8](row_i)
            #     col_to_append.data.store[width=8](row_i, SIMD[DType.float64, 8](col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
            #                                           col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
            #                                           col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
            #                                           col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]))
            
            # for row_i in range(limit, end_i):
            #     var matched_idx = new_left_indexer[row_i].__int__()
            #     col_to_append[row_i] = col[matched_idx]


            var start_i = chunk_id * chunk_size
            var end_i = min(start_i + chunk_size, num_rows)
            var limit = ((end_i - start_i) // 8) * 8 + start_i 

            for row_i in range(start_i, limit, 8):
                var matched_idxs = new_left_indexer.data.load[width=8](row_i)

                col_to_append.data.store[width=8](row_i, SIMD[DType.float64, 8](
                    col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
                    col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
                    col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
                    col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
                ))

            # Handle remaining elements
            for row_i in range(limit, end_i):
                var matched_idx = new_left_indexer[row_i].__int__()
                col_to_append[row_i] = col[matched_idx]

                # Run parallel workers
        parallelize[worker](n_chunks, num_threads)

        var end = perf_counter()
        print()
        print("parallel fill time table 1:", end - start)

        col_data1.append(col_to_append)

    # var col_data2 = List[Int32Array]()
    # var col_names2 = List[String]()

    # [0,1,2,3,4] [1,1,2,2,3]

    for col_i in range (df2.column_names.__len__()):
        if df2.column_names[col_i] != key_column:
            col_names1.append(df2.column_names[col_i])
            # var row_input_idx = 0
            var col = df2[col_i]
            var col_to_append = Float64Array(num_rows)

            # for row_i in range(num_rows):
            #     #print("get:", col[(right_indexer[row_i]).__int__()])
            #     var matched_idx = new_right_indexer[row_i].__int__()
            #     col_to_append[row_i] = col[matched_idx.__int__()]
            #     # row_input_idx += 1
        
            var n_chunks = (num_rows + chunk_size - 1) // chunk_size

            @parameter
            fn worker2(chunk_id: Int):
                var start_i = chunk_id * chunk_size
                var end_i = min(start_i + chunk_size, num_rows)
                var limit = ((end_i - start_i) // 8) * 8 + start_i  # Process in chunks of 32 elements

                # Copy a slice of rows using 4 SIMD registers
                for row_i in range(start_i, limit, 8):  # Increment by 32
                    var matched_idxs = new_right_indexer.data.load[width=8](row_i)

                    col_to_append.data.store[width=8](row_i, SIMD[DType.float64, 8](
                        col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
                        col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
                        col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
                        col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
                    ))


                # Handle remaining elements
                for row_i in range(limit, end_i):
                    var matched_idx = new_right_indexer[row_i].__int__()
                    col_to_append[row_i] = col[matched_idx]

            parallelize[worker2](n_chunks, num_threads)
        
            col_data1.append(col_to_append)

    
    # var built_columns = fill_joined_columns_parallel(df1, df2, new_left_indexer, new_right_indexer, plan)

    end_time = monotonic()
    print("fill data time:", (end_time - start_time) / 1000000000)

    # return DataFrameF64(built_columns.columns, built_columns.names)

    return DataFrameF64(col_data1,col_names1)


# fn inner_join_f64(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameF64:
#     # slow runtime with factorization
#     var key_column1 = df1[key_column]
#     var key_column2 = df2[key_column]

#     var col1_size = key_column1.size
#     var col2_size = key_column2.size

#     var left_labels = Float64Array(key_column1.size)
#     var global_label = 1
#     var right_labels = Float64Array(key_column2.size)

#     var uniques = Dict[FloatKey, Int]()

#     var start_uniqe = perf_counter()

#     # var uniques = CompactDict[Int](capacity=4194304)
#     for i in range(col1_size):
#         var cur_key = FloatKey(key_column1[i])
#         var get_or_set_default = uniques.setdefault(cur_key, global_label)
#         global_label += 1
#         left_labels[i] = get_or_set_default


#     for i in range(col2_size):
#         var cur_key = FloatKey(key_column2[i])
#         var get_or_set_default = uniques.setdefault(cur_key, global_label)
#         global_label += 1
#         right_labels[i] = get_or_set_default
    
    
#     var end_unique = perf_counter()
#     print("unique time:", end_unique - start_uniqe)

#     var max1 = left_labels[0]
#     var max2 = right_labels[0]

#     # var max1 = find_max_in_parallel(key_column1)

#     # # Find max in parallel for key_column2
#     # var max2 = find_max_in_parallel(key_column2)
#     var max_limit1 = (left_labels.size // 8) * 8
#     var max_limit2 = (right_labels.size // 8) * 8

#     for i in range(0, max_limit1, 8):
#         var maxs = left_labels.data.load[width=8](i)
#         var temp_max = maxs.reduce_max()
#         max1 = max(max1, temp_max)
    
#     # take care of remaining elements for column 1
#     for i in range(max_limit1, left_labels.size):
#         max1 = max(max1, left_labels[i])

#     for i in range(0, max_limit2, 8):
#         var maxs = right_labels.data.load[width=8](i)
#         var temp_max = maxs.reduce_max()
#         max2 = max(max2, temp_max)
    
#     # take care of remaining elements for column 2
#     for i in range(max_limit2, right_labels.size):
#         max2 = max(max2, right_labels[i])

#     # var max1 = parallel_find_max(key_column1)
#     # var max2 = parallel_find_max(key_column2)
    
#     var max_groups = max(max1, max2).__int__()
#     print("max groups:", max_groups)
#     var count = 0
    
#     # print("max grps")
#     # print(max_groups)
#     # var left_count = List[Int]()
#     # var right_count = List[Int]()

#     start_time = monotonic()

#     var left_count = Int32Array(max_groups + 1)
#     var right_count = Int32Array(max_groups + 1)
#     # left_count.resize(max_groups + 1, 0)
#     # right_count.resize(max_groups + 1, 0)
    
#     # # First pass to count frequencies of each group/key
#     for i in range(left_labels.size):
#         left_count[left_labels[i].__int__()] += 1

#     for i in range(right_labels.size):
#         right_count[right_labels[i].__int__()] += 1

#     # var left_count = count_group_freq_parallel(key_column1, max_groups)
#     # var right_count = count_group_freq_parallel(key_column2, max_groups)

#     end_time = monotonic()
#     print("count time:", (end_time - start_time) / 1000000000)

#     # Determine how many combinations will result from this group in the output
#     # cartesion product
#     # if the group has entries in both columns

#     start_time = monotonic()

#     for i in range(1, max_groups+1):
#         var lc = (left_count[i].__int__())
#         var rc = (right_count[i].__int__())
#         if lc > 0 and rc > 0:
#             count += lc * rc

#     var left_indexer = Int32Array(count)
#     var right_indexer = Int32Array(count)

#     var left_pos = left_count[0]
#     var right_pos = right_count[0]
#     var position = 0

#     for i in range(1, max_groups+1):
#         var lc = (left_count[i].__int__())
#         var rc = (right_count[i].__int__())

#         if rc > 0 and lc > 0:
#             for j in range(lc):
               
#                 # Calculate the starting index in the result arrays
#                 # for group left[i] and all elements from right belonging to the same group.
#                 # 1,2,3      1,2,1,3,2
#                 # 0,0,1,1,2          0,1,2,3,4
#                 # Group 1 -> lc:1 rc:2

#                 # offset = 0 + 0 * 2
#                 var offset = position + j * rc
#                 # print("offset:", offset)
    
#                 for k in range(rc):
#                     # left_indexer[0 + 0] = 0 + 0
#                     # left_indexer[0 + 1] = 0 + 0
#                     left_indexer[offset + k] = left_pos + j
#                     # right_indexer[0 + 0] = 0 + 0
#                     # right_indexer[0 + 1] = 0 + 1
#                     right_indexer[offset + k] = right_pos + k
#             # update pointer to skip to next location after all combinations in current group
#             position += lc * rc

#         # skip to next group
#         left_pos += lc
#         right_pos += rc

#     end_time = monotonic()
#     print("indexer time:", (end_time - start_time) / 1000000000)

#     # print("left indexer:")
#     # for i in range(left_indexer.size):
#     #     print(left_indexer[i])

#     # print("right indexer:")
#     # for i in range(right_indexer.size):
#     #     print(right_indexer[i])

#     #var start_time = monotonic()
    
#     start_time = monotonic()

#     var new_left_indexer = groupsort_indexer(left_labels, left_indexer, left_count, max1.__int__())
#     var new_right_indexer = groupsort_indexer(right_labels, right_indexer, right_count, max2.__int__())

#     end_time = monotonic()
#     print("sort time:", (end_time - start_time) / 1000000000)
    
#     # var plan = build_column_plan(df1, df2, key_column)

#     start_time = monotonic()

#     # Use indexers to build DataFrame
#     var col_data1 = List[Float64Array]()
#     var col_names1 = List[String]()
    
#     # [0,1,2,3,4] [1,1,2,2,3]
#     col_names1 = df1.column_names
    
#     var chunk_size = 6400

#     var num_rows = left_indexer.size
#     # var main_limit = (num_rows // 8) * 8


#     for col_i in range (df1.column_names.__len__()):
#         # var row_input_idx = 0
#         var col = df1[col_i]
#         var col_to_append = Float64Array(num_rows)

#         var start = perf_counter()

#         # for row_i in range(left_indexer.size):
#         #     #print("get:", col[(left_indexer[row_i]).__int__()])
#         #     col_to_append[row_i] = col[(left_indexer[row_i]).__int__()]
        
        
#         var n_chunks = (num_rows + chunk_size - 1) // chunk_size

#         # Copy a slice of rows using 4 SIMD registers

#         @parameter
#         fn worker(chunk_id: Int):
#             # var start_i = chunk_id * chunk_size
#             # var end_i = min(start_i + chunk_size, left_indexer.size)
#             # var limit = ((end_i - start_i) // 8) * 8 + start_i

#             # # Copy a slice of rows
#             # for row_i in range(start_i, limit, 8):
#             #     var matched_idxs = new_left_indexer.data.load[width=8](row_i)
#             #     col_to_append.data.store[width=8](row_i, SIMD[DType.float64, 8](col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
#             #                                           col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
#             #                                           col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
#             #                                           col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]))
            
#             # for row_i in range(limit, end_i):
#             #     var matched_idx = new_left_indexer[row_i].__int__()
#             #     col_to_append[row_i] = col[matched_idx]


#             var start_i = chunk_id * chunk_size
#             var end_i = min(start_i + chunk_size, num_rows)
#             var limit = ((end_i - start_i) // 8) * 8 + start_i  # Process in chunks of 32 elements

#             # Copy a slice of rows using 4 SIMD registers
#             for row_i in range(start_i, limit, 8):
#                 var matched_idxs = new_left_indexer.data.load[width=8](row_i)

#                 col_to_append.data.store[width=8](row_i, SIMD[DType.float64, 8](
#                     col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
#                     col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
#                     col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
#                     col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
#                 ))

#             # Handle remaining elements
#             for row_i in range(limit, end_i):
#                 var matched_idx = new_left_indexer[row_i].__int__()
#                 col_to_append[row_i] = col[matched_idx]

#                 # Run parallel workers
#         parallelize[worker](n_chunks)

#         var end = perf_counter()
#         print()
#         print("parallel fill time table 1:", end - start)

#         col_data1.append(col_to_append)

#     # var col_data2 = List[Int32Array]()
#     # var col_names2 = List[String]()

#     # [0,1,2,3,4] [1,1,2,2,3]

#     for col_i in range (df2.column_names.__len__()):
#         if df2.column_names[col_i] != key_column:
#             col_names1.append(df2.column_names[col_i])
#             # var row_input_idx = 0
#             var col = df2[col_i]
#             var col_to_append = Float64Array(num_rows)

#             # for row_i in range(num_rows):
#             #     #print("get:", col[(right_indexer[row_i]).__int__()])
#             #     var matched_idx = new_right_indexer[row_i].__int__()
#             #     col_to_append[row_i] = col[matched_idx.__int__()]
#             #     # row_input_idx += 1
        
#             var n_chunks = (num_rows + chunk_size - 1) // chunk_size

#             @parameter
#             fn worker2(chunk_id: Int):
#                 var start_i = chunk_id * chunk_size
#                 var end_i = min(start_i + chunk_size, num_rows)
#                 var limit = ((end_i - start_i) // 8) * 8 + start_i  # Process in chunks of 32 elements

#                 # Copy a slice of rows using 4 SIMD registers
#                 for row_i in range(start_i, limit, 8):  # Increment by 32
#                     var matched_idxs = new_right_indexer.data.load[width=8](row_i)

#                     col_to_append.data.store[width=8](row_i, SIMD[DType.float64, 8](
#                         col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
#                         col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
#                         col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
#                         col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
#                     ))


#                 # Handle remaining elements
#                 for row_i in range(limit, end_i):
#                     var matched_idx = new_right_indexer[row_i].__int__()
#                     col_to_append[row_i] = col[matched_idx]

#             parallelize[worker2](n_chunks)
        
#             col_data1.append(col_to_append)

    
#     # var built_columns = fill_joined_columns_parallel(df1, df2, new_left_indexer, new_right_indexer, plan)

#     end_time = monotonic()
#     print("fill data time:", (end_time - start_time) / 1000000000)

#     # return DataFrameF64(built_columns.columns, built_columns.names)

#     return DataFrameF64(col_data1,col_names1)


fn parallel_find_max(read column: Float64Array) raises -> SIMD[DType.float64, 1]:
    alias simd_width = simdwidthof[DType.float64]()
    var size = column.size

    var chunk_size = (size + 4 - 1) // 4
    chunk_size = (chunk_size // simd_width) * simd_width
    print("chunk size:", chunk_size)
    var n_chunks = (size + chunk_size - 1) // chunk_size
    print("n_chunks:", n_chunks)

    # Prepare partial maxima array
    var partial_max = Float64Array(n_chunks)

    @parameter
    fn max_worker(thread_id: Int):
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, size)

        var local_max = column[start]

        # vectorized local max computation
        @parameter
        fn _max_chunk[width: Int](offset: Int):
            var col_data_slice = column.data.load[width=width](start + offset)
            var temp_max = col_data_slice.reduce_max()
            local_max = max(local_max, temp_max)

        var length = end - start
        # Run the vectorized pass
        vectorize[_max_chunk, simd_width](length)

        # leftover elements
        var leftover_start = start + (length // simd_width) * simd_width

        for i in range(leftover_start, end):
            local_max = max(local_max, column[i])

        # Store partial result
        partial_max[thread_id] = local_max

    # Launch threads
    parallelize[max_worker](n_chunks)

    var global_max = partial_max[0]
    for i in range(1, n_chunks):
        global_max = max(global_max, partial_max[i])

    return global_max



fn inner_join_f64_reindex(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameWithIndexers:
    # find the max number of distinct groups in both join columns
    var key_column1 = df1[key_column]
    var key_column2 = df2[key_column]

    var max1 = key_column1[0]
    var max2 = key_column2[0]

    for i in range(key_column1.size):
        max1 = max(max1, key_column1[i])

    for i in range(key_column2.size):
        max2 = max(max2, key_column2[i])
    
    var max_groups = max(max1, max2).__int__()
    var count = 0
    
    # var left_count = List[Int]()
    # var right_count = List[Int]()
    # left_count.resize(max_groups + 1, 0)
    # right_count.resize(max_groups + 1, 0)

    var left_count = Int32Array(max_groups + 1)
    var right_count = Int32Array(max_groups + 1)

    
    # # First pass to count frequencies of each group/key
    for i in range(key_column1.size):
        left_count[key_column1[i].__int__()] += 1

    for i in range(key_column2.size):
        right_count[key_column2[i].__int__()] += 1

    # Determine how many combinations will result from this group in the output
    # cartesion product
    # if the group has entries in both columns
    for i in range(1, max_groups+1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())
        if lc > 0 and rc > 0:
            count += lc * rc

    var left_indexer = Int32Array(count)
    var right_indexer = Int32Array(count)

    var left_pos = left_count[0]
    var right_pos = right_count[0]
    var position = 0

    for i in range(1, max_groups+1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())

        if rc > 0 and lc > 0:
            for j in range(lc):
               
                # Calculate the starting index in the result arrays
                # for group left[i] and all elements from right belonging to the same group.
                # 1,2,3      1,2,1,3,2
                # 0,0,1,1,2          0,1,2,3,4
                # Group 1 -> lc:1 rc:2

                # offset = 0 + 0 * 2
                var offset = position + j * rc
                # print("offset:", offset)
    
                for k in range(rc):
                    # left_indexer[0 + 0] = 0 + 0
                    # left_indexer[0 + 1] = 0 + 0
                    left_indexer[offset + k] = left_pos + j
                    # right_indexer[0 + 0] = 0 + 0
                    # right_indexer[0 + 1] = 0 + 1
                    right_indexer[offset + k] = right_pos + k
            # update pointer to skip to next location after all combinations in current group
            position += lc * rc

        # skip to next group
        left_pos += lc
        right_pos += rc
    
    # print("left indexer:")
    # for i in range(left_indexer.size):
    #     print(left_indexer[i])

    # print("right indexer:")
    # for i in range(right_indexer.size):
    #     print(right_indexer[i])

    var new_left_indexer = groupsort_indexer(key_column1, left_indexer, left_count, max1.__int__())
    var new_right_indexer = groupsort_indexer(key_column2, right_indexer, right_count, max2.__int__())

    # Use indexers to build DataFrame
    var col_data1 = List[Float64Array]()
    var col_names1 = List[String]()
    
    # [0,1,2,3,4] [1,1,2,2,3]
    col_names1 = df1.column_names
    
    for col_i in range (df1.column_names.__len__()):
        var row_input_idx = 0
        var col = df1[col_i]
        var col_to_append = Float64Array(left_indexer.size)

        for row_i in range(left_indexer.size):
            #print("get:", col[(left_indexer[row_i]).__int__()])
            var matched_idx = new_left_indexer[row_i].__int__()
            col_to_append[row_input_idx] = col[matched_idx.__int__()]
            row_input_idx += 1
        col_data1.append(col_to_append)

    # var col_data2 = List[Int32Array]()
    # var col_names2 = List[String]()

    # [0,1,2,3,4] [1,1,2,2,3]

    for col_i in range (df2.column_names.__len__()):
        if df2.column_names[col_i] != key_column:
            col_names1.append(df2.column_names[col_i])
            var row_input_idx = 0
            var col = df2[col_i]
            var col_to_append = Float64Array(right_indexer.size)

            for row_i in range(right_indexer.size):
                #print("get:", col[(right_indexer[row_i]).__int__()])
                var matched_idx = new_right_indexer[row_i].__int__()
                col_to_append[row_input_idx] = col[matched_idx.__int__()]
                row_input_idx += 1
            col_data1.append(col_to_append)
    
    return DataFrameWithIndexers(col_data1^, col_names1, List[Int32Array](new_left_indexer, new_right_indexer))

fn reindex_string_column(read original_col: List[String], read new_indexer: Int32Array) raises -> List[String]:
    var reindexed_string_col = List[String]()
    reindexed_string_col.resize(new_indexer.size, "")

    for i in range(new_indexer.size):
        reindexed_string_col[i] = original_col[new_indexer[i].__int__()]

    return reindexed_string_col


fn groupsort_indexer(mut index: Float64Array, mut indexer: Int32Array, mut counts: Int32Array, ngroups: Int) raises -> Int32Array:
    var n = index.size
    var start_time = monotonic()

    var sorter = Int32Array(n)
    var where = Int32Array(ngroups + 1)

    end_time = monotonic()
    print("init time:", (end_time - start_time) / 1000000000)

    start_time = monotonic()
    # mark the start of each contiguous group of like-indexed data
    for i in range(1, ngroups + 1):
        where[i] = where[i - 1] + counts[i - 1]

    end_time = monotonic()

    # print("where time:", (end_time - start_time) / 1000000000)

    # start_time = monotonic()

    # var chunk_size = 6400
    # var n_chunks   = (ngroups + chunk_size - 1) // chunk_size

    # var partial_sums = List[Int32](capacity=n_chunks)

    # # Pass 1: compute partial sums in parallel
    # @parameter
    # fn pass1_worker(chunk_id: Int):
    #     var start_i = chunk_id * chunk_size
    #     var end_i   = min(start_i + chunk_size, ngroups)

    #     var sum_local = SIMD[DType.int32, 1](0)
    #     for i in range(start_i, end_i):
    #         sum_local += counts[i]
    #     partial_sums[chunk_id] = sum_local

    # parallelize[pass1_worker](n_chunks, 6400)

    # # Build prefix of partial_sums sequentially
    # var running_offset = SIMD[DType.int32, 1](0)
    # for c in range(n_chunks):
    #     var tmp = partial_sums[c]
    #     partial_sums[c] = running_offset
    #     running_offset += tmp

    # # Pass 2: fill 'where' directly in parallel
    # @parameter
    # fn pass2_worker(chunk_id: Int):
    #     var start_i = chunk_id * chunk_size
    #     var end_i   = min(start_i + chunk_size, ngroups)

    #     var offset = partial_sums[chunk_id]
    #     var local_running = offset
    #     for i in range(start_i, end_i):
    #         local_running += Int(counts[i])
    #         where[i+1] = local_running

    # parallelize[pass2_worker](n_chunks, 6400)

    # end_time = monotonic()

    # print("where parallel time:", (end_time - start_time) / 1000000000)


    
    start_time = monotonic()
    # indexer
    for i in range(n):
        var label = index[i].__int__()
        sorter[(where[label]).__int__()] = i
        where[label] += 1

    end_time = monotonic()

    print("sorter time:", (end_time - start_time) / 1000000000)

    # print("sorter")
    # for i in range(sorter.size):
    #     print(sorter[i])

    start_time = monotonic()
    for i in range(indexer.size):
        indexer[i] = sorter[(indexer[i]).__int__()]
    
    end_time = monotonic()
    print("reindex time:", (end_time - start_time) / 1000000000)

    return indexer

fn count_group_freq_parallel(read key_column: Float64Array, max_groups: Int, chunk_size: Int = 64000) -> List[Int]:
    var n = key_column.size
    var n_chunks = (n + chunk_size - 1) // chunk_size
    # partial_freqs[chunk_id] = local freq array
    var partial_freqs = List[List[Int]]()
    partial_freqs.resize(n_chunks, List[Int]())

    @parameter
    fn worker(chunk_id: Int):
        var start_row = chunk_id * chunk_size
        var end_row = min(start_row + chunk_size, n)

        var local_freq = List[Int]()
        local_freq.resize(max_groups + 1, 0)

        for r in range(start_row, end_row):
            var g = key_column[r].__int__()
            local_freq[g] += 1

        partial_freqs[chunk_id] = local_freq

    # Build partial frequencies
    parallelize[worker](n_chunks, 2)

    # Merge partial frequencies
    var global_freq = List[Int]()
    global_freq.resize(max_groups + 1, 0)

    for chunk in range(n_chunks):
        var loc = partial_freqs[chunk]
        for g in range(max_groups + 1):
            global_freq[g] += loc[g]

    return global_freq

fn insertion_sort(mut arr: Float64Array, mut indices: List[Int], left: Int, right: Int):
    for i in range(left + 1, right):
        var value_to_place = indices[i]
        var j = i
        while j > left and (arr[value_to_place] < arr[indices[j - 1]]):
            indices[j] = indices[j - 1]
            j -= 1
        indices[j] = value_to_place
        

fn mergesort_helper(mut arr: Float64Array, mut indices: List[Int], left: Int, right: Int, mut temp: List[Int]):
    if (right - left) <= 32:  # SMALL_MERGESORT set to 10
        insertion_sort(arr, indices, left, right)
    else:
        var mid = (left + right) // 2
        mergesort_helper(arr, indices, left, mid, temp)
        mergesort_helper(arr, indices, mid, right, temp)

        for i in range(left, mid):
            temp[i] = indices[i]

        var i = left
        var j = left
        var k = mid

        # merge two sorted arrays, compare elements and merge in order
        while j < mid and k < right:
            if arr[temp[j]] <= arr[indices[k]]:
                indices[i] = temp[j]
                j += 1
            else:
                indices[i] = indices[k]
                k += 1
            i += 1

        # copy left over elements into the merge array
        while j < mid:
            indices[i] = temp[j]
            j += 1
            i += 1

fn find_max_in_parallel(read col_data: Float64Array) -> Float64:
    var n = col_data.size

    var chunk_size = 6400
    var n_chunks   = (n + chunk_size - 1) // chunk_size

    var partial_maxes = List[Float64](capacity=n_chunks)

    @parameter
    fn worker(chunk_id: Int):
        var start_i = chunk_id * chunk_size
        var end_i = min(start_i + chunk_size, n)

        var local_max = col_data[start_i]
        for i in range(start_i+1, end_i):
            local_max = max(local_max, col_data[i])

        partial_maxes[chunk_id] = local_max

    parallelize[worker](n_chunks, n_chunks)

    var global_max = partial_maxes[0]
    for c in range(1, n_chunks):
        global_max = max(global_max, partial_maxes[c])

    return global_max

@value
struct ColumnPlan(CollectionElement):
    var source_df_id: Int  
    var src_col_index: Int 
    var out_col_index: Int
    var name: String

    fn __init__(mut self, source_df_id: Int, src_col_index: Int, out_col_index: Int, name: String):
        self.source_df_id = source_df_id
        self.src_col_index = src_col_index
        self.out_col_index = out_col_index
        self.name = name

    fn __moveinit__(mut self, owned existing: Self):
        self.source_df_id = existing.source_df_id
        self.src_col_index = existing.src_col_index
        self.out_col_index = existing.out_col_index
        self.name = (existing.name)^
        
    fn __copyinit__(mut self, existing: Self):
        self.source_df_id = existing.source_df_id
        self.src_col_index = existing.src_col_index
        self.out_col_index = existing.out_col_index
        self.name = existing.name

struct ColumnBuildResult:
    var columns: List[Float64Array]
    var names: List[String]

    fn __init__(mut self, columns: List[Float64Array], names: List[String]):
        self.columns = columns
        self.names = names

fn build_column_plan(read df1: DataFrameF64, read df2: DataFrameF64, key_column: String) -> List[ColumnPlan]:
    var plan = List[ColumnPlan]()

    for i in range(df1.columns.size):
        var col_plan = ColumnPlan(0, i, plan.size, df1.column_names[i])
        plan.append(col_plan)

    for i in range(df2.columns.size):
        if df2.column_names[i] != key_column:
            var col_plan = ColumnPlan(1, i, plan.size, df2.column_names[i])
            plan.append(col_plan)

    return plan


fn fill_joined_columns_parallel(mut df1: DataFrameF64,mut df2: DataFrameF64,
                                read left_indexer: Int32Array, read right_indexer: Int32Array,
                                read plan: List[ColumnPlan]) raises -> ColumnBuildResult:
    """
    Build the final columns in parallel, each ColumnPlan is filled by a worker.
    We assume 'plan' is a list of all columns (from df1 + df2 except key) we want in the output.

    Returns (col_data1, col_names1).
    """

    var total_count = left_indexer.size  # or right_indexer.size, same length
    var num_cols = plan.size

    # final joined table columns
    var col_data = List[Float64Array]()
    col_data.resize(num_cols, Float64Array(0))

    # 2) We'll also gather column names in order
    var col_names = List[String]()
    col_names.resize(num_cols, "")

    # 3) The worker function that handles exactly one column from the plan
    #    If you want to chunk columns, you can do it differently, but typically
    #    "one worker per column" is easiest when the number of columns is large.
    @parameter
    fn fill_one_column(col_id: Int):
        var cp = plan[col_id]
        var chunk_size = 6400
        var n_chunks   = (total_count + chunk_size - 1) // chunk_size

        try:
            # Create the output array for this column
            var out_col = Float64Array(total_count)

            if cp.source_df_id == 0:
                # column from df1 => use left_indexer
                var src_col = df1[cp.src_col_index]
                # for row_i in range(total_count):
                #     var matched_idx = Int(left_indexer[row_i])
                #     out_col[row_i] = src_col[matched_idx]

                @parameter
                fn row_chunk_worker(chunk_id: Int):
                    var start_i = chunk_id * chunk_size
                    var end_i   = min(start_i + chunk_size, total_count)

                    for row_i in range(start_i, end_i):
                        var matched_idx = Int(left_indexer[row_i])
                        out_col[row_i] = src_col[matched_idx]

                # Parallelize row chunks for this column
                parallelize[row_chunk_worker](n_chunks, 16)
            else:
                # column from df2 => use right_indexer
                var src_col = df2[cp.src_col_index]
                # for row_i in range(total_count):
                #     var matched_idx = Int(right_indexer[row_i])
                #     out_col[row_i] = src_col[matched_idx]


                @parameter
                fn row_chunk_worker2(chunk_id: Int):
                    var start_i = chunk_id * chunk_size
                    var end_i   = min(start_i + chunk_size, total_count)

                    for row_i in range(start_i, end_i):
                        var matched_idx = Int(right_indexer[row_i])
                        out_col[row_i] = src_col[matched_idx]

                parallelize[row_chunk_worker2](n_chunks, 16)

            col_data[col_id] = out_col
            col_names[col_id] = cp.name
        except:
            pass

    # 4) Launch parallel tasks
    #    We'll have num_cols tasks, each filling one column.
    #    If num_cols < # of CPU cores, you might not saturate all cores,
    #    but this is straightforward to implement.
    parallelize[fill_one_column](num_cols, num_cols)

    return ColumnBuildResult(col_data, col_names)


def mergesort(mut arr: Float64Array, mut indices: List[Int]) -> List[Int]:
    var temp = List[Int](capacity=arr.size)

    mergesort_helper(arr, indices, 0, arr.size, temp)

    return indices

trait PredicateF64:
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool: ...

@value
struct EQPredF64(PredicateF64):
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool:
        # return (x == value_cmp) or isclose(x, value_cmp)
        return (x == value_cmp)
@value
struct NEQPredF64(PredicateF64):
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool:
        # return (x != value_cmp) and (isclose(x, value_cmp) == False)
        return (x != value_cmp)

@value
struct GTPredF64(PredicateF64):
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool:
        # return (not isclose(x, value_cmp)) and (x > value_cmp)
        return x > value_cmp
@value
struct GTEPredF64(PredicateF64):
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool:
        # return isclose(x, value_cmp) or (x > value_cmp)
        return x >= value_cmp
@value
struct LEPredF64(PredicateF64):
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool:
        # return isclose(x, value_cmp) or (x < value_cmp)
        return x <= value_cmp
@value
struct LTPredF64(PredicateF64):
    fn evaluate(self, x: SIMD[DType.float64, 1], value_cmp: SIMD[DType.float64, 1]) -> Bool:
        # return (not isclose(x, value_cmp)) and (x < value_cmp)
        # var p = x * x * x + value_cmp * value_cmp + 42.0
    
        # # 2) A couple of trig calls
        # var s = math.sin(x) + math.cos(value_cmp)
        
        # # 3) Combine them
        # var t = math.exp(s * p)  # exponent can be expensive

        # # 4) A contrived threshold check (just to yield a boolean)
        # #    e.g. if t is above some "random" threshold or something
        # if t > 100_000.0:
        #     return True
        # else:
        #     return False
        return x < value_cmp

struct DataFrameWithIndexers():
    var df: DataFrameF64
    var indexers: List[Int32Array]

    fn __init__(mut self, owned df_data: List[Float64Array], df_col_names: List[String], owned indexers: List[Int32Array]) raises:
        self.df = DataFrameF64(df_data, df_col_names)
        self.indexers = indexers

@value
struct IntKey(KeyElement):
    var i: SIMD[DType.int32, 1]

    fn __init__(mut self, owned i: SIMD[DType.int32, 1]):
        self.i = i

    fn __hash__(self) -> UInt:
        return hash(self.i)

    fn __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    fn __ne__(self, other: Self) -> Bool:
        return self.i != other.i


@value
struct DoubleTup(CollectionElement):
    var data: Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1]]

    fn __init__(mut self, data: Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1]]):
        self.data = Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1]](data[0], data[1])
    
    fn __moveinit__(mut self, owned existing: Self):
        self.data = (existing.data)^
        
    fn __copyinit__(mut self, existing: Self):
        self.data = Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1]](existing.data[0], existing.data[1])

@value
struct TripleTup(CollectionElement):
    var data: Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]

    fn __init__(mut self, data: Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]):
        self.data = Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]
                            (data[0], data[1], data[2])
    
    fn __moveinit__(mut self, owned existing: Self):
        self.data = (existing.data)^
        
    fn __copyinit__(mut self, existing: Self):
        self.data = Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]
                            (existing.data[0], existing.data[1], existing.data[2])

@value
struct QuadTup(CollectionElement):
    var data: Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]

    fn __init__(mut self, data: Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]):
        self.data = Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]
                            (data[0], data[1], data[2], data[3])
    
    fn __moveinit__(mut self, owned existing: Self):
        self.data = (existing.data)^
        
    fn __copyinit__(mut self, existing: Self):
        self.data = Tuple[SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1], SIMD[DType.float64, 1]]
                            (existing.data[0], existing.data[1], existing.data[2], existing.data[3])

@value
struct DoubleTupleKey(KeyElement):
    var i: DoubleTup

    fn __moveinit__(mut self, owned existing: Self):
        self.i = (existing.i)^

    fn __copyinit__(mut self, existing: Self):
       self.i = existing.i

    fn __hash__(self) -> UInt:
        # Python hashing for a tuple
        # https://github.com/python/cpython/blob/v3.7.0/Objects/tupleobject.c#L336-L369
        var hash_val = 0x345678
        var multiplier = 1000003
        var add_const = 82520

        # Compute the hash of individual element in the compound key tuple
        var compound_key0_hash = hash(self.i.data[0])
        hash_val = (hash_val ^ compound_key0_hash) * multiplier
        multiplier += (add_const + 0)

        var compound_key1_hash = hash(self.i.data[1])
        hash_val = (hash_val ^ compound_key1_hash) * multiplier
        multiplier += (add_const + 2)
    
        hash_val += 97531


        #### Define our own naive hash function to combine the values in the tuple into one value
        return hash_val
        #return hash(self.i.data[0] + ((10**6) * 1e-6) + self.i.data[1] + ((10**12) * 1e-6) + self.i.data[2])

    fn __eq__(self, other: Self) -> Bool:
        return (self.i.data[0] == other.i.data[0]) and (self.i.data[1] == other.i.data[1])

    fn __ne__(self, other: Self) -> Bool:
        return ((self.i.data[0] != other.i.data[0]) or (self.i.data[1] != other.i.data[1]))

@value
struct TupleKey(KeyElement):
    var i: TripleTup

    fn __moveinit__(mut self, owned existing: Self):
        self.i = (existing.i)^

    fn __copyinit__(mut self, existing: Self):
       self.i = existing.i

    fn __hash__(self) -> UInt:
        # Python hashing for a tuple
        # https://github.com/python/cpython/blob/v3.7.0/Objects/tupleobject.c#L336-L369
        var hash_val = 0x345678
        var multiplier = 1000003
        var add_const = 82520

        # Compute the hash of individual element in the compound key tuple
        var compound_key0_hash = hash(self.i.data[0])
        hash_val = (hash_val ^ compound_key0_hash) * multiplier
        multiplier += (add_const + 0)

        var compound_key1_hash = hash(self.i.data[1])
        hash_val = (hash_val ^ compound_key1_hash) * multiplier
        multiplier += (add_const + 2)

        var compound_key2_hash = hash(self.i.data[2])
        hash_val = (hash_val ^ compound_key2_hash) * multiplier
        multiplier += (add_const + 4)
    
        hash_val += 97531


        #### Define our own naive hash function to combine the values in the tuple into one value
        return hash_val
        #return hash(self.i.data[0] + ((10**6) * 1e-6) + self.i.data[1] + ((10**12) * 1e-6) + self.i.data[2])

    fn __eq__(self, other: Self) -> Bool:
        return (self.i.data[0] == other.i.data[0])
            and (self.i.data[1] == other.i.data[1])
            and (self.i.data[2] == other.i.data[2])

    fn __ne__(self, other: Self) -> Bool:
        return ((self.i.data[0] != other.i.data[0])
                    or (self.i.data[1] != other.i.data[1])
                    or (self.i.data[2] != other.i.data[2]))


@value
struct QuadTupleKey(KeyElement):
    var i: QuadTup

    fn __moveinit__(mut self, owned existing: Self):
        self.i = (existing.i)^

    fn __copyinit__(mut self, existing: Self):
       self.i = existing.i

    fn __hash__(self) -> UInt:
        # Python hashing for a tuple
        # https://github.com/python/cpython/blob/v3.7.0/Objects/tupleobject.c#L336-L369
        var hash_val = 0x345678
        var multiplier = 1000003
        var add_const = 82520

        # Compute the hash of individual element in the compound key tuple
        var compound_key0_hash = hash(self.i.data[0])
        hash_val = (hash_val ^ compound_key0_hash) * multiplier
        multiplier += (add_const + 0)

        var compound_key1_hash = hash(self.i.data[1])
        hash_val = (hash_val ^ compound_key1_hash) * multiplier
        multiplier += (add_const + 2)

        var compound_key2_hash = hash(self.i.data[2])
        hash_val = (hash_val ^ compound_key2_hash) * multiplier
        multiplier += (add_const + 4)

        var compound_key3_hash = hash(self.i.data[3])
        hash_val = (hash_val ^ compound_key3_hash) * multiplier
        multiplier += (add_const + 6)
    
        hash_val += 97531


        #### Define our own naive hash function to combine the values in the tuple into one value
        return hash_val
        #return hash(self.i.data[0] + ((10**6) * 1e-6) + self.i.data[1] + ((10**12) * 1e-6) + self.i.data[2])

    fn __eq__(self, other: Self) -> Bool:
        return (self.i.data[0] == other.i.data[0])
            and (self.i.data[1] == other.i.data[1])
            and (self.i.data[2] == other.i.data[2])
            and (self.i.data[3] == other.i.data[3])

    fn __ne__(self, other: Self) -> Bool:
        return ((self.i.data[0] != other.i.data[0])
                    or (self.i.data[1] != other.i.data[1])
                    or (self.i.data[2] != other.i.data[2])
                    or (self.i.data[3] != other.i.data[3]))

@value
struct FloatKey(KeyElement):
    var i: SIMD[DType.float64, 1]

    fn __init__(mut self, owned i: SIMD[DType.float64, 1]):
        self.i = i

    fn __hash__(self) -> UInt:
        return _hash_simd[DType.float64, 1](self.i)

    fn __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    fn __ne__(self, other: Self) -> Bool:
        return self.i != other.i

@value
struct FloatKeyCompact(Keyable):
    var data: Float64

    fn accept[T: KeysBuilder](self, mut keys_builder: T):
        keys_builder.add(self.data)
