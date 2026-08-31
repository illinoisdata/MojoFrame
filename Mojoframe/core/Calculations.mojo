from core.DataFrame import DataFrameF64, DataFrameF32, DataFrameI32
from core.Arrays import Float64Array, Float32Array, Int32Array
from std.collections import Dict, Set, List
from std.math import isclose, min, max
from std.time import monotonic, perf_counter
from std.sys.info import simd_width_of

comptime INSERTION_SORT_THRESHOLD = 32
comptime FLOAT_VEC_WIDTH = simd_width_of[DType.float64]()

comptime HASH_SEED: UInt = 0x345678
comptime HASH_MULTIPLIER_INIT: UInt = 1000003
comptime HASH_ADD_CONST: UInt = 82520
comptime HASH_FINAL_ADD: UInt = 97531

def parallelize[func: def(Int) capturing[_] -> None](num_work_items: Int, num_threads: Int = 1):
    for i in range(num_work_items):
        func(i)

def array_max_f64(arr: Float64Array) raises -> SIMD[DType.float64, 1]:
    var cur_max = arr[0]

    for i in range(1, arr.size):
        if arr[i] > cur_max:
            cur_max = arr[i]

    return cur_max

def element_mult_f64(mut arr1: Float64Array, mut arr2: Float64Array) raises -> Float64Array:
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
    
    return result_arr^

def pairwise_sum_f64(mut arr: Float64Array, n: Int, start: Int, stop: Int) -> SIMD[DType.float64, 1]:
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
                

def pairwise_sum_f32(mut arr: Float32Array, n: Int, start: Int, stop: Int) -> SIMD[DType.float32, 1]:
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

def pairwise_sum_i32(mut arr: Int32Array, n: Int, start: Int, stop: Int) -> SIMD[DType.int32, 1]:
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

def column_wise_mult_f64(mut arr1: Float64Array, mut arr2: Float64Array) raises -> Float64Array:
    var result_arr = Float64Array(arr1.size)
    for i in range(arr1.size):
        result_arr[i] = (arr1[i] * arr2[i])
    return result_arr^

def aggregation_sum_i32(mut columns: List[Int32Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Int32Array]:
    # Iterate through each row of the DF (stored in columnar format)
    # For the groupby column, group all the rows by the different keys

    # The groupby sum table in Pandas is a 2d array
    # Dict that maps group to sums takes 14s to run for 10M records
    # var groupby_table = Dict[Int, Int32Array]()
    var start_time = perf_counter()

    var groupby_table = List[Int32Array]()
    var groups_vec = List[IntKey]()
    var num_cols = len(columns)
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
    for i in range(len(groups_vec)):
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
    var end_time = perf_counter()
    print((end_time - start_time) / 1000000000)
    # After building the groupby dict, return the^ result as a DF
    # var summed_data = List[Int32Array]()
    # var group_vec_size = groups_vec.size
    # for i in range(group_vec_size):
    #     summed_data.append(groupby_table[groups_vec[i]])
    
    return groupby_table^

# def aggregation_sum_i32_alt(mut columns: List[Int32Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Int32Array]:
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


    # return columns^

def aggregation_sum_f64_multicol(mut df: DataFrameF64, groupby_cols: List[String]) raises -> List[Float64Array]:
    if len(groupby_cols) == 2:
        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[DoubleTupleKey, Int]()
       
        var groups_vec = List[DoubleTupleKey]()
        var num_cols = len(df.columns)
       
        
        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

        for i in range(len(groupby_cols)):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True

    
        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            # record row index for the corresponding key
            if not(compound_key in group_to_idx):
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                group_to_idx[compound_key] = 0
        

        for i in range(len(groups_vec)):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            key_to_index[i] = group_to_idx[compound_key]

        for _ in range(num_cols - len(groupby_cols)):
            groupby_table.append(Float64Array(len(groups_vec)))

        var agg_i = 0
        # for every col, there are rows belonging to a group, aggregate values for groups
        for col_i in range(num_cols):
            if not (df.column_names[col_i] in groupby_cols_dict):
                var cur_col = (df.columns[col_i])
                for row_i in range(key_col_len):
                    groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
                agg_i += 1
        
        var group_col1 = Float64Array(len(groups_vec))
        var group_col2 = Float64Array(len(groups_vec))
        
        for key_i in range(len(groups_vec)):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]

        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        return groupby_table^

    elif len(groupby_cols) == 3:
         ########## Incremental hashing approach, conceptually similar to Pandas, slower but more generalizable
        # var groupby_table = List[Float64Array]()
        # var group_to_idx = Dict[MultiFloatKeyIncremental, Int]()
        # var groups_vec = List[MultiFloatKeyIncremental]()
        # var num_cols = len(df.columns)
        
        # var groupby_cols_dict = Dict[String, Bool]()
      
        # var key_col_len = df[0].size

        # var key_to_index = List[Int]()
        # key_to_index.resize(key_col_len, 0)

        # var start_increment = perf_counter()

        # # list to hold the incrementally built key for each row
        # var row_keys = List[MultiFloatKeyIncremental](capacity=key_col_len)
  
        # for _ in range(key_col_len):
        #     row_keys.append(MultiFloatKeyIncremental(len(groupby_cols)))

        # var end_increment = perf_counter()
        # print("incremental key creation time: ", end_increment - start_increment)

        # var start_hash_update = perf_counter()
        # for i in range(len(groupby_cols)):
        #     var col = df[groupby_cols[i]]
        #     for row_i in range(key_col_len):
        #         row_keys[row_i].update(col[row_i])
        #     groupby_cols_dict[groupby_cols[i]] = True

        # var end_hash_update = perf_counter()
        # print("incremental key update time: ", end_hash_update - start_hash_update)

        # var start_hash_finalize = perf_counter()
        # for row_i in range(key_col_len):
        #     row_keys[row_i].finalize_hash()

        # var end_hash_finalize = perf_counter()
        # print("incremental key finalize time: ", end_hash_finalize - start_hash_finalize)

        # var start_dict = perf_counter()
        # for row_i in range(key_col_len):
        #     if not (row_keys[row_i] in group_to_idx):
        #         groups_vec.append(row_keys[row_i])
        #         group_to_idx[row_keys[row_i]] = 0
        # var end_dict = perf_counter()
        # print("dict creation time: ", end_dict - start_dict)
        
       
        # for i in range(len(groups_vec)):
        #     var cur_key = groups_vec[i]
        #     # map a compound key to an integer group
        #     group_to_idx[cur_key] = i
        
        # print("unique groups: ", len(groups_vec))

        
        # var start_key_to_index = perf_counter()
        # for i in range(key_col_len):
        #     key_to_index[i] = group_to_idx[row_keys[i]]
        # var end_key_to_index = perf_counter()
        # print("key to index time: ", end_key_to_index - start_key_to_index)

        # for _ in range(num_cols - len(groupby_cols)):
        #     groupby_table.append(Float64Array(len(groups_vec)))
        

        # var agg_time = perf_counter()
        # var agg_i = 0
        # # for every col, there are rows belonging to a group, aggregate values for groups
        # for col_i in range(num_cols):
        #     if not groupby_cols_dict.__contains__(df.column_names[col_i]):
        #         var cur_col = (df.columns[col_i])
        #         for row_i in range(key_col_len):
        #             groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
        #         agg_i += 1
        
        # var end_agg_time = perf_counter()
        # print("aggregation time: ", end_agg_time - agg_time)


        # for groupkey_idx in range(len(groupby_cols) - 1, -1, -1):
        #     var cur_out_key_col = Float64Array(len(groups_vec))
        #     for group_i in range(len(groups_vec)):
        #         var cur_key = groups_vec[group_i]
        #         cur_out_key_col[group_i] = cur_key.values[groupkey_idx]
        #     groupby_table.insert(0, cur_out_key_col)

        # return groupby_table^

        ########## TupleKey approach, optimized for small k-columns
        var keys_creation_start = perf_counter()

        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[TupleKey, Int]()
        var groups_vec = List[TupleKey]()
        var num_cols = len(df.columns)
        # append the columns of keys into a flat list
        # then create compound keys
        

        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

    
        for i in range(len(groupby_cols)):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True
        
       
        var next_group_id = 0

        for i in range(key_col_len):
            var compound_key = TupleKey(TripleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)])))
            # record row index for the corresponding key
            _ = -1
            if (compound_key in group_to_idx):
                group_id = group_to_idx[compound_key]
            else:
                group_id = next_group_id
                group_to_idx[compound_key] = group_id
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                next_group_id += 1
            key_to_index[i] = group_id

        var keys_creation_end = perf_counter()
        print("keys creation + mapping time : ", keys_creation_end - keys_creation_start)


        for _ in range(num_cols - len(groupby_cols)):
            groupby_table.append(Float64Array(len(groups_vec)))

      
        var start_agg = perf_counter()
       
        var agg_i = 0
        # for every col, there are rows belonging to a group, aggregate values for groups
        for col_i in range(num_cols):
            if not groupby_cols_dict.__contains__(df.column_names[col_i]):
                var cur_col = (df.columns[col_i])
                # if there is a compound key, there is a row where the values need to be aggregated
                for row_i in range(key_col_len):
                    groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
                agg_i += 1
        
        var group_col1 = Float64Array(len(groups_vec))
        var group_col2 = Float64Array(len(groups_vec))
        var group_col3 = Float64Array(len(groups_vec))
        for key_i in range(len(groups_vec)):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]
            group_col3[key_i] = cur_key.i.data[2]

        groupby_table.insert(0, group_col3)
        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        var end_agg = perf_counter()
        print("aggregation time: ", end_agg - start_agg)

        return groupby_table^
    
    elif len(groupby_cols) == 4:
        var groupby_table = List[Float64Array]()
        var group_to_idx = Dict[QuadTupleKey, Int]()
       
        var groups_vec = List[QuadTupleKey]()
        var num_cols = len(df.columns)
        
        var groupby_cols_dict = Dict[String, Bool]()
        var flat_keys = List[SIMD[DType.float64, 1]]()
        var key_col_len = df[0].size

        var key_to_index = List[Int]()
        key_to_index.resize(key_col_len, 0)

        for i in range(len(groupby_cols)):
            var col = df[groupby_cols[i]]
            for row_i in range(key_col_len):
                flat_keys.append(col[row_i])
            groupby_cols_dict[groupby_cols[i]] = True

    
        for i in range(key_col_len):
            var compound_key = QuadTupleKey(QuadTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)], flat_keys[i+(key_col_len*3)])))
            # record row index for the corresponding key
            if not (compound_key in group_to_idx):
                groups_vec.append(compound_key)
                # create the groups for the aggregated result
                group_to_idx[compound_key] = 0

        for i in range(len(groups_vec)):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = QuadTupleKey(QuadTup(Tuple(flat_keys[i], flat_keys[i+key_col_len], flat_keys[i+(key_col_len*2)], flat_keys[i+(key_col_len*3)])))
            key_to_index[i] = group_to_idx[compound_key]

        for _ in range(num_cols - len(groupby_cols)):
            groupby_table.append(Float64Array(len(groups_vec)))

        var agg_i = 0
        # for every col, there are rows belonging to a group, aggregate values for groups
        for col_i in range(num_cols):
            if not (df.column_names[col_i] in groupby_cols_dict):
                var cur_col = (df.columns[col_i])
                for row_i in range(key_col_len):
                    groupby_table[agg_i][key_to_index[row_i]] += (cur_col[row_i])
                agg_i += 1
        
        var group_col1 = Float64Array(len(groups_vec))
        var group_col2 = Float64Array(len(groups_vec))
        var group_col3 = Float64Array(len(groups_vec))
        var group_col4 = Float64Array(len(groups_vec))
        for key_i in range(len(groups_vec)):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]
            group_col3[key_i] = cur_key.i.data[2]
            group_col4[key_i] = cur_key.i.data[3]

        groupby_table.insert(0, group_col4)
        groupby_table.insert(0, group_col3)
        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        return groupby_table^

    return List[Float64Array]()


def aggregation_count_f64_multicol(mut df: DataFrameF64, groupby_cols: List[String]) raises -> List[Float64Array]:
    if len(groupby_cols) == 2:
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

        for i in range(len(groupby_cols)):
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

        for i in range(len(groups_vec)):
            # map a compound key to an integer group
            group_to_idx[groups_vec[i]] = i

        for i in range(key_col_len):
            var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
            key_to_index[i] = group_to_idx[compound_key]

        groupby_table.append(Float64Array(len(groups_vec)))

        for row_i in range(num_rows):
            var cur_row_group_idx = key_to_index[row_i]
            groupby_table[0][cur_row_group_idx] += 1

        
        var group_col1 = Float64Array(len(groups_vec))
        var group_col2 = Float64Array(len(groups_vec))
        
        for key_i in range(len(groups_vec)):
            var cur_key = groups_vec[key_i]
            group_col1[key_i] = cur_key.i.data[0]
            group_col2[key_i] = cur_key.i.data[1]

        groupby_table.insert(0, group_col2)
        groupby_table.insert(0, group_col1)

        return groupby_table^
    
    return List[Float64Array]()


def aggregation_all_f64_multicol(mut df: DataFrameF64, groupby_cols: List[String]) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var group_to_idx = Dict[DoubleTupleKey, Int]()
    # var key_to_index = Dict[DoubleTupleKey, List[Int]]()
   
    var groups_vec = List[DoubleTupleKey]()
    var num_cols_df = len(df.columns)
    # append the columns of keys into a flat list
    # then create compound keys
    # [1,2,3,8001,8002,8003,121,122,123]
    var num_cols = (len(df.columns) - len(groupby_cols)) * 2 + 1
    
    var groupby_cols_dict = Dict[String, Bool]()
    var flat_keys = List[SIMD[DType.float64, 1]]()
    var key_col_len = df[0].size
    
    # use to keep track of which group the current row belongs to
    var key_to_index = List[Int]()
    key_to_index.resize(key_col_len, 0)

    for i in range(len(groupby_cols)):
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
    print("num of double keys: ", len(groups_vec))
    
    # use to keep track of which group the current row belongs to
    var key_to_count = List[Int]()
    key_to_count.resize(len(groups_vec), 0)

    for i in range(len(groups_vec)):
        # map a compound key to an integer group
        group_to_idx[groups_vec[i]] = i

    for i in range(key_col_len):
        var compound_key = DoubleTupleKey(DoubleTup(Tuple(flat_keys[i], flat_keys[i+key_col_len])))
        key_to_index[i] = group_to_idx[compound_key]
        key_to_count[group_to_idx[compound_key]] += 1

    for _ in range(num_cols):
        groupby_table.append(Float64Array(len(groups_vec)))

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
            
            for group in range(len(groups_vec)):
                groupby_table[agg_i + num_cols_df - len(groupby_cols)][group] = (groupby_table[agg_i][group]) / Float64(key_to_count[group])
            
            agg_i += 1

    var group_col1 = Float64Array(len(groups_vec))
    var group_col2 = Float64Array(len(groups_vec))
    # record count for each group
    for group_index in range(len(groups_vec)):
        var cur_key = groups_vec[group_index]
        group_col1[group_index] = cur_key.i.data[0]
        group_col2[group_index] = cur_key.i.data[1]

        groupby_table[len(groupby_table) - 1][group_index] = Float64(key_to_count[group_index])

    groupby_table.insert(0, group_col2)
    groupby_table.insert(0, group_col1)
    # print("groups vec:", groups_vec.size)

    return groupby_table^

def aggregation_sum_f64_parallel(mut columns: List[Float64Array],
                                col_names: List[String],
                                groupby_col_idx: Int,
                                chunk_size: Int = 640) raises -> List[Float64Array]:

    # 1) Build the group->index mapping (same as single-threaded)
    var num_cols = len(columns)
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]

    var group_to_idx = Dict[FloatKey, Int]()
    var groups_vec = List[FloatKey]()

    for row_i in range(num_rows):
        var g_val = FloatKey(group_by_col[row_i])
        if not (g_val in group_to_idx):
            groups_vec.append(g_val)
            group_to_idx[g_val] = 0

    for i in range(len(groups_vec)):
        group_to_idx[groups_vec[i]] = i

    var total_groups = len(groups_vec)

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
    for chunk_id in range(n_chunks):
        var start_row = chunk_id * chunk_size
        var end_row = min(start_row + chunk_size, num_rows)

        # Create local aggregator (num_cols-1) columns
        var local_agg = List[Float64Array]()
        for _ in range(num_cols - 1):
            local_agg.append(Float64Array(total_groups))

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

        partials[chunk_id] = local_agg^

    # 4) Launch parallel tasks

    # 5) Merge partials into final_table
    for chunk_id in range(n_chunks):
        var partial_agg = partials[chunk_id].copy()
        for c in range(len(partial_agg)):
            var partial_col = partial_agg[c]
            # var final_col   = final_table[c]
            for g in range(total_groups):
                final_table[c][g] += partial_col[g]

    # 6) Optionally insert the group-key column at the front
    var group_col = Float64Array(total_groups)
    for key_i in range(total_groups):
        group_col[key_i] = groups_vec[key_i].i
    final_table.insert(0, group_col)

    return final_table^

def aggregation_sum_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = len(columns)
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
    for i in range(len(groups_vec)):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(len(groups_vec)))

    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                # l_extendedprice * (1 - l_discount) sum this for each row if this condition is given
                groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            agg_i += 1
    
    var group_col = Float64Array(len(groups_vec))
   
    for key_i in range(len(groups_vec)):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table^

def aggregation_sum_conditional_f64(mut columns: List[Float64Array], col_names: List[String], mask: List[Bool], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = len(columns)
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
    for i in range(len(groups_vec)):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(len(groups_vec)))

    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                if mask[row_i]:
                    groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            agg_i += 1
    
    var group_col = Float64Array(len(groups_vec))
   
    for key_i in range(len(groups_vec)):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table^

def aggregation_min_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = len(columns)
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()
    var neg_inf = -Float64.MAX_FINITE

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
    for i in range(len(groups_vec)):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(len(groups_vec), True))

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
    
    var group_col = Float64Array(len(groups_vec))
   
    for key_i in range(len(groups_vec)):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table^

def aggregation_count_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
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
    for i in range(len(groups_vec)):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))


    groupby_table.append(Float64Array(len(groups_vec)))
    
    for row_i in range(num_rows):
        var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
        groupby_table[0][cur_row_group_idx] += 1
    
    var group_col = Float64Array(len(groups_vec))
   
    for key_i in range(len(groups_vec)):
        # var cur_key = groups_vec[key_i]
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table^

def aggregation_count_distinct_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int, aggregation_col_idx: Int) raises -> List[Float64Array]:
    var start_time = perf_counter()

    var groupby_table = List[Float64Array]()
    # var groups_vec = List[SIMD[DType.float64, 1]]()
    
    
    var group_by_col = columns[groupby_col_idx]
    var aggregation_col = columns[aggregation_col_idx]

    var num_rows = group_by_col.size

    var groups_vec = List[FloatKey](capacity=num_rows)
    var group_to_idx = Dict[FloatKey, Int](capacity=4194304)
    # var existing_groups = Set[FloatKey]()


    var end_time = perf_counter()
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

    end_time = perf_counter()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("groups_vec creation time:", execution_time_seconds)
    #return groupby_table^
    # table
    #    selected_col count unique
    # 0
    # 1
    # 2
    # Map groups to index like 0, 1, 2
    for i in range(len(groups_vec)):
        # print(groups_vec[i])
        group_to_idx[groups_vec[i]] = i
        # var cur_group_distinct = List[Bool]()
        # cur_group_distinct.resize(max_value_in_elements.__int__() + 1, False)
        # group_to_distinct_elements.append(cur_group_distinct)

    group_to_distinct_elements.resize(len(groups_vec), SetElement())

    print("group_vec size:", len(groups_vec))
    end_time = perf_counter()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("table creation time:", execution_time_seconds)

    
    # groupby_table.append(Float64Array(groups_vec.size))

    var distinct_count_array = Float64Array(len(groups_vec))


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

    
    # end_time = perf_counter()
    # execution_time_nanoseconds = end_time - start_time
    # execution_time_seconds = execution_time_nanoseconds / 1000000000

    # print("partial sets time:", execution_time_seconds)


    # # def parallel_count_distinct(thread_id: Int):
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
    # for chunk_id in range(num_work_items): parallel_count_distinct(chunk_id)

    # # merge partial results for 1st worker
    # var local_sets_worker1 = partial_sets[0]
    # var local_sets_worker2 = partial_sets[1]
    # for group_i in range(groups_vec.size):
    #     group_to_distinct_elements[group_i].distinct_elements.update(local_sets_worker1[group_i].distinct_elements)
    #     group_to_distinct_elements[group_i].distinct_elements.update(local_sets_worker2[group_i].distinct_elements)
    #     distinct_count_array[group_i] = group_to_distinct_elements[group_i].distinct_elements.__len__()
        

    # # def parallel_count_distinct(thread_id: Int):
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
    
    # for chunk_id in range(num_work_items): parallel_count_distinct(chunk_id)

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
    
    end_time = perf_counter()
    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("distinct count time:", execution_time_seconds)
    
    var group_col = Float64Array(len(groups_vec))
   
    for key_i in range(len(groups_vec)):
        distinct_count_array[key_i] = Float64(len(group_to_distinct_elements[key_i].distinct_elements))
        group_col[key_i] = groups_vec[key_i].i

    # groupby_table.insert(0, group_col)
    # groupby_table.insert(0, group_col)

    groupby_table.append(group_col)
    groupby_table.append(distinct_count_array)

    return groupby_table^

def aggregation_all_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]() 
    # sum, avg two agg methods, then one column for groups and one for counts
    var num_cols_df = len(columns)
    var num_cols = (len(columns) - 1) * 2 + 2
    var num_rows = columns[groupby_col_idx].size
    var group_by_col = columns[groupby_col_idx]
    var group_to_idx = Dict[FloatKey, Int]()

    for row_i in range(num_rows):
        # Each row in the groupby column belongs to a group
        var cur_row_group = FloatKey(group_by_col[row_i])
        if not (cur_row_group in group_to_idx):
            groups_vec.append(cur_row_group)
            group_to_idx[cur_row_group] = 0
    
    var num_groups = len(groups_vec)
    var group_count = List[Int]()
    # # Map groups to index like 0, 1, 2
    # print(groups_vec.size)
    for i in range(len(groups_vec)):
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
            groupby_table[agg_i][group] = sums[group] / Float64(group_count)
        agg_i += 1
    
    for group in range(num_groups):
        var group_count = group_count[group] / (num_cols_df - 1)
        groupby_table[num_cols - 1][group] = Float64(group_count)
        groupby_table[0][group] = groups_vec[group].i

    return groupby_table^

def aggregation_mean_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:

    var groupby_table = List[Float64Array]()
    var groups_vec = List[FloatKey]()
    var num_cols = len(columns)
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
    group_count.resize(len(groups_vec), 0)

    # # Map groups to index like 0, 1, 2
    for i in range(len(groups_vec)):
        group_to_idx[groups_vec[i]] = i
        # groupby_table.append(Float64Array(num_cols - 1))
            
    for i in range(group_by_col.size):
        group_count[group_to_idx[FloatKey(group_by_col[i])]] += 1

    for _ in range(num_cols - 1):
        groupby_table.append(Float64Array(len(groups_vec)))

    
    var agg_i = 0
    
    for col_i in range(num_cols):
        if col_i != groupby_col_idx:
            var cur_col = columns[col_i]
            for row_i in range(num_rows):
                var cur_row_group_idx = group_to_idx[FloatKey(group_by_col[row_i])]
                groupby_table[agg_i][cur_row_group_idx] += cur_col[row_i]
            
            for group in range(len(groups_vec)):
                groupby_table[agg_i][group] = (groupby_table[agg_i][group]) / Float64(group_count[group])

            agg_i += 1

    var group_col = Float64Array(len(groups_vec))
   
    for key_i in range(len(groups_vec)):
        group_col[key_i] = groups_vec[key_i].i

    groupby_table.insert(0, group_col)

    return groupby_table^


# def aggregation_sum_f64(mut columns: List[Float64Array], col_names: List[String], groupby_col_idx: Int) raises -> List[Float64Array]:
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

#     return groupby_table^

def combine_masks(masks: List[List[Bool]], logical_operator: String) raises -> List[Bool]:
        var filtered_mask = masks[0].copy()
        var mask_len = len(filtered_mask)

        if logical_operator == "AND":
            for i in range(1, len(masks)):
                var cur_mask = masks[i].copy()
                for j in range(mask_len):
                    filtered_mask[j] = (filtered_mask[j] and cur_mask[j])
        elif logical_operator == "OR":
            for i in range(1, len(masks)):
                var cur_mask = masks[i].copy()
                for j in range(mask_len):
                    filtered_mask[j] = (filtered_mask[j] or cur_mask[j])
        
        return filtered_mask^

def filter_string_equal(mut df: DataFrameF64, str_col: List[String], filter_str: String) raises:
    var selected_indices = List[Int]()
    for i in range(df.columns[0].size):
        if str_col[i] == filter_str:
            selected_indices.append(i)

    var filtered_data = List[Float64Array]()
    for col_i in range(len(df.columns)):
        var col_to_fill = Float64Array(len(selected_indices))
        var original_col = df.columns[col_i]
        for row_i in range(len(selected_indices)):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data^

def filter_string_equal_mask(str_col: List[String], filter_str: String) ->  List[Bool]:
    var n = len(str_col)
    # var chunk_size = 64000
    var num_work_items = 8
    var chunk_size = (n + num_work_items - 1) // num_work_items

    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(n, False)
   
    for chunk_id in range(num_work_items):
        var start = chunk_id * chunk_size
        var end = min(start + chunk_size, n)
        for i in range(start, end):
            if str_col[i] == filter_str:
                selected_indices_mask[i] = True


    return selected_indices_mask^

def filter_string_not_equal_mask(str_col: List[String], filter_str: String) ->  List[Bool]:
    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(len(str_col), False)

    for i in range(len(str_col)):
        if str_col[i] != filter_str:
            selected_indices_mask[i] = True

    return selected_indices_mask^

def filter_string_contains(mut df: DataFrameF64, str_col: List[String], filter_str: String) raises:
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
        
    # df.columns = filtered_data^

    var n = df.columns[0].size
    var chunk_size = 640000
    var n_chunks   = (n + chunk_size - 1) // chunk_size

    # Each chunk will store matching row indices in its own list
    var partial_lists = List[List[Int]]()
    partial_lists.resize(n_chunks, List[Int]())

    for chunk_id in range(n_chunks):
        var start_i = chunk_id * chunk_size
        var end_i = min(start_i + chunk_size, n)

        var local_matches = List[Int]()

        for i in range(start_i, end_i):
            if filter_str in str_col[i]:
                local_matches.append(i)

        partial_lists[chunk_id] = local_matches^

    # Launch parallel tasks to find matching rows

    var selected_indices = List[Int]()
    for c in range(n_chunks):
        var local_matches = partial_lists[c].copy()
        for idx in range(len(local_matches)):
            selected_indices.append(local_matches[idx])


    var filtered_data = List[Float64Array]()
    for col_i in range(len(df.columns)):
        var col_to_fill = Float64Array(len(selected_indices))
        var original_col = df.columns[col_i]
        for row_i in range(len(selected_indices)):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data^
    

def filter_string_endwith(mut df: DataFrameF64, str_col: List[String], filter_str: String) raises:
    var selected_indices = List[Int]()
    for i in range(df.columns[0].size):
        if str_col[i].endswith(filter_str):
            selected_indices.append(i)

    var filtered_data = List[Float64Array]()
    for col_i in range(len(df.columns)):
        var col_to_fill = Float64Array(len(selected_indices))
        var original_col = df.columns[col_i]
        for row_i in range(len(selected_indices)):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data^

def filter_string_startwith(mut df: DataFrameF64, str_col: List[String], filter_str: String) raises:
    var selected_indices = List[Int]()
    # var n = df.columns[0].size

    # var chunk_size = 64000

    # var n_chunks = (n + chunk_size - 1) // chunk_size

    # var partial_results = List[List[Int]]()
    # partial_results.resize(n_chunks, List[Int]())


    # # def filter_worker(chunk_id: Int):
    #     var start_i = chunk_id * chunk_size
    #     var end_i = min(start_i + chunk_size, n)

    #     var local_list = List[Int]()

    #     for i in range(start_i, end_i):
    #         if str_col[i].startswith(filter_str):
    #             local_list.append(i)

    #     partial_results[chunk_id] = local_list^

    # # 6) Launch parallel tasks (each chunk in its own task)
    # for chunk_id in range(n_chunks): filter_worker(chunk_id)

    # for chunk_id in range(n_chunks):
    #     var local_list = partial_results[chunk_id]
    #     for idx in range(local_list.size):
    #         selected_indices.append(local_list[idx])

    for i in range(df.columns[0].size):
        if str_col[i].startswith(filter_str):
            selected_indices.append(i)

    var filtered_data = List[Float64Array]()
    for col_i in range(len(df.columns)):
        var col_to_fill = Float64Array(len(selected_indices))
        var original_col = df.columns[col_i]
        for row_i in range(len(selected_indices)):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data^

def filter_string_not_startwith_mask(str_col: List[String], filter_str: String) -> List[Bool]:
    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(len(str_col), False)

    for i in range(len(str_col)):
        if not (str_col[i].startswith(filter_str)):
            selected_indices_mask[i] = True

    return selected_indices_mask^

def filter_f64_IN_mask(float_col: Float64Array, filter_list: Float64Array) -> List[Bool]:
    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(float_col.size, False)

    var float_list = Set[FloatKey]()

    for i in range(filter_list.size):
        float_list.add(FloatKey(filter_list[i]))

    for i in range(float_col.size):
        if FloatKey(float_col[i]) in float_list:
            selected_indices_mask[i] = True

    return selected_indices_mask^

def filter_string_IN_mask(string_col: List[String], filter_list: List[String]) -> List[Bool]:
    var n = len(string_col)
    var num_work_items = 8
    # var chunk_size = 64000
    var chunk_size = (n + num_work_items - 1) // num_work_items

    var selected_indices_mask = List[Bool]()
    selected_indices_mask.resize(len(string_col), False)

    var string_list = Set[String]()

    for i in range(len(filter_list)):
        string_list.add(filter_list[i])

    # for i in range(string_col.size):
    #     if string_col[i] in string_list:
    #         selected_indices_mask[i] = True

    # 4) Per-thread function
    for chunk_id in range(num_work_items):
        var start = chunk_id * chunk_size
        var end = min(start + chunk_size, n)
        for i in range(start, end):
            if string_col[i] in string_list:
                selected_indices_mask[i] = True


    return selected_indices_mask^

def filter_not_string_exists_before(mut df: DataFrameF64, str_col: List[String], filter_str1: String, filter_str2: String) raises:
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
    var chunk_size = 640000
    var n_chunks   = (n + chunk_size - 1) // chunk_size

    # Each chunk collects matching indices in local lists
    var partial_results = List[List[Int]]()
    partial_results.resize(n_chunks, List[Int]())

    for chunk_id in range(n_chunks):
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

        partial_results[chunk_id] = local_list^

    # Launch parallel filtering

    # Merge partial results
    var selected_indices = List[Int]()
    for c in range(n_chunks):
        var local_list = partial_results[c].copy()
        for idx in range(len(local_list)):
            selected_indices.append(local_list[idx])

    var filtered_data = List[Float64Array]()
    for col_i in range(len(df.columns)):
        var col_to_fill = Float64Array(len(selected_indices))
        var original_col = df.columns[col_i]
        for row_i in range(len(selected_indices)):
            col_to_fill[row_i] = original_col[selected_indices[row_i]]
        filtered_data.append(col_to_fill)
        
    df.columns = filtered_data^


def cast_as_float64(str_col: List[String], substr_start_index: Int, substr_end_index: Int) raises -> Float64Array:
    var float_col = Float64Array(len(str_col))

    for i in range(len(str_col)):
        var substr = str_col[i][byte=substr_start_index:substr_end_index]
        float_col[i] = Float64(substr)

    return float_col^


def evaluate_query6[T: PredicateF64, T2: PredicateF64,
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

    return vector_of_index^

def evaluate_f64_alt[T: PredicateF64, T2: PredicateF64](column_1: Float64Array, column_2: Float64Array,
                      predicate_1: T, predicate_2: T2,
                      value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                      logical_operator: String) raises -> Int32Array:
    
    var n = column_1.size
    var total_count = 0

    if logical_operator == "AND":
        var start_filter = perf_counter()

        var chunk_size = 640000
        var n_chunks   = (n + chunk_size - 1) // chunk_size

        # We'll store partial results in one list per chunk
        var partial_results = List[List[Int]]()
        partial_results.resize(n_chunks, List[Int]())

        var partial_count = Int32Array(n_chunks)
        # var start = perf_counter()
        for chunk_id in range(n_chunks):
            var start_i = chunk_id * chunk_size
            var end_i   = min(start_i + chunk_size, n)

            var local_list = List[Int]()
            var local_count = 0
            for row_i in range(start_i, end_i):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
                    local_list.append(row_i)
                    local_count += 1

            partial_results[chunk_id] = local_list^
            partial_count[chunk_id] = Int32(local_count)

        
        for c in range(n_chunks):
            total_count += (partial_count[c].__int__())
        
        var filtered_idxs = Int32Array(total_count)
        var i = 0
        for c in range(n_chunks):
            var local_list = partial_results[c].copy()
            for idx in range(len(local_list)):
                filtered_idxs[i] = Int32(local_list[idx])
                i += 1
        
        var end_filter = perf_counter()
        print("AND filter time:", end_filter - start_filter)
        
        return filtered_idxs^
    elif logical_operator == "OR":
        var chunk_size = 640000
        var n_chunks   = (n + chunk_size - 1) // chunk_size

        # We'll store partial results in one list per chunk
        var partial_results = List[List[Int]]()
        partial_results.resize(n_chunks, List[Int]())

        var partial_count = Int32Array(n_chunks)
        # var start = perf_counter()
        for chunk_id in range(n_chunks):
            var start_i = chunk_id * chunk_size
            var end_i   = min(start_i + chunk_size, n)

            var local_list = List[Int]()
            var local_count = 0
            for row_i in range(start_i, end_i):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1) or predicate_2.evaluate(column_2[row_i], value_cmp_2):
                    local_list.append(row_i)
                    local_count += 1

            partial_results[chunk_id] = local_list^
            partial_count[chunk_id] = Int32(local_count)

        
        for c in range(n_chunks):
            total_count += (partial_count[c].__int__())
        
        var filtered_idxs = Int32Array(total_count)
        var i = 0
        for c in range(n_chunks):
            var local_list = partial_results[c].copy()
            for idx in range(len(local_list)):
                filtered_idxs[i] = Int32(local_list[idx])
                i += 1
        
        return filtered_idxs^
    else:
        if logical_operator == "COL":
            #var start = perf_counter()
            var chunk_size = 640000
            var n_chunks   = (n + chunk_size - 1) // chunk_size

            # We'll store partial results in one list per chunk
            var partial_results = List[List[Int]]()
            partial_results.resize(n_chunks, List[Int]())

            var partial_count = Int32Array(n_chunks)
            # var start = perf_counter()
            for chunk_id in range(n_chunks):
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

                partial_results[chunk_id] = local_list^
                partial_count[chunk_id] = Int32(local_count)


            
            for c in range(n_chunks):
                total_count += (partial_count[c].__int__())
            
            var filtered_idxs = Int32Array(total_count)
            var i = 0
            for c in range(n_chunks):
                var local_list = partial_results[c].copy()
                for idx in range(len(local_list)):
                    filtered_idxs[i] = Int32(local_list[idx])
                    i += 1
            
            #var end = perf_counter()
        
            return filtered_idxs^
        else:
            #var start = perf_counter()
            var chunk_size = 640000
            var n_chunks   = (n + chunk_size - 1) // chunk_size

            # We'll store partial results in one list per chunk
            var partial_results = List[List[Int]]()
            partial_results.resize(n_chunks, List[Int]())

            var partial_count = Int32Array(n_chunks)
            # var start = perf_counter()
            for chunk_id in range(n_chunks):
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

                partial_results[chunk_id] = local_list^
                partial_count[chunk_id] = Int32(local_count)


            
            for c in range(n_chunks):
                total_count += (partial_count[c].__int__())
            
            var filtered_idxs = Int32Array(total_count)
            var i = 0
            for c in range(n_chunks):
                var local_list = partial_results[c].copy()
                for idx in range(len(local_list)):
                    filtered_idxs[i] = Int32(local_list[idx])
                    i += 1
            
            #var end = perf_counter()
        
            return filtered_idxs^


def evaluate_f64[T: PredicateF64, T2: PredicateF64](column_1: Float64Array, column_2: Float64Array,
                      predicate_1: T, predicate_2: T2,
                      value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                      logical_operator: String) raises -> List[Int]:
    
    var n = column_1.size
    var vector_of_index = List[Int]()

    if logical_operator == "AND":
        var chunk_size = 640000
        var n_chunks   = (n + chunk_size - 1) // chunk_size

        # We'll store partial results in one list per chunk
        var partial_results = List[List[Int]]()
        partial_results.resize(n_chunks, List[Int]())

        # var start = perf_counter()
        for chunk_id in range(n_chunks):
            var start_i = chunk_id * chunk_size
            var end_i   = min(start_i + chunk_size, n)

            var local_list = List[Int]()

            # initialize local idxs to have (end_i - start_i) elements
            # SIMD load column data
            for row_i in range(start_i, end_i):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
                    local_list.append(row_i)

            partial_results[chunk_id] = local_list^


        # var end = perf_counter()
        # print("Time taken to filter parallel:", end - start)
        # for row_i in range(column_1.size):
        #     if predicate_1.evaluate(column_1[row_i], value_cmp_1) and predicate_2.evaluate(column_2[row_i], value_cmp_2):
        #         vector_of_index.append(row_i)

        for c in range(n_chunks):
            var local_list = partial_results[c].copy()
            for idx in range(len(local_list)):
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

            # # def worker2(chunk_id: Int):
            #     var start_i = chunk_id * chunk_size
            #     var end_i   = min(start_i + chunk_size, n)

            #     var local_list = List[Int]()

            #     for row_i in range(start_i, end_i):
            #         if predicate_1.evaluate(column_1[row_i], column_2[row_i]):
            #             local_list.append(row_i)

            #     partial_results[chunk_id] = local_list^

            # for chunk_id in range(n_chunks): worker2(chunk_id)
            
            # # # var i = 0
            # for c in range(n_chunks):
            #     var local_list = partial_results[c].copy()
            #     for idx in range(local_list.size):
            #         vector_of_index.append(local_list[idx])

        else:
            var start = perf_counter()
            for row_i in range(column_1.size):
                if predicate_1.evaluate(column_1[row_i], value_cmp_1):
                    vector_of_index.append(row_i)
            var end = perf_counter()
            print("Time taken to filter parallel:", end - start)
    
    return vector_of_index^

def evaluate_f64_mask[T: PredicateF64, T2: PredicateF64](column_1: Float64Array, column_2: Float64Array,
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
    
    return index_mask^

def evaluate_f32(column_1: Float32Array, column_2: Float32Array,
                      value_cmp_1: SIMD[DType.float32, 1], value_cmp_2: SIMD[DType.float32, 1]) raises -> List[Int]:
    var vector_of_index = List[Int]()
    
    for row_i in range(column_1.size):
        if column_1[row_i] > value_cmp_1 and column_2[row_i] <= value_cmp_2:
            vector_of_index.append(row_i)
    
    return vector_of_index^

def evaluate_i32(mut column: Int32Array, operation: String, value_cmp: SIMD[DType.int32, 1]) raises -> List[Int]:
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
    
    return vector^

# def inner_join_i32_naive(mut df1: DataFrameI32, mut df2: DataFrameI32, key_column: String) raises:
    
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

def inner_join_i32(mut df1: DataFrameI32, mut df2: DataFrameI32, key_column: String) raises -> DataFrameI32:
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

    var left_pos = left_count[0].__int__()
    var right_pos = right_count[0].__int__()
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
                    left_indexer[offset + k] = Int32(left_pos + j)
                    # right_indexer[0 + 0] = 0 + 0
                    # right_indexer[0 + 1] = 0 + 1
                    right_indexer[offset + k] = Int32(right_pos + k)
            # update pointer to skip to next location after all combinations in current group
            position += lc * rc

        # skip to next group
        left_pos += lc
        right_pos += rc
    

    # Use indexers to build DataFrame
    var col_data1 = List[Int32Array]()
    
    # [0,1,2,3,4] [1,1,2,2,3]
    col_names1 = df1.column_names.copy()
    
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

def left_join_f64(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameF64:
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

    var running_left_pos = left_count[0].__int__()   
    var running_right_pos = right_count[0].__int__()


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
                    left_indexer[offset + k] = Int32(running_left_pos + j)
                    right_indexer[offset + k] = Int32(running_right_pos + k)
            position += lc * rc
        else:
            # no matching right rows -> need lc rows, right_indexer = -1
            for j in range(lc):
                left_indexer[position + j] = Int32(running_left_pos + j)
                right_indexer[position + j] = -1

            position += lc

        running_left_pos += lc
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
    
    col_names = df1.column_names.copy()

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
                    out_col[row_input_idx] = -Float64.MAX_FINITE
                else:
                    out_col[row_input_idx] = col[potential_matched_idx]
                row_input_idx += 1
            
            col_data.append(out_col)

    return DataFrameF64(col_data, col_names)

def inner_join_f64(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameF64:
        var start_time = perf_counter()

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
        var maxs = key_column1.load[8](i)
        var temp_max = maxs.reduce_max()
        max1 = max(max1, temp_max)
    
    # take care of remaining elements for column 1
    for i in range(max_limit1, key_column1.size):
        max1 = max(max1, key_column1[i])

    for i in range(0, max_limit2, 8):
        var maxs = key_column2.load[8](i)
        var temp_max = maxs.reduce_max()
        max2 = max(max2, temp_max)
    
    # take care of remaining elements for column 2
    for i in range(max_limit2, key_column2.size):
        max2 = max(max2, key_column2[i])

    # var max1 = parallel_find_max(key_column1)
    # var max2 = parallel_find_max(key_column2)
    
    var max_groups = max(max1, max2).__int__()
    
    var count = 0
    
    end_time = perf_counter()

    print("find max time: ", (end_time - start_time) / 1000000000)

    # print("max grps")
    # print(max_groups)
    # var left_count = List[Int]()
    # var right_count = List[Int]()

    start_time = perf_counter()

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

    end_time = perf_counter()
    print("count time:", (end_time - start_time) / 1000000000)

    # Determine how many combinations will result from this group in the output
    # cartesion product
    # if the group has entries in both columns

    start_time = perf_counter()

    for i in range(1, max_groups+1):
        var lc = (left_count[i].__int__())
        var rc = (right_count[i].__int__())
        if lc > 0 and rc > 0:
            count += lc * rc

    var left_indexer = Int32Array(count)
    var right_indexer = Int32Array(count)

    var left_pos = left_count[0].__int__()
    var right_pos = right_count[0].__int__()
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
                    left_indexer[offset + k] = Int32(left_pos + j)
                    # right_indexer[0 + 0] = 0 + 0
                    # right_indexer[0 + 1] = 0 + 1
                    right_indexer[offset + k] = Int32(right_pos + k)
            # update pointer to skip to next location after all combinations in current group
            position += lc * rc

        # skip to next group
        left_pos += lc
        right_pos += rc

    end_time = perf_counter()
    print("indexer time:", (end_time - start_time) / 1000000000)

    # print("left indexer:")
    # for i in range(left_indexer.size):
    #     print(left_indexer[i])

    # print("right indexer:")
    # for i in range(right_indexer.size):
    #     print(right_indexer[i])

    #var start_time = perf_counter()
    
    start_time = perf_counter()

    var new_left_indexer = groupsort_indexer(key_column1, left_indexer, left_count, max1.__int__())
    var new_right_indexer = groupsort_indexer(key_column2, right_indexer, right_count, max2.__int__())

    end_time = perf_counter()
    print("sort time:", (end_time - start_time) / 1000000000)
    
    # var plan = build_column_plan(df1, df2, key_column)

    start_time = perf_counter()

    # Use indexers to build DataFrame
    var col_data1 = List[Float64Array]()
        
    # [0,1,2,3,4] [1,1,2,2,3]
    col_names1 = df1.column_names.copy()
    
    var chunk_size = 640000

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

        for chunk_id in range(n_chunks):
            # var start_i = chunk_id * chunk_size
            # var end_i = min(start_i + chunk_size, left_indexer.size)
            # var limit = ((end_i - start_i) // 8) * 8 + start_i

            # # Copy a slice of rows
            # for row_i in range(start_i, limit, 8):
            #     var matched_idxs = new_left_indexer.load[8](row_i)
            #     col_to_append.store[8](row_i, SIMD[DType.float64, 8](col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
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
                var matched_idxs = new_left_indexer.load[8](row_i)

                col_to_append.store[8](row_i, SIMD[DType.float64, 8](
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

            for chunk_id in range(n_chunks):
                var start_i = chunk_id * chunk_size
                var end_i = min(start_i + chunk_size, num_rows)
                var limit = ((end_i - start_i) // 8) * 8 + start_i  # Process in chunks of 32 elements

                # Copy a slice of rows using 4 SIMD registers
                for row_i in range(start_i, limit, 8):  # Increment by 32
                    var matched_idxs = new_right_indexer.load[8](row_i)

                    col_to_append.store[8](row_i, SIMD[DType.float64, 8](
                        col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
                        col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
                        col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
                        col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
                    ))


                # Handle remaining elements
                for row_i in range(limit, end_i):
                    var matched_idx = new_right_indexer[row_i].__int__()
                    col_to_append[row_i] = col[matched_idx]

        
            col_data1.append(col_to_append)

    
    # var built_columns = fill_joined_columns_parallel(df1, df2, new_left_indexer, new_right_indexer, plan)

    end_time = perf_counter()
    print("fill data time:", (end_time - start_time) / 1000000000)

    # return DataFrameF64(built_columns.columns, built_columns.names)

    return DataFrameF64(col_data1,col_names1)

def inner_join_sort_merge_f64(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String, chunk_size: Int=640000, num_threads: Int=8) raises -> DataFrameF64:

    var start_sort = perf_counter()

    var key_column1 = df1[key_column]
    var key_column2 = df2[key_column]

    var n1 = key_column1.size
    var n2 = key_column2.size

    var sorted_left_idx = parallel_argsort_f64(key_column1, num_threads)
    var sorted_right_idx = parallel_argsort_f64(key_column2, num_threads)

    var end_sort = perf_counter()
    print("sort time:", end_sort - start_sort)
    # pointer for sorted left idx
    var i = 0
    var k = 0

    var final_left_idx = List[Int]()
    var final_right_idx = List[Int]()

    while i < n1 and k < n2:
        var original_i = sorted_left_idx[i].__int__()
        var original_k = sorted_right_idx[k].__int__()

        var original_left_key = key_column1[original_i]
        var original_right_key = key_column2[original_k]

        if original_left_key < original_right_key:
            # didn't find a match, move pointer in left key col
            i += 1 
        elif original_left_key > original_right_key:
            # didn't find a match, move pointer in right key col
            k += 1 
        else: 
            # find all the matching keys in both columns
            # find the segment where the matching keys are
            var i_start = i
            var i_end = i + 1
            while i_end < n1 and key_column1[sorted_left_idx[i_end].__int__()] == original_left_key:
                i_end += 1

            var k_start = k
            var k_end = k + 1
            while k_end < n2 and key_column2[sorted_right_idx[k_end].__int__()] == original_right_key:
                k_end += 1

            # create combinations of the matching keys
            for cur_i_idx in range(i_start, i_end):
                # var current_original_i = sorted_left_idx[cur_i_idx]
                for cur_k_idx in range(k_start, k_end):
                    # var current_original_k = sorted_right_idx[cur_k_idx]
                    final_left_idx.append(sorted_left_idx[cur_i_idx].__int__())
                    final_right_idx.append(sorted_right_idx[cur_k_idx].__int__())

            # move left and right pointers to next matching key group
            i = i_end
            k = k_end

   

    print(len(final_left_idx))
    print(len(final_right_idx))

    var num_final_rows = len(final_left_idx) 

    var final_left_indices = Int32Array(num_final_rows)
    var final_right_indices = Int32Array(num_final_rows)

    for idx in range(num_final_rows):
        final_left_indices[idx] = Int32(final_left_idx[idx])
        final_right_indices[idx] = Int32(final_right_idx[idx])

   

    # var plan = build_column_plan(df1, df2, key_column)

    # start_time = perf_counter()

    # # Use indexers to build DataFrame
    var col_data1 = List[Float64Array]()
    var col_names1 = List[String]((df1.column_names)^)
    
    # # [0,1,2,3,4] [1,1,2,2,3]
    # col_names1 = df1.column_names.copy()
    
    # var chunk_size = 640000

    # var main_limit = (num_rows // 8) * 8

    var start = perf_counter()
    
    for col_i in range (df1.column_names.__len__()):
        
        # var row_input_idx = 0
        var col = df1[col_i]
        var col_to_append = Float64Array(num_final_rows)

        # var start = perf_counter()

        # for row_i in range(left_indexer.size):
        #     #print("get:", col[(left_indexer[row_i]).__int__()])
        #     col_to_append[row_i] = col[(left_indexer[row_i]).__int__()]
        
        
        var n_chunks = (num_final_rows + chunk_size - 1) // chunk_size

        # Copy a slice of rows using 4 SIMD registers

        for chunk_id in range(n_chunks):
            # var start_i = chunk_id * chunk_size
            # var end_i = min(start_i + chunk_size, left_indexer.size)
            # var limit = ((end_i - start_i) // 8) * 8 + start_i

            # # Copy a slice of rows
            # for row_i in range(start_i, limit, 8):
            #     var matched_idxs = new_left_indexer.load[8](row_i)
            #     col_to_append.store[8](row_i, SIMD[DType.float64, 8](col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
            #                                           col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
            #                                           col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
            #                                           col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]))
            
            # for row_i in range(limit, end_i):
            #     var matched_idx = new_left_indexer[row_i].__int__()
            #     col_to_append[row_i] = col[matched_idx]


            var start_i = chunk_id * chunk_size
            var end_i = min(start_i + chunk_size, num_final_rows)
            var limit = ((end_i - start_i) // FLOAT_VEC_WIDTH) * FLOAT_VEC_WIDTH + start_i 

            for row_i in range(start_i, limit, FLOAT_VEC_WIDTH):
                var matched_idxs = final_left_indices.load[FLOAT_VEC_WIDTH](row_i)
                var values = SIMD[DType.float64, FLOAT_VEC_WIDTH]()

                for k in range(FLOAT_VEC_WIDTH):
                    values[k] = col[matched_idxs[k].__int__()]

                col_to_append.store[FLOAT_VEC_WIDTH](row_i, values)
                # col_to_append.store[8](row_i, SIMD[DType.float64, 8](
                #     col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
                #     col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
                #     col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
                #     col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
                # ))

            # handle remaining elements
            for row_i in range(limit, end_i):
                var matched_idx = final_left_indices[row_i].__int__()
                col_to_append[row_i] = col[matched_idx]


        col_data1.append(col_to_append)
    
    var end = perf_counter()
    print("parallel fill time table 1:", end - start)

    # var col_data2 = List[Int32Array]()
    # var col_names2 = List[String]()

    # [0,1,2,3,4] [1,1,2,2,3]

    var start2 = perf_counter()

    for col_i in range (df2.column_names.__len__()):
        if df2.column_names[col_i] != key_column:
            col_names1.append(df2.column_names[col_i])
            # var row_input_idx = 0
            var col = df2[col_i]
            var col_to_append = Float64Array(num_final_rows)

            # for row_i in range(num_rows):
            #     #print("get:", col[(right_indexer[row_i]).__int__()])
            #     var matched_idx = new_right_indexer[row_i].__int__()
            #     col_to_append[row_i] = col[matched_idx.__int__()]
            #     # row_input_idx += 1
        
            var n_chunks = (num_final_rows + chunk_size - 1) // chunk_size

            for chunk_id in range(n_chunks):
                var start_i = chunk_id * chunk_size
                var end_i = min(start_i + chunk_size, num_final_rows)
                var limit = ((end_i - start_i) // FLOAT_VEC_WIDTH) * FLOAT_VEC_WIDTH + start_i

                for row_i in range(start_i, limit, FLOAT_VEC_WIDTH):
                    var matched_idxs = final_right_indices.load[FLOAT_VEC_WIDTH](row_i)

                    var values = SIMD[DType.float64, FLOAT_VEC_WIDTH]()

                    for k in range(FLOAT_VEC_WIDTH):
                        values[k] = col[matched_idxs[k].__int__()]

                    col_to_append.store[FLOAT_VEC_WIDTH](row_i, values)
                    # col_to_append.store[8](row_i, SIMD[DType.float64, 8](
                    #     col[matched_idxs[0].__int__()], col[matched_idxs[1].__int__()],
                    #     col[matched_idxs[2].__int__()], col[matched_idxs[3].__int__()],
                    #     col[matched_idxs[4].__int__()], col[matched_idxs[5].__int__()],
                    #     col[matched_idxs[6].__int__()], col[matched_idxs[7].__int__()]
                    # ))


                # Handle remaining elements
                for row_i in range(limit, end_i):
                    var matched_idx = final_right_indices[row_i].__int__()
                    col_to_append[row_i] = col[matched_idx]

        
            col_data1.append(col_to_append)
    var end2 = perf_counter()
    print("parallel fill time table 2:", end2 - start2)
    
    # # var built_columns = fill_joined_columns_parallel(df1, df2, new_left_indexer, new_right_indexer, plan)

    # end_time = perf_counter()
    # print("fill data time:", (end_time - start_time) / 1000000000)

    # return DataFrameF64(built_columns.columns, built_columns.names)

    return DataFrameF64(col_data1,col_names1)



def parallel_find_max(column: Float64Array) raises -> SIMD[DType.float64, 1]:
    comptime simd_width = simd_width_of[DType.float64]()
    var size = column.size

    var chunk_size = (size + 4 - 1) // 4
    chunk_size = (chunk_size // simd_width) * simd_width
    print("chunk size:", chunk_size)
    var n_chunks = (size + chunk_size - 1) // chunk_size
    print("n_chunks:", n_chunks)

    # Prepare partial maxima array
    var partial_max = Float64Array(n_chunks)

    for thread_id in range(n_chunks):
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, size)
        var local_max = column[start]
        for i in range(start + 1, end):
            local_max = max(local_max, column[i])
        partial_max[thread_id] = local_max

    var global_max = partial_max[0]
    for i in range(1, n_chunks):
        global_max = max(global_max, partial_max[i])

    return global_max



def inner_join_f64_reindex(mut df1: DataFrameF64, mut df2: DataFrameF64, key_column: String) raises -> DataFrameWithIndexers:
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

    var left_pos = left_count[0].__int__()
    var right_pos = right_count[0].__int__()
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
                    left_indexer[offset + k] = Int32(left_pos + j)
                    # right_indexer[0 + 0] = 0 + 0
                    # right_indexer[0 + 1] = 0 + 1
                    right_indexer[offset + k] = Int32(right_pos + k)
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
        
    # [0,1,2,3,4] [1,1,2,2,3]
    col_names1 = df1.column_names.copy()
    
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
    
    return DataFrameWithIndexers(col_data1^, col_names1, [new_left_indexer, new_right_indexer])

def reindex_string_column(original_col: List[String], new_indexer: Int32Array) raises -> List[String]:
    var reindexed_string_col = List[String]()
    reindexed_string_col.resize(new_indexer.size, "")

    for i in range(new_indexer.size):
        reindexed_string_col[i] = original_col[new_indexer[i].__int__()]

    return reindexed_string_col^


def groupsort_indexer(mut index: Float64Array, mut indexer: Int32Array, mut counts: Int32Array, ngroups: Int) raises -> Int32Array:
    var n = index.size
    var start_time = perf_counter()
    var end_time = perf_counter()

    var sorter = Int32Array(n)
    var where = Int32Array(ngroups + 1)

    end_time = perf_counter()
    print("init time:", (end_time - start_time) / 1000000000)

    start_time = perf_counter()
    # mark the start of each contiguous group of like-indexed data
    for i in range(1, ngroups + 1):
        where[i] = where[i - 1] + counts[i - 1]

    end_time = perf_counter()

    start_time = perf_counter()
    # indexer
    for i in range(n):
        var label = index[i].__int__()
        sorter[(where[label]).__int__()] = Int32(i)
        where[label] += 1

    end_time = perf_counter()

    print("sorter time:", (end_time - start_time) / 1000000000)

    start_time = perf_counter()
    for i in range(indexer.size):
        indexer[i] = sorter[(indexer[i]).__int__()]
    
    end_time = perf_counter()
    print("reindex time:", (end_time - start_time) / 1000000000)

    return indexer.copy()

# def count_group_freq_parallel(key_column: Float64Array, max_groups: Int, chunk_size: Int = 64000) -> List[Int]:
#     var n = key_column.size
#     var n_chunks = (n + chunk_size - 1) // chunk_size
#     # partial_freqs[chunk_id] = local freq array
#     var partial_freqs = List[List[Int]]()
#     partial_freqs.resize(n_chunks, List[Int]())

#     #     def worker(chunk_id: Int):
#         var start_row = chunk_id * chunk_size
#         var end_row = min(start_row + chunk_size, n)

#         var local_freq = List[Int]()
#         local_freq.resize(max_groups + 1, 0)

#         for r in range(start_row, end_row):
#             var g = key_column[r].__int__()
#             local_freq[g] += 1

#         partial_freqs[chunk_id] = local_freq

#     # Build partial frequencies
#     for chunk_id in range(n_chunks): worker(chunk_id)

#     # Merge partial frequencies
#     var global_freq = List[Int]()
#     global_freq.resize(max_groups + 1, 0)

#     for chunk in range(n_chunks):
#         var loc = partial_freqs[chunk]
#         for g in range(max_groups + 1):
#             global_freq[g] += loc[g]

#     return global_freq^

def insertion_sort(mut arr: Float64Array, mut indices: List[Int], left: Int, right: Int):
    for i in range(left + 1, right):
        var value_to_place = indices[i]
        var j = i
        while j > left and (arr[value_to_place] < arr[indices[j - 1]]):
            indices[j] = indices[j - 1]
            j -= 1
        indices[j] = value_to_place

def insertion_sort_tensor(mut arr: Float64Array, mut indices: Int32Array, left: Int, right: Int):
    for i in range(left + 1, right):
        var value_to_place = indices[i].__int__()
        var j = i
        while j > left and (arr[value_to_place] < arr[indices[j - 1].__int__()]):
            indices[j] = indices[j - 1]
            j -= 1
        indices[j] = Int32(value_to_place)
        

def mergesort_helper(mut arr: Float64Array, mut indices: List[Int], left: Int, right: Int, mut temp: List[Int]):
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

@always_inline
def merge_argsort(arr: Float64Array, mut indices: Int32Array, left: Int, mid: Int, right: Int, mut temp: Int32Array):
    if arr[Int(indices[mid - 1])] <= arr[Int(indices[mid])]:
        return
    var len_left = mid - left
    for i in range(len_left):
        temp[i] = indices[left + i]

    var i = left
    var j = 0
    var k = mid

    while j < len_left and k < right:
        if arr[Int(temp[j])] <= arr[Int(indices[k])]:
            indices[i] = temp[j]
            j += 1
        else:
            indices[i] = indices[k]
            k += 1
        i += 1

    while j < len_left:
        indices[i] = temp[j]
        j += 1
        i += 1

def parallel_argsort_f64(mut arr: Float64Array, num_threads: Int = 4) raises -> Int32Array:
    var n = arr.size
    
    var start_indices = perf_counter()
    # print(num_threads)
    # var indices = List[Int](capacity=n)
    var indices = Int32Array(n)
    for i in range(n):
        indices[i] = Int32(i)
    
    var end_indices = perf_counter()
    print("indices init time:", end_indices - start_indices)

    # if n <= INSERTION_SORT_THRESHOLD:
    #     insertion_sort_argsort(arr, indices, 0, n)
    #     return indices.copy()

    var start_group_sort = perf_counter()
    # parallel sort for chunks using insertion sort
    var groups_count = (n + INSERTION_SORT_THRESHOLD - 1) // INSERTION_SORT_THRESHOLD

    for group_idx in range(groups_count):
        var start = group_idx * INSERTION_SORT_THRESHOLD
        var end = min(start + INSERTION_SORT_THRESHOLD, n)
        insertion_sort_tensor(arr, indices, start, end)

    var end_group_sort = perf_counter()
    print("group sort time:", end_group_sort - start_group_sort)

    var start_merge = perf_counter()

    # parallel merge iterations
    var size = INSERTION_SORT_THRESHOLD
    # var temp_storage_size = (n + 1) // 2
    # var temp_indices = List[Int]()
    # temp_indices.resize(temp_storage_size, 0)

    while size < n:
        var current_merge_size = 2 * size
        var chunks_count = (n + current_merge_size - 1) // current_merge_size

        for chunk_idx in range(chunks_count):
            var start = chunk_idx * current_merge_size
            var mid = min(n, start + size)
            var end = min(n, start + current_merge_size)
            if mid < end:
                var temp_len = mid - start
                var local_temp_indices = Int32Array(temp_len)
                merge_argsort(arr, indices, start, mid, end, local_temp_indices)

        size *= 2
    
    var end_merge = perf_counter()
    print("merge time:", end_merge - start_merge)

    return indices.copy()


# def find_max_in_parallel(col_data: Float64Array) -> Float64:
#     var n = col_data.size

#     var chunk_size = 6400
#     var n_chunks   = (n + chunk_size - 1) // chunk_size

#     var partial_maxes = List[Float64](capacity=n_chunks)

#     #     def worker(chunk_id: Int):
#         var start_i = chunk_id * chunk_size
#         var end_i = min(start_i + chunk_size, n)

#         var local_max = col_data[start_i]
#         for i in range(start_i+1, end_i):
#             local_max = max(local_max, col_data[i])

#         partial_maxes[chunk_id] = local_max

#     for chunk_id in range(n_chunks): worker(chunk_id)

#     var global_max = partial_maxes[0]
#     for c in range(1, n_chunks):
#         global_max = max(global_max, partial_maxes[c])

#     return global_max



struct ColumnPlan(ImplicitlyCopyable, Copyable, Movable):
    var source_df_id: Int  
    var src_col_index: Int 
    var out_col_index: Int
    var name: String

    def __init__(out self, source_df_id: Int, src_col_index: Int, out_col_index: Int, name: String):
        self.source_df_id = source_df_id
        self.src_col_index = src_col_index
        self.out_col_index = out_col_index
        self.name = name

    def __init__(out self, *, deinit move: Self):
        self.source_df_id = move.source_df_id
        self.src_col_index = move.src_col_index
        self.out_col_index = move.out_col_index
        self.name = move.name^

    def __init__(out self, *, copy: Self):
        self.source_df_id = copy.source_df_id
        self.src_col_index = copy.src_col_index
        self.out_col_index = copy.out_col_index
        self.name = copy.name


def mergesort(mut arr: Float64Array, mut indices: List[Int]) -> List[Int]:
    var temp = List[Int](capacity=arr.size)
    mergesort_helper(arr, indices, 0, arr.size, temp)
    return indices.copy()


trait PredicateF64:
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool: ...


struct EQPredF64(PredicateF64, ImplicitlyCopyable, Copyable, Movable):
    def __init__(out self): pass
    def __init__(out self, *, copy: Self): pass
    def __init__(out self, *, deinit move: Self): pass
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool:
        return x == value_cmp


struct NEQPredF64(PredicateF64, ImplicitlyCopyable, Copyable, Movable):
    def __init__(out self): pass
    def __init__(out self, *, copy: Self): pass
    def __init__(out self, *, deinit move: Self): pass
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool:
        return x != value_cmp


struct GTPredF64(PredicateF64, ImplicitlyCopyable, Copyable, Movable):
    def __init__(out self): pass
    def __init__(out self, *, copy: Self): pass
    def __init__(out self, *, deinit move: Self): pass
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool:
        return x > value_cmp


struct GTEPredF64(PredicateF64, ImplicitlyCopyable, Copyable, Movable):
    def __init__(out self): pass
    def __init__(out self, *, copy: Self): pass
    def __init__(out self, *, deinit move: Self): pass
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool:
        return x >= value_cmp


struct LEPredF64(PredicateF64, ImplicitlyCopyable, Copyable, Movable):
    def __init__(out self): pass
    def __init__(out self, *, copy: Self): pass
    def __init__(out self, *, deinit move: Self): pass
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool:
        return x <= value_cmp


struct LTPredF64(PredicateF64, ImplicitlyCopyable, Copyable, Movable):
    def __init__(out self): pass
    def __init__(out self, *, copy: Self): pass
    def __init__(out self, *, deinit move: Self): pass
    def evaluate(self, x: Float64, value_cmp: Float64) -> Bool:
        return x < value_cmp


struct DataFrameWithIndexers(ImplicitlyCopyable, Copyable, Movable):
    var df: DataFrameF64
    var indexers: List[Int32Array]

    def __init__(out self, var df_data: List[Float64Array], df_col_names: List[String], var indexers: List[Int32Array]) raises:
        self.df = DataFrameF64(df_data, df_col_names)
        self.indexers = indexers.copy()

    def __init__(out self, *, copy: Self):
        self.df = copy.df
        self.indexers = copy.indexers.copy()

    def __init__(out self, *, deinit move: Self):
        self.df = move.df
        self.indexers = move.indexers^


struct IntKey(ImplicitlyCopyable, Copyable, Movable, Hashable, Equatable):
    var i: Int32

    def __init__(out self, i: Int32):
        self.i = i

    def __init__(out self, *, copy: Self):
        self.i = copy.i

    def __init__(out self, *, deinit move: Self):
        self.i = move.i

    def __hash__(self) -> UInt:
        return UInt(hash(self.i))

    def __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    def __ne__(self, other: Self) -> Bool:
        return self.i != other.i


struct DoubleTup(ImplicitlyCopyable, Copyable, Movable, Equatable):
    var data: Tuple[Float64, Float64]

    def __init__(out self, data: Tuple[Float64, Float64]):
        self.data = (data[0], data[1])

    def __init__(out self, *, deinit move: Self):
        self.data = move.data

    def __init__(out self, *, copy: Self):
        self.data = copy.data

    def __eq__(self, other: Self) -> Bool:
        return self.data[0] == other.data[0] and self.data[1] == other.data[1]


struct TripleTup(ImplicitlyCopyable, Copyable, Movable, Equatable):
    var data: Tuple[Float64, Float64, Float64]

    def __init__(out self, data: Tuple[Float64, Float64, Float64]):
        self.data = (data[0], data[1], data[2])

    def __init__(out self, *, deinit move: Self):
        self.data = move.data

    def __init__(out self, *, copy: Self):
        self.data = copy.data

    def __eq__(self, other: Self) -> Bool:
        return self.data[0] == other.data[0] and self.data[1] == other.data[1] and self.data[2] == other.data[2]


struct QuadTup(ImplicitlyCopyable, Copyable, Movable, Equatable):
    var data: Tuple[Float64, Float64, Float64, Float64]

    def __init__(out self, data: Tuple[Float64, Float64, Float64, Float64]):
        self.data = (data[0], data[1], data[2], data[3])

    def __init__(out self, *, deinit move: Self):
        self.data = move.data

    def __init__(out self, *, copy: Self):
        self.data = copy.data

    def __eq__(self, other: Self) -> Bool:
        return self.data[0] == other.data[0] and self.data[1] == other.data[1] and self.data[2] == other.data[2] and self.data[3] == other.data[3]


struct DoubleTupleKey(ImplicitlyCopyable, Copyable, Movable, Hashable, Equatable):
    var i: DoubleTup

    def __init__(out self, i: DoubleTup):
        self.i = i

    def __init__(out self, *, deinit move: Self):
        self.i = move.i

    def __init__(out self, *, copy: Self):
        self.i = copy.i

    def __hash__(self) -> UInt:
        var hash_val: UInt = 0x345678
        var multiplier: UInt = 1000003
        var add_const: UInt = 82520

        var compound_key0_hash: UInt = UInt(hash(self.i.data[0]))
        hash_val = (hash_val ^ compound_key0_hash) * multiplier
        multiplier += (add_const + 0)

        var compound_key1_hash: UInt = UInt(hash(self.i.data[1]))
        hash_val = (hash_val ^ compound_key1_hash) * multiplier
        multiplier += (add_const + 2)

        hash_val += 97531
        return hash_val

    def __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    def __ne__(self, other: Self) -> Bool:
        return self.i != other.i


struct TupleKey(ImplicitlyCopyable, Copyable, Movable, Hashable, Equatable):
    var i: TripleTup

    def __init__(out self, i: TripleTup):
        self.i = i

    def __init__(out self, *, deinit move: Self):
        self.i = move.i

    def __init__(out self, *, copy: Self):
        self.i = copy.i

    def __hash__(self) -> UInt:
        var hash_val: UInt = 0x345678
        var multiplier: UInt = 1000003
        var add_const: UInt = 82520

        var compound_key0_hash: UInt = UInt(hash(self.i.data[0]))
        hash_val = (hash_val ^ compound_key0_hash) * multiplier
        multiplier += (add_const + 0)

        var compound_key1_hash: UInt = UInt(hash(self.i.data[1]))
        hash_val = (hash_val ^ compound_key1_hash) * multiplier
        multiplier += (add_const + 2)

        var compound_key2_hash: UInt = UInt(hash(self.i.data[2]))
        hash_val = (hash_val ^ compound_key2_hash) * multiplier
        multiplier += (add_const + 4)

        hash_val += 97531
        return hash_val

    def __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    def __ne__(self, other: Self) -> Bool:
        return self.i != other.i


struct QuadTupleKey(ImplicitlyCopyable, Copyable, Movable, Hashable, Equatable):
    var i: QuadTup

    def __init__(out self, i: QuadTup):
        self.i = i

    def __init__(out self, *, deinit move: Self):
        self.i = move.i

    def __init__(out self, *, copy: Self):
        self.i = copy.i

    def __hash__(self) -> UInt:
        var hash_val: UInt = 0x345678
        var multiplier: UInt = 1000003
        var add_const: UInt = 82520

        var compound_key0_hash: UInt = UInt(hash(self.i.data[0]))
        hash_val = (hash_val ^ compound_key0_hash) * multiplier
        multiplier += (add_const + 0)

        var compound_key1_hash: UInt = UInt(hash(self.i.data[1]))
        hash_val = (hash_val ^ compound_key1_hash) * multiplier
        multiplier += (add_const + 2)

        var compound_key2_hash: UInt = UInt(hash(self.i.data[2]))
        hash_val = (hash_val ^ compound_key2_hash) * multiplier
        multiplier += (add_const + 4)

        var compound_key3_hash: UInt = UInt(hash(self.i.data[3]))
        hash_val = (hash_val ^ compound_key3_hash) * multiplier
        multiplier += (add_const + 6)

        hash_val += 97531
        return hash_val

    def __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    def __ne__(self, other: Self) -> Bool:
        return self.i != other.i


struct MultiFloatKeyIncremental(ImplicitlyCopyable, Copyable, Movable, Hashable, Equatable):
    var values: List[Float64]
    var _hash_value: UInt
    var _hash_multiplier: UInt
    var _elements_added: Int

    def __init__(out self, num_groupby_cols: Int):
        self.values = List[Float64]()
        self._hash_value = HASH_SEED
        self._hash_multiplier = HASH_MULTIPLIER_INIT
        self._elements_added = 0

    def __init__(out self, *, deinit move: Self):
        self.values = move.values^
        self._hash_value = move._hash_value
        self._hash_multiplier = move._hash_multiplier
        self._elements_added = move._elements_added

    def __init__(out self, *, copy: Self):
        self.values = copy.values.copy()
        self._hash_value = copy._hash_value
        self._hash_multiplier = copy._hash_multiplier
        self._elements_added = copy._elements_added

    def update(mut self, value: Float64):
        self.values.append(value)
        var value_hash: UInt = UInt(hash(value))
        self._hash_value = (self._hash_value ^ value_hash) * self._hash_multiplier
        self._hash_multiplier += (HASH_ADD_CONST + UInt(self._elements_added * 2))
        self._elements_added += 1

    def finalize_hash(mut self):
        self._hash_value += HASH_FINAL_ADD

    def __hash__(self) -> UInt:
        return self._hash_value

    def __eq__(self, other: Self) -> Bool:
        if len(self.values) != len(other.values):
            return False
        for i in range(len(self.values)):
            if self.values[i] != other.values[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


struct FloatKey(ImplicitlyCopyable, Copyable, Movable, Hashable, Equatable):
    var i: Float64

    def __init__(out self, i: Float64):
        self.i = i

    def __init__(out self, *, copy: Self):
        self.i = copy.i

    def __init__(out self, *, deinit move: Self):
        self.i = move.i

    def __hash__(self) -> UInt:
        return UInt(hash(self.i))

    def __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    def __ne__(self, other: Self) -> Bool:
        return self.i != other.i


struct SetElement(ImplicitlyCopyable, Copyable, Movable):
    var distinct_elements: Set[FloatKey]

    def __init__(out self):
        self.distinct_elements = Set[FloatKey]()

    def __init__(out self, *, copy: Self):
        self.distinct_elements = copy.distinct_elements.copy()

    def __init__(out self, *, deinit move: Self):
        self.distinct_elements = move.distinct_elements^
