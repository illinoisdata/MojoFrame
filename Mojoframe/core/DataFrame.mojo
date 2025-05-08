from collections.dict import Dict, KeyElement
from core.Calculations import mergesort, FloatKey, evaluate_f64_tensor
from collections import Set
from algorithm import parallelize, vectorize
from time import monotonic, perf_counter
from max.tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index, IndexList
from buffer import NDBuffer, DimList
from sys.info import simdwidthof

alias FLOAT_VEC_WIDTH = simdwidthof[DType.float64]()

struct DataFrameF64:
    var columns: List[Float64Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int

    fn __init__(mut self, columns: List[Float64Array], column_names: List[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = List[Float64Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(len(columns)):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Float64Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Float64Array:
        var column_index = Int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(mut self, i: Int, value: Float64Array):
        self.columns[i] = value
    
    fn append_column(mut self, owned new_column: Float64Array, new_column_name: String):
        self.columns.append(new_column)
        self.column_names.append(new_column_name)
        self.col_name_to_idx[new_column_name] = (len(self.columns) - 1)

    fn sum(mut self, axis: Int) raises -> Float64Array:
        var sums = Float64Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_f64(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    fn groupby(mut self, column: String, aggregation: String, aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            # Pass in columns for groupby without making copies
            self.columns = aggregation_sum_f64(self.columns, self.column_names, self.col_name_to_idx[column])
            # self.columns = aggregation_sum_f64_parallel(self.columns, self.column_names, self.col_name_to_idx[column])
        elif aggregation == "mean":
            self.columns = aggregation_mean_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        elif aggregation == "count":
            self.columns = aggregation_count_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        elif aggregation == "count_distinct":
            # the first column is the groups, the second column is the aggregation results
            self.columns = aggregation_count_distinct_f64(self.columns, self.column_names, self.col_name_to_idx[column], self.col_name_to_idx[aggregated_col_names[1]])
        elif aggregation == "min":
            self.columns = aggregation_min_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        else:
            self.columns = aggregation_all_f64(self.columns, self.column_names, self.col_name_to_idx[column])

        # if aggregation == "count_distinct":
        #     # the first column is the groups, the second column is the aggregation results
        #     self.columns = aggregation_count_distinct_f64(self.columns, self.column_names, self.col_name_to_idx[column], self.col_name_to_idx[aggregated_col_names[1]])
        
        self.column_names = aggregated_col_names
        self.col_name_to_idx.clear()
        # re-map column names to data
        for i in range(len(self.columns)):
            self.col_name_to_idx[aggregated_col_names[i]] = i
    
    fn groupby_conditional(mut self, column: String, aggregation: String, read mask: List[Bool], aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            # Pass in columns for groupby without making copies
            self.columns = aggregation_sum_conditional_f64(self.columns, self.column_names, mask, self.col_name_to_idx[column])
        
        self.column_names = aggregated_col_names
        self.col_name_to_idx.clear()
        # re-map column names to data
        for i in range(len(self.columns)):
            self.col_name_to_idx[aggregated_col_names[i]] = i
    
    fn groupby_multicol(mut self, col_names: List[String], aggregation: String, aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            # Pass in columns for groupby without making copies
            self.columns = aggregation_sum_f64_multicol(self, col_names)

        elif aggregation == "count":
            self.columns = aggregation_count_f64_multicol(self, col_names)

        else:
            # Pass in columns for groupby without making copies
            self.columns = aggregation_all_f64_multicol(self, col_names)
        
        self.column_names = aggregated_col_names
        self.col_name_to_idx.clear()
        # re-map column names to data
        for i in range(len(self.columns)):
            self.col_name_to_idx[aggregated_col_names[i]] = i
        
        
        

    # fn sum[simd_width: Int](self, axis: Int) raises -> Float64Array:
    #     var sums = Float64Array(self.columns.__len__())
    #     if simd_width:
    #         for i in range(self.columns.__len__()):
    #             sums[i] = (self.columns[i].data.simd_load[simd_width](0)).reduce_sum()
    #     else:
    #         if axis == self.index_axis:
    #             for i in range(self.columns.__len__()):
    #                 sums[i] = pairwise_sum_f64(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
    #     return sums
    
    # select_mask -> [0, 1, 0, 0]
    # select_complex(List[List[Int]], logical_operator)
    fn select_complex(mut self, read masks: List[List[Bool]], logical_operator: String) raises:
        var filtered_mask = masks[0]
        var mask_len = len(filtered_mask)

        if logical_operator == "AND":
            for i in range(1, len(masks)):
                var cur_mask = masks[i]
                for j in range(mask_len):
                    filtered_mask[j] = (filtered_mask[j] and cur_mask[j])
        elif logical_operator == "OR":
            for i in range(1, len(masks)):
                var cur_mask = masks[i]
                for j in range(mask_len):
                    filtered_mask[j] = (filtered_mask[j] or cur_mask[j])

        var selected_indices = List[Int]()
        for i in range(mask_len):
            if filtered_mask[i]:
                selected_indices.append(i)

        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Float64Array]()
       
        for col_i in range(len(self.columns)):
            var col_to_fill = Float64Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data
    
    fn select_mask[T: PredicateF64, T2: PredicateF64](mut self, column_1: String, column_2: String,
                    predicate_1: T, predicate_2: T2,
                    value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                    logical_operator: String) raises -> List[Bool]:

        var selected_mask = evaluate_f64_mask(self.columns[self.col_name_to_idx[column_1]],
                                            self.columns[self.col_name_to_idx[column_2]],
                                            predicate_1, predicate_2,
                                            value_cmp_1, value_cmp_2, logical_operator)
        return selected_mask

    fn select[T: PredicateF64, T2: PredicateF64](mut self, column_1: String, column_2: String,
                    predicate_1: T, predicate_2: T2,
                    value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                    logical_operator: String) raises:

        # var selected_indices = evaluate_f64(self.columns[self.col_name_to_idx[column_1]],
        #                                     self.columns[self.col_name_to_idx[column_2]],
        #                                     predicate_1, predicate_2,
        #                                     value_cmp_1, value_cmp_2, logical_operator)
        

        var num_threads = 8
        var selected_indices = evaluate_f64_alt(self.columns[self.col_name_to_idx[column_1]],
                                            self.columns[self.col_name_to_idx[column_2]],
                                            predicate_1, predicate_2,
                                            value_cmp_1, value_cmp_2, logical_operator)
        # Construct filtered DataFrame based on selected indices and its dimension(size)
        
        # for i in range(selected_indices.__len__()):
        #     if idx[i] != selected_indices[i]:
        #         print("broke index:", i)
        #         print("broke here:",selected_indices[i])
        #         break

        var filtered_rows_N = selected_indices.size

        var filtered_data = List[Float64Array]()
        var chunk_size = 640000
        var n_chunks = (filtered_rows_N + chunk_size - 1) // chunk_size
        # for i in range(114160-300, 114160-200):
        #     print((self.columns[self.col_name_to_idx[column_1]])[selected_indices[i]])
        for col_i in range(len(self.columns)):
            var col_to_fill = Float64Array(filtered_rows_N)
            # for row_i in range(filtered_rows_N):
            #     col_to_fill[row_i] = self.columns[col_i][selected_indices[row_i]]

            @parameter
            fn fill_data_worker(chunk_id: Int):
                var start_i = chunk_id * chunk_size
                var end_i = min(start_i + chunk_size, filtered_rows_N)
                var limit = ((end_i - start_i) // FLOAT_VEC_WIDTH) * FLOAT_VEC_WIDTH + start_i

                
                # Copy a slice of rows
                for row_i in range(start_i, limit, FLOAT_VEC_WIDTH):
                    var filtered_idxs = selected_indices.data.load[width=FLOAT_VEC_WIDTH](row_i)
                    var gathered_values = SIMD[DType.float64, FLOAT_VEC_WIDTH]()

                    @parameter
                    for k in range(FLOAT_VEC_WIDTH):
                        gathered_values[k] = self.columns[col_i][filtered_idxs[k].__int__()]
                    
                    col_to_fill.data.store[FLOAT_VEC_WIDTH](row_i, gathered_values)
                    # col_to_fill.data.store[width=](row_i, SIMD[DType.float64, 8](
                    #                                 self.columns[col_i][filtered_idxs[0].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[1].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[2].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[3].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[4].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[5].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[6].__int__()],
                    #                                 self.columns[col_i][filtered_idxs[7].__int__()]))
                
                for row_i in range(limit, end_i):
                    var matched_idx = selected_indices[row_i].__int__()
                    col_to_fill[row_i] = self.columns[col_i][matched_idx]

                # @parameter
                # fn gather_vector_op[vec_width: Int](i: Int): 
                #     # # i is index in the chunk (0 ... chunk_len-1)
                #     var absolute_i = start_i + i

                #     var filtered_idxs = selected_indices.data.load[vec_width](absolute_i)
                #     var gathered_values = SIMD[DType.float64, vec_width]()

                #     for k in range(vec_width):
                #         gathered_values[k] = self.columns[col_i][filtered_idxs[k].__int__()]
                    
                #     col_to_fill.data.store[vec_width](absolute_i, gathered_values)

                # # apply vectorized op over the chunk length
                # vectorize[gather_vector_op, FLOAT_VEC_WIDTH](chunk_len)


            parallelize[fill_data_worker](n_chunks, num_threads)

            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data^
    
    fn sort_by(mut self, by: List[String]) raises:
        var key0 = self.__getitem__(by[0])
        var sorted_indexer = List[Int](capacity=key0.size)
        for i in range(key0.size):
            sorted_indexer.append(i)

        # If there is only a single key, perform argsort
        if len(by) == 1:
            _ = mergesort(key0, sorted_indexer)
        
        elif len(by) > 1:
            # If there are multiple keys, perform lex sort
            # Lexicographic sort example 
            # (1, 4, 5) (1, 3, 6) (1, 1, 8)
            # Sort by the last key first, then proceed to the primary left most key
            # indices: 0, 1, 2 -> (1, 4, 5) (1, 3, 6) (1, 1, 8)
            # Now sort by the second to last key, using the above indices to access elements [4, 3, 1]
            # indices: 2, 1, 0 -> (1, 1, 8) (1, 3, 6) (1, 4, 5)

            # What if the primary key is sorted first?
            # (1, 4, 5) (1, 3, 6) (1, 1, 8)
            # (1, 1, 8) (1, 3, 6) (1, 4, 5)
            # (1, 4, 5) (1, 3, 6) (1, 1, 8) the sorting violates the lexicographic order

            for i in range(len(by) - 1, -1, -1):
                var key = self.__getitem__(by[i])
                _ = mergesort(key, sorted_indexer)
        
        var sorted_data = List[Float64Array]()
        for col_i in range(len(self.columns)):
            var col_to_fill = Float64Array(key0.size)
            var original_col = self.columns[col_i]
            for row_i in range(key0.size):
                col_to_fill[row_i] = original_col[sorted_indexer[row_i]]
            sorted_data.append(col_to_fill)
        
        self.columns = sorted_data
            
    
    # fn select[T: PredicateF64, T2: PredicateF64,
    #           T3: PredicateF64, T4: PredicateF64,
    #           T5: PredicateF64](mut self, column_1: String, column_2: String, column_3: String,
    #                             predicate_1: T, predicate_2: T2, predicate_3: T3, predicate_4: T4, predicate_5: T5,
    #                             value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
    #                             value_cmp_3: SIMD[DType.float64, 1], value_cmp_4: SIMD[DType.float64, 1],
    #                             value_cmp_5: SIMD[DType.float64, 1], logical_operator: String) raises:

    #     var selected_indices = evaluate_query6(self.columns[self.col_name_to_idx[column_1]],
    #                                         self.columns[self.col_name_to_idx[column_2]],
    #                                         self.columns[self.col_name_to_idx[column_3]],
    #                                         predicate_1, predicate_2, predicate_3, predicate_4, predicate_5,
    #                                         value_cmp_1, value_cmp_2, value_cmp_3, value_cmp_4, value_cmp_5, logical_operator)
    #     # Construct filtered DataFrame based on selected indices and its dimension(size)
        
    #     var filtered_rows_N = selected_indices.__len__()
    #     print("filtered:", filtered_rows_N)
    #     var filtered_data = List[Float64Array]()
    #     for col_i in range(self.columns.size):
    #         var col_to_fill = Float64Array(filtered_rows_N)
    #         var original_col = self.columns[col_i]
    #         for row_i in range(filtered_rows_N):
    #             col_to_fill[row_i] = original_col[selected_indices[row_i]]
    #         filtered_data.append(col_to_fill)
        
    #     self.columns = filtered_data

    fn rename_column(mut self, original_col_name: String, new_col_name: String) raises:
        var col_index = self.col_name_to_idx[original_col_name]
        self.column_names[col_index] = new_col_name
        self.col_name_to_idx[new_col_name] = col_index
        _ = self.col_name_to_idx.pop(original_col_name)


struct DataFrameF32:
    var columns: List[Float32Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int

    fn __init__(mut self, columns: List[Float32Array], column_names: List[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = List[Float32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(len(columns)):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Float32Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Float32Array:
        var column_index = Int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(mut self, i: Int, value: Float32Array):
        self.columns[i] = value

    fn sum(mut self, axis: Int) raises -> Float32Array:
        var sums = Float32Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_f32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    fn select(mut self, column_1: String, column_2: String,
                    value_cmp_1: SIMD[DType.float32, 1], value_cmp_2: SIMD[DType.float32, 1]) raises:

        var selected_indices = evaluate_f32(self.columns[self.col_name_to_idx[column_1]],
                                            self.columns[self.col_name_to_idx[column_2]],
                                            value_cmp_1, value_cmp_2)
        # Construct filtered DataFrame based on selected indices and its dimension(size)
        
        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Float32Array]()
        for col_i in range(len(self.columns)):
            var col_to_fill = Float32Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data

struct DataFrameI32:
    var columns: List[Int32Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int
    
    fn __init__(mut self, columns: List[Int32Array], column_names: List[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = List[Int32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(len(columns)):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i
    
    fn __getitem__(self, i: Int) raises -> Int32Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Int32Array:
        var column_index = Int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(mut self, i: Int, value: Int32Array):
        self.columns[i] = value
    
    fn sum(mut self, axis: Int) raises -> Int32Array:
        var sums = Int32Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                # pass in columns without making copies
                sums[i] = pairwise_sum_i32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    # fn groupby(mut self, column: String, aggregation: String) raises:
    #     if aggregation == "sum":
    #         # Pass in columns for groupby without making copies
    #         self.columns = aggregation_sum_i32(self.columns, self.column_names, self.col_name_to_idx[column])

    fn select(mut self, column: String, operation: String, value_cmp: SIMD[DType.int32, 1]) raises:
        var selected_indices = evaluate_i32(self.columns[self.col_name_to_idx[column]], operation, value_cmp)
        # Construct filtered DataFrame based on selected indices and its dimension(size)
    
        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Int32Array]()
        for col_i in range(len(self.columns)):
            var col_to_fill = Int32Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data


    # fn test_row_sum(mut self) raises:
    #     var sm = SIMD[DType.int32, 1](0)
    #     var cur_col = self.columns[0]
    #     var num_rows = self.columns[0].size
    #     for row_i in range(num_rows):
    #         sm += cur_col[row_i]

struct TensorDataFrameF64:
    # Transposed storage D_columns x N_rows
    var data: Tensor[DType.float64]
    var _num_rows: Int 
    var _num_cols: Int 
    var column_names: List[String] 
    var col_name_to_idx: Dict[String, Int] 

    fn __init__(out self, read columns: List[Float64Array], column_names: List[String]) raises:
        self._num_cols = len(columns)
        self._num_rows = columns[0].size

        self.column_names = List[String](column_names)
        self.col_name_to_idx = Dict[String, Int]()
        for j in range(self._num_cols):
            self.col_name_to_idx[column_names[j]] = j

        var spec = TensorSpec(DType.float64, TensorShape(self._num_cols, self._num_rows))

        self.data = Tensor[DType.float64](spec)
        for j in range(self._num_cols):
            for i in range(self._num_rows):
                self.data[Index(j, i)] = columns[j][i]
        

        var base_ptr = self.data.unsafe_ptr()
        var row_stride = self._num_rows
        var start_offset = 0 * row_stride
        var view_start_ptr = base_ptr + start_offset

        # Create Span covering the N contiguous elements of the row
        var view_span = Span(base_ptr, 5)
        print(view_span[3])
    
    fn select[T: PredicateF64, T2: PredicateF64](
            mut self,
            column_1: String, column_2: String,
            predicate_1: T, predicate_2: T2,
            value_cmp_1: SIMD[DType.float64, 1],
            value_cmp_2: SIMD[DType.float64, 1],
            logical_operator: String,
            num_threads: Int = 4
            ) raises:

        var dict_time = perf_counter()


        var col1_idx = self.col_name_to_idx[column_1]
        var col2_idx = self.col_name_to_idx[column_2]

        var end_dict_time = perf_counter()

        print("Time taken to map col: ", end_dict_time - dict_time, " seconds")


        var before_filter = perf_counter()

        var selected_indices = evaluate_f64_tensor(
                                self,
                                col1_idx, col2_idx,
                                predicate_1, predicate_2,
                                value_cmp_1, value_cmp_2,
                                logical_operator, num_threads)

        var after_filter = perf_counter()
        print("Time taken to create indices: ", after_filter - before_filter, " seconds")

        var filtered_rows_N = selected_indices.size
        print("Number of rows after filtering:", filtered_rows_N)

        var start_create = perf_counter()
        var result_spec = TensorSpec(DType.float64, TensorShape(self._num_cols, filtered_rows_N))
        var result_tensor = Tensor[DType.float64](result_spec)
        var end_create = perf_counter()
        print("Time taken to create empty tensor: ", end_create - start_create, " seconds")

        for j in range(self._num_cols):
            var start_col_worker_timer = perf_counter()

            var chunk_size = 640000 # tune
            var n_chunks = (filtered_rows_N + chunk_size - 1) // chunk_size

            @parameter
            fn fill_column_worker(chunk_id: Int):
                var start_new_i = chunk_id * chunk_size 
                var end_new_i = min(start_new_i + chunk_size, filtered_rows_N)

                for new_i in range(start_new_i, end_new_i):
                    var original_i = selected_indices[new_i].__int__()
                    var value_to_copy = self.data[Index(j, original_i)]

                    result_tensor[Index(j, new_i)] = value_to_copy

            # parallel workers for the current column j
            parallelize[fill_column_worker](n_chunks, num_threads)

            var end_col_worker_timer = perf_counter()
            print("Time taken to fill column ", j, ": ", end_col_worker_timer - start_col_worker_timer, " seconds")

        var change_assignment = perf_counter()

        self._num_rows = filtered_rows_N
        self.data = result_tensor^

        var end_change_assignment = perf_counter()
        print("Time taken to change assignment: ", end_change_assignment - change_assignment, " seconds")

    
# @value
# struct String(KeyElement):
#     var s: String

#     fn __init__(mut self, owned s: String):
#         self.s = s ^

#     fn __init__(mut self, s: StringLiteral):
#         self.s = String(s)

#     fn __hash__(self) -> Int:
#         var ptr = self.s._buffer.data.value
#         return hash(DTypePointer[DType.int8](ptr), len(self.s))

#     fn __eq__(self, other: Self) -> Bool:
#         return self.s == other.s


struct SetElement(CollectionElement):
    var distinct_elements: Set[FloatKey]
    
    fn __init__(mut self) raises:
        self.distinct_elements = Set[FloatKey]()
        
    fn __moveinit__(mut self, owned existing: Self):
        self.distinct_elements = (existing.distinct_elements)^
        
    fn __copyinit__(mut self, existing: Self):
        self.distinct_elements = Set[FloatKey]()
        self.distinct_elements = self.distinct_elements.union(existing.distinct_elements)
