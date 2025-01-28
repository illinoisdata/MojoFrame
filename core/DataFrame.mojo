from collections.dict import Dict, KeyElement
from core.Calculations import mergesort, FloatKey
from collections import Set

struct DataFrameF64:
    var columns: List[Float64Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int

    fn __init__(inout self, columns: List[Float64Array], column_names: List[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = List[Float64Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Float64Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Float64Array:
        var column_index = int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(inout self, i: Int, value: Float64Array):
        self.columns[i] = value
    
    fn append_column(inout self, owned new_column: Float64Array, new_column_name: String):
        self.columns.append(new_column)
        self.column_names.append(new_column_name)
        self.col_name_to_idx[new_column_name] = (self.columns.size - 1)

    fn sum(inout self, axis: Int) raises -> Float64Array:
        var sums = Float64Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_f64(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    fn groupby(inout self, column: String, aggregation: String, aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            # Pass in columns for groupby without making copies
            self.columns = aggregation_sum_f64(self.columns, self.column_names, self.col_name_to_idx[column])
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
        for i in range(self.columns.size):
            self.col_name_to_idx[aggregated_col_names[i]] = i
    
    # fn groupby_conditional(inout self, column: String, aggregation: String, borrowed mask: List[Bool], aggregated_col_names: List[String]) raises:
    #     if aggregation == "sum":
    #         # Pass in columns for groupby without making copies
    #         self.columns = aggregation_sum_conditional_f64(self.columns, self.column_names, mask, self.col_name_to_idx[column])
        
    #     self.column_names = aggregated_col_names
    #     self.col_name_to_idx.clear()
    #     # re-map column names to data
    #     for i in range(self.columns.size):
    #         self.col_name_to_idx[aggregated_col_names[i]] = i
    
    fn groupby_multicol(inout self, col_names: List[String], aggregation: String, aggregated_col_names: List[String]) raises:
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
        for i in range(self.columns.size):
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
    fn select_complex(inout self, borrowed masks: List[List[Bool]], logical_operator: String) raises:
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

        var selected_indices = List[Int]()
        for i in range(mask_len):
            if filtered_mask[i]:
                selected_indices.append(i)

        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Float64Array]()
       
        for col_i in range(self.columns.size):
            var col_to_fill = Float64Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data
    
    fn select_mask[T: PredicateF64, T2: PredicateF64](inout self, column_1: String, column_2: String,
                    predicate_1: T, predicate_2: T2,
                    value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                    logical_operator: String) raises -> List[Bool]:

        var selected_mask = evaluate_f64_mask(self.columns[self.col_name_to_idx[column_1]],
                                            self.columns[self.col_name_to_idx[column_2]],
                                            predicate_1, predicate_2,
                                            value_cmp_1, value_cmp_2, logical_operator)
        return selected_mask

    fn select[T: PredicateF64, T2: PredicateF64](inout self, column_1: String, column_2: String,
                    predicate_1: T, predicate_2: T2,
                    value_cmp_1: SIMD[DType.float64, 1], value_cmp_2: SIMD[DType.float64, 1],
                    logical_operator: String) raises:

        var selected_indices = evaluate_f64(self.columns[self.col_name_to_idx[column_1]],
                                            self.columns[self.col_name_to_idx[column_2]],
                                            predicate_1, predicate_2,
                                            value_cmp_1, value_cmp_2, logical_operator)
        # Construct filtered DataFrame based on selected indices and its dimension(size)
        
        # for i in range(selected_indices.__len__()):
        #     if idx[i] != selected_indices[i]:
        #         print("broke index:", i)
        #         print("broke here:",selected_indices[i])
        #         break

        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Float64Array]()
        # for i in range(114160-300, 114160-200):
        #     print((self.columns[self.col_name_to_idx[column_1]])[selected_indices[i]])
        for col_i in range(self.columns.size):
            var col_to_fill = Float64Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data
    
    fn sort_by(inout self, by: List[String]) raises:
        var key0 = self.__getitem__(by[0])
        var sorted_indexer = List[Int](capacity=key0.size)
        for i in range(key0.size):
            sorted_indexer.append(i)

        # If there is only a single key, perform argsort
        if by.size == 1:
            _ = mergesort(key0, sorted_indexer)
        
        elif by.size > 1:
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

            for i in range(by.size - 1, -1, -1):
                var key = self.__getitem__(by[i])
                _ = mergesort(key, sorted_indexer)
        
        var sorted_data = List[Float64Array]()
        for col_i in range(self.columns.size):
            var col_to_fill = Float64Array(key0.size)
            var original_col = self.columns[col_i]
            for row_i in range(key0.size):
                col_to_fill[row_i] = original_col[sorted_indexer[row_i]]
            sorted_data.append(col_to_fill)
        
        self.columns = sorted_data
            
    
    # fn select[T: PredicateF64, T2: PredicateF64,
    #           T3: PredicateF64, T4: PredicateF64,
    #           T5: PredicateF64](inout self, column_1: String, column_2: String, column_3: String,
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

    fn rename_column(inout self, original_col_name: String, new_col_name: String) raises:
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

    fn __init__(inout self, columns: List[Float32Array], column_names: List[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = List[Float32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Float32Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Float32Array:
        var column_index = int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(inout self, i: Int, value: Float32Array):
        self.columns[i] = value

    fn sum(inout self, axis: Int) raises -> Float32Array:
        var sums = Float32Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_f32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    fn select(inout self, column_1: String, column_2: String,
                    value_cmp_1: SIMD[DType.float32, 1], value_cmp_2: SIMD[DType.float32, 1]) raises:

        var selected_indices = evaluate_f32(self.columns[self.col_name_to_idx[column_1]],
                                            self.columns[self.col_name_to_idx[column_2]],
                                            value_cmp_1, value_cmp_2)
        # Construct filtered DataFrame based on selected indices and its dimension(size)
        
        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Float32Array]()
        for col_i in range(self.columns.size):
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
    
    fn __init__(inout self, columns: List[Int32Array], column_names: List[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = List[Int32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i
    
    fn __getitem__(self, i: Int) raises -> Int32Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Int32Array:
        var column_index = int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(inout self, i: Int, value: Int32Array):
        self.columns[i] = value
    
    fn sum(inout self, axis: Int) raises -> Int32Array:
        var sums = Int32Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                # pass in columns without making copies
                sums[i] = pairwise_sum_i32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    # fn groupby(inout self, column: String, aggregation: String) raises:
    #     if aggregation == "sum":
    #         # Pass in columns for groupby without making copies
    #         self.columns = aggregation_sum_i32(self.columns, self.column_names, self.col_name_to_idx[column])

    fn select(inout self, column: String, operation: String, value_cmp: SIMD[DType.int32, 1]) raises:
        var selected_indices = evaluate_i32(self.columns[self.col_name_to_idx[column]], operation, value_cmp)
        # Construct filtered DataFrame based on selected indices and its dimension(size)
    
        var filtered_rows_N = selected_indices.__len__()
        var filtered_data = List[Int32Array]()
        for col_i in range(self.columns.size):
            var col_to_fill = Int32Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)
        
        self.columns = filtered_data


    # fn test_row_sum(inout self) raises:
    #     var sm = SIMD[DType.int32, 1](0)
    #     var cur_col = self.columns[0]
    #     var num_rows = self.columns[0].size
    #     for row_i in range(num_rows):
    #         sm += cur_col[row_i]


# @value
# struct String(KeyElement):
#     var s: String

#     fn __init__(inout self, owned s: String):
#         self.s = s ^

#     fn __init__(inout self, s: StringLiteral):
#         self.s = String(s)

#     fn __hash__(self) -> Int:
#         var ptr = self.s._buffer.data.value
#         return hash(DTypePointer[DType.int8](ptr), len(self.s))

#     fn __eq__(self, other: Self) -> Bool:
#         return self.s == other.s


struct SetElement(CollectionElement):
    var distinct_elements: Set[FloatKey]
    
    fn __init__(inout self) raises:
        self.distinct_elements = Set[FloatKey]()
        
    fn __moveinit__(inout self, owned existing: Self):
        self.distinct_elements = (existing.distinct_elements)^
        
    fn __copyinit__(inout self, existing: Self):
        self.distinct_elements = Set[FloatKey]()
        self.distinct_elements = self.distinct_elements.union(existing.distinct_elements)
