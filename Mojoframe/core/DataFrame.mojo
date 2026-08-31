from std.collections import Dict, Set, List
from std.time import perf_counter
from std.sys.info import simd_width_of
from core.Arrays import Float64Array, Float32Array, Int32Array
from core.Calculations import (
    pairwise_sum_f64,
    pairwise_sum_f32,
    pairwise_sum_i32,
    aggregation_sum_f64,
    aggregation_mean_f64,
    aggregation_count_f64,
    aggregation_count_distinct_f64,
    aggregation_min_f64,
    aggregation_all_f64,
    aggregation_sum_conditional_f64,
    aggregation_sum_f64_multicol,
    aggregation_count_f64_multicol,
    aggregation_all_f64_multicol,
    evaluate_f64_mask,
    evaluate_f64_alt,
    evaluate_f32,
    evaluate_i32,
    mergesort,
    FloatKey,
    PredicateF64,
)

comptime FLOAT_VEC_WIDTH = simd_width_of[DType.float64]()


struct DataFrameF64(ImplicitlyCopyable, Copyable, Movable):
    var columns: List[Float64Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int

    def __init__(out self):
        self.columns = List[Float64Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1

    def __init__(out self, columns: List[Float64Array], column_names: List[String]) raises:
        self.columns = List[Float64Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1

        for i in range(len(columns)):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    def __init__(out self, *, copy: Self):
        self.columns = copy.columns.copy()
        self.column_names = copy.column_names.copy()
        self.col_name_to_idx = copy.col_name_to_idx.copy()
        self.index_axis = copy.index_axis
        self.column_axis = copy.column_axis

    def __init__(out self, *, deinit move: Self):
        self.columns = move.columns^
        self.column_names = move.column_names^
        self.col_name_to_idx = move.col_name_to_idx^
        self.index_axis = move.index_axis
        self.column_axis = move.column_axis

    def __getitem__(self, i: Int) raises -> Float64Array:
        return self.columns[i]

    def __getitem__(self, column_name: String) raises -> Float64Array:
        var column_index = Int(self.col_name_to_idx[column_name])
        return self.columns[column_index]

    def __setitem__(mut self, i: Int, value: Float64Array):
        self.columns[i] = value

    def append_column(mut self, var new_column: Float64Array, new_column_name: String):
        self.columns.append(new_column^)
        self.column_names.append(new_column_name)
        self.col_name_to_idx[new_column_name] = len(self.columns) - 1

    def sum(mut self, axis: Int) raises -> Float64Array:
        var sums = Float64Array(len(self.columns))
        if axis == self.index_axis:
            for i in range(len(self.columns)):
                sums[i] = pairwise_sum_f64(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    def groupby(mut self, column: String, aggregation: String, aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            self.columns = aggregation_sum_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        elif aggregation == "mean":
            self.columns = aggregation_mean_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        elif aggregation == "count":
            self.columns = aggregation_count_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        elif aggregation == "count_distinct":
            self.columns = aggregation_count_distinct_f64(self.columns, self.column_names, self.col_name_to_idx[column], self.col_name_to_idx[aggregated_col_names[1]])
        elif aggregation == "min":
            self.columns = aggregation_min_f64(self.columns, self.column_names, self.col_name_to_idx[column])
        else:
            self.columns = aggregation_all_f64(self.columns, self.column_names, self.col_name_to_idx[column])

        self.column_names = aggregated_col_names.copy()
        self.col_name_to_idx = Dict[String, Int]()
        for i in range(len(self.columns)):
            self.col_name_to_idx[aggregated_col_names[i]] = i

    def groupby_conditional(mut self, column: String, aggregation: String, mask: List[Bool], aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            self.columns = aggregation_sum_conditional_f64(self.columns, self.column_names, mask, self.col_name_to_idx[column])

        self.column_names = aggregated_col_names.copy()
        self.col_name_to_idx = Dict[String, Int]()
        for i in range(len(self.columns)):
            self.col_name_to_idx[aggregated_col_names[i]] = i

    def groupby_multicol(mut self, col_names: List[String], aggregation: String, aggregated_col_names: List[String]) raises:
        if aggregation == "sum":
            self.columns = aggregation_sum_f64_multicol(self, col_names)
        elif aggregation == "count":
            self.columns = aggregation_count_f64_multicol(self, col_names)
        else:
            self.columns = aggregation_all_f64_multicol(self, col_names)

        self.column_names = aggregated_col_names.copy()
        self.col_name_to_idx = Dict[String, Int]()
        for i in range(len(self.columns)):
            self.col_name_to_idx[aggregated_col_names[i]] = i

    def select_complex(mut self, masks: List[List[Bool]], logical_operator: String) raises:
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

        var selected_indices = List[Int]()
        for i in range(mask_len):
            if filtered_mask[i]:
                selected_indices.append(i)

        var filtered_rows_N = len(selected_indices)
        var filtered_data = List[Float64Array]()

        for col_i in range(len(self.columns)):
            var col_to_fill = Float64Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)

        self.columns = filtered_data^

    def select_mask[T: PredicateF64, T2: PredicateF64](
        mut self,
        column_1: String,
        column_2: String,
        predicate_1: T,
        predicate_2: T2,
        value_cmp_1: Float64,
        value_cmp_2: Float64,
        logical_operator: String
    ) raises -> List[Bool]:
        var selected_mask = evaluate_f64_mask(
            self.columns[self.col_name_to_idx[column_1]],
            self.columns[self.col_name_to_idx[column_2]],
            predicate_1,
            predicate_2,
            value_cmp_1,
            value_cmp_2,
            logical_operator
        )
        return selected_mask^

    def select[T: PredicateF64, T2: PredicateF64](
        mut self,
        column_1: String,
        column_2: String,
        predicate_1: T,
        predicate_2: T2,
        value_cmp_1: Float64,
        value_cmp_2: Float64,
        logical_operator: String
    ) raises:
        var selected_indices = evaluate_f64_alt(
            self.columns[self.col_name_to_idx[column_1]],
            self.columns[self.col_name_to_idx[column_2]],
            predicate_1,
            predicate_2,
            value_cmp_1,
            value_cmp_2,
            logical_operator
        )
        var filtered_rows_N = selected_indices.size
        var filtered_data = List[Float64Array]()
        var chunk_size = 640000
        var n_chunks = (filtered_rows_N + chunk_size - 1) // chunk_size

        for col_i in range(len(self.columns)):
            var col_to_fill = Float64Array(filtered_rows_N)

            for chunk_id in range(n_chunks):
                var start_i = chunk_id * chunk_size
                var end_i = min(start_i + chunk_size, filtered_rows_N)
                var limit = ((end_i - start_i) // FLOAT_VEC_WIDTH) * FLOAT_VEC_WIDTH + start_i

                for row_i in range(start_i, limit, FLOAT_VEC_WIDTH):
                    var filtered_idxs = selected_indices.load[FLOAT_VEC_WIDTH](row_i)
                    var gathered_values = SIMD[DType.float64, FLOAT_VEC_WIDTH]()

                    for k in range(FLOAT_VEC_WIDTH):
                        gathered_values[k] = self.columns[col_i][Int(filtered_idxs[k])]

                    col_to_fill.store[FLOAT_VEC_WIDTH](row_i, gathered_values)

                for row_i in range(limit, end_i):
                    var matched_idx = Int(selected_indices[row_i])
                    col_to_fill[row_i] = self.columns[col_i][matched_idx]

            filtered_data.append(col_to_fill)

        self.columns = filtered_data^

    def sort_by(mut self, by: List[String]) raises:
        var key0 = self.__getitem__(by[0])
        var sorted_indexer = List[Int](capacity=key0.size)
        for i in range(key0.size):
            sorted_indexer.append(i)

        if len(by) == 1:
            _ = mergesort(key0, sorted_indexer)
        elif len(by) > 1:
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

        self.columns = sorted_data^

    def rename_column(mut self, original_col_name: String, new_col_name: String) raises:
        var col_index = self.col_name_to_idx[original_col_name]
        self.column_names[col_index] = new_col_name
        self.col_name_to_idx[new_col_name] = col_index
        _ = self.col_name_to_idx.pop(original_col_name)


struct DataFrameF32(ImplicitlyCopyable, Copyable, Movable):
    var columns: List[Float32Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int

    def __init__(out self):
        self.columns = List[Float32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1

    def __init__(out self, columns: List[Float32Array], column_names: List[String]) raises:
        self.columns = List[Float32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1

        for i in range(len(columns)):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    def __init__(out self, *, copy: Self):
        self.columns = copy.columns.copy()
        self.column_names = copy.column_names.copy()
        self.col_name_to_idx = copy.col_name_to_idx.copy()
        self.index_axis = copy.index_axis
        self.column_axis = copy.column_axis

    def __init__(out self, *, deinit move: Self):
        self.columns = move.columns^
        self.column_names = move.column_names^
        self.col_name_to_idx = move.col_name_to_idx^
        self.index_axis = move.index_axis
        self.column_axis = move.column_axis

    def __getitem__(self, i: Int) raises -> Float32Array:
        return self.columns[i]

    def __getitem__(self, column_name: String) raises -> Float32Array:
        var column_index = Int(self.col_name_to_idx[column_name])
        return self.columns[column_index]

    def __setitem__(mut self, i: Int, value: Float32Array):
        self.columns[i] = value

    def sum(mut self, axis: Int) raises -> Float32Array:
        var sums = Float32Array(len(self.columns))
        if axis == self.index_axis:
            for i in range(len(self.columns)):
                sums[i] = pairwise_sum_f32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    def select(mut self, column_1: String, column_2: String, value_cmp_1: Float32, value_cmp_2: Float32) raises:
        var selected_indices = evaluate_f32(
            self.columns[self.col_name_to_idx[column_1]],
            self.columns[self.col_name_to_idx[column_2]],
            value_cmp_1,
            value_cmp_2
        )
        var filtered_rows_N = len(selected_indices)
        var filtered_data = List[Float32Array]()
        for col_i in range(len(self.columns)):
            var col_to_fill = Float32Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)

        self.columns = filtered_data^


struct DataFrameI32(ImplicitlyCopyable, Copyable, Movable):
    var columns: List[Int32Array]
    var column_names: List[String]
    var col_name_to_idx: Dict[String, Int]
    var index_axis: Int
    var column_axis: Int

    def __init__(out self):
        self.columns = List[Int32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1

    def __init__(out self, columns: List[Int32Array], column_names: List[String]) raises:
        self.columns = List[Int32Array]()
        self.column_names = List[String]()
        self.col_name_to_idx = Dict[String, Int]()
        self.index_axis = 0
        self.column_axis = 1

        for i in range(len(columns)):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    def __init__(out self, *, copy: Self):
        self.columns = copy.columns.copy()
        self.column_names = copy.column_names.copy()
        self.col_name_to_idx = copy.col_name_to_idx.copy()
        self.index_axis = copy.index_axis
        self.column_axis = copy.column_axis

    def __init__(out self, *, deinit move: Self):
        self.columns = move.columns^
        self.column_names = move.column_names^
        self.col_name_to_idx = move.col_name_to_idx^
        self.index_axis = move.index_axis
        self.column_axis = move.column_axis

    def __getitem__(self, i: Int) raises -> Int32Array:
        return self.columns[i]

    def __getitem__(self, column_name: String) raises -> Int32Array:
        var column_index = Int(self.col_name_to_idx[column_name])
        return self.columns[column_index]

    def __setitem__(mut self, i: Int, value: Int32Array):
        self.columns[i] = value

    def sum(mut self, axis: Int) raises -> Int32Array:
        var sums = Int32Array(len(self.columns))
        if axis == self.index_axis:
            for i in range(len(self.columns)):
                sums[i] = pairwise_sum_i32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

    def select(mut self, column: String, operation: String, value_cmp: Int32) raises:
        var selected_indices = evaluate_i32(self.columns[self.col_name_to_idx[column]], operation, value_cmp)
        var filtered_rows_N = len(selected_indices)
        var filtered_data = List[Int32Array]()
        for col_i in range(len(self.columns)):
            var col_to_fill = Int32Array(filtered_rows_N)
            var original_col = self.columns[col_i]
            for row_i in range(filtered_rows_N):
                col_to_fill[row_i] = original_col[selected_indices[row_i]]
            filtered_data.append(col_to_fill)

        self.columns = filtered_data^
