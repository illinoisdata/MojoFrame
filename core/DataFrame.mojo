from collections.dict import Dict, KeyElement

struct DataFrameF64:
    var columns: DynamicVector[Float64Array]
    var column_names: DynamicVector[String]
    var col_name_to_idx: Dict[StringKey, Int]
    var index_axis: Int
    var column_axis: Int

    fn __init__(inout self, columns: DynamicVector[Float64Array], column_names: DynamicVector[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = DynamicVector[Float64Array]()
        self.column_names = DynamicVector[String]()
        self.col_name_to_idx = Dict[StringKey, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Float64Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Float64Array:
        let column_index = int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(inout self, i: Int, value: Float64Array):
        self.columns[i] = value
    
    fn sum(self, axis: Int) raises -> Float64Array:
        var sums = Float64Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_f64(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums
    
struct DataFrameF32:
    var columns: DynamicVector[Float32Array]
    var column_names: DynamicVector[String]
    var col_name_to_idx: Dict[StringKey, Int]
    var index_axis: Int
    var column_axis: Int

    fn __init__(inout self, columns: DynamicVector[Float32Array], column_names: DynamicVector[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = DynamicVector[Float32Array]()
        self.column_names = DynamicVector[String]()
        self.col_name_to_idx = Dict[StringKey, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Float32Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Float32Array:
        let column_index = int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(inout self, i: Int, value: Float32Array):
        self.columns[i] = value

    fn sum(self, axis: Int) raises -> Float32Array:
        var sums = Float32Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_f32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

struct DataFrameI32:
    var columns: DynamicVector[Int32Array]
    var column_names: DynamicVector[String]
    var col_name_to_idx: Dict[StringKey, Int]
    var index_axis: Int
    var column_axis: Int

    fn __init__(inout self, columns: DynamicVector[Int32Array], column_names: DynamicVector[String]) raises:
        # Columns contains a vector of SIMD vectors
        self.columns = DynamicVector[Int32Array]()
        self.column_names = DynamicVector[String]()
        self.col_name_to_idx = Dict[StringKey, Int]()
        self.index_axis = 0
        self.column_axis = 1
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) raises -> Int32Array:
        return self.columns[i]
    
    fn __getitem__(self, column_name: String) raises -> Int32Array:
        let column_index = int((self.col_name_to_idx[column_name]))
        return self.columns[column_index]

    fn __setitem__(inout self, i: Int, value: Int32Array):
        self.columns[i] = value
    
    fn sum(self, axis: Int) raises -> Int32Array:
        var sums = Int32Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                sums[i] = pairwise_sum_i32(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        return sums

@value
struct StringKey(KeyElement):
    var s: String

    fn __init__(inout self, owned s: String):
        self.s = s

    fn __init__(inout self, s: StringLiteral):
        self.s = String(s)

    fn __hash__(self) -> Int:
        let ptr = self.s._buffer.data.value
        return hash(DTypePointer[DType.int8](ptr), len(self.s))

    fn __eq__(self, other: Self) -> Bool:
        return self.s == other.s
