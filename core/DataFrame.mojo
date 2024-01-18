from python import Python

struct DataFrame:
    var columns: DynamicVector[Float64Array]
    var column_names: DynamicVector[String]
    var col_name_to_idx: python.Dictionary
    var index_axis: Int
    var column_axis: Int

    fn __init__(inout self, columns: DynamicVector[Float64Array], column_names: DynamicVector[String]) raises:
        # Columns should be a list of Arrays, or Array Chunks
        self.columns = DynamicVector[Float64Array]()
        self.column_names = DynamicVector[String]()
        self.col_name_to_idx = Python.dict()
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
        var column_sums = Float64Array(self.columns.__len__())
        if axis == self.index_axis:
            for i in range(self.columns.__len__()):
                column_sums[i] = pairwise_sum(self.columns[i], self.columns[i].size, 0, self.columns[i].size)
        
        return column_sums
            