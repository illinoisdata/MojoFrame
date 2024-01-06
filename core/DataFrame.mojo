from python import Python

struct DataFrame:
    var columns: DynamicVector[Float64Array]
    var column_names: DynamicVector[String]
    var col_name_to_idx: python.Dictionary

    fn __init__(inout self, columns: DynamicVector[Float64Array], column_names: DynamicVector[String]) raises:
        # Columns should be a list of Arrays, or Array Chunks
        self.columns = DynamicVector[Float64Array]()
        self.column_names = DynamicVector[String]()
        self.col_name_to_idx = Python.dict()
        
        for i in range(columns.size):
            self.columns.append(columns[i])
            self.column_names.append(column_names[i])
            self.col_name_to_idx[column_names[i]] = i

    fn __getitem__(self, i: Int) -> Float64Array:
        return self.columns[i]

    fn __setitem__(inout self, i: Int, value: Float64Array):
        self.columns[i] = value
