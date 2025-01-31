from max.tensor import Tensor
from utils.numerics import neg_inf

# Array wrapper classes
# DataFrame initialization requires defined type
# @value decorator used to generate Trait methods not used in Array

@value
struct Float64Array(CollectionElement):
    var data: Tensor[DType.float64]
    var size: Int
    
    fn __init__(inout self, num_elements: Int) raises:
        self.size = num_elements
        self.data = Tensor[DType.float64] (self.size)
    
    fn __init__(inout self, num_elements: Int, as_min: Bool) raises:
        self.size = num_elements
        self.data = Tensor[DType.float64] (self.size)
        var neg_inf = neg_inf[DType.float64]()
        if as_min:
            for i in range(self.size):
                self.data[i] = neg_inf

    fn __copyinit__(inout self, existing: Self):
        self.size = existing.size
        self.data = Tensor[DType.float64] (self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]

    fn __getitem__(self, i: Int) -> SIMD[DType.float64, 1]:
        return self.data[i]
    
    fn __setitem__(inout self, i: Int, owned value: SIMD[DType.float64, 1]):
        self.data[i] = value

@value
struct Float32Array(CollectionElement):
    var data: Tensor[DType.float32]
    var size: Int
    
    fn __init__(inout self, num_elements: Int) raises:
        self.size = num_elements
        self.data = Tensor[DType.float32] (self.size)
        
    fn __copyinit__(inout self, existing: Self):
        self.size = existing.size
        self.data = Tensor[DType.float32] (self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]

    fn __getitem__(self, i: Int) -> SIMD[DType.float32, 1]:
        return self.data[i]

    fn __setitem__(inout self, i: Int, value: SIMD[DType.float32, 1]):
        self.data[i] = value

@value
struct Int32Array(CollectionElement):
    var data: Tensor[DType.int32]
    var size: Int
    
    fn __init__(inout self, num_elements: Int) raises:
        self.size = num_elements
        self.data = Tensor[DType.int32] (self.size)
        
    fn __copyinit__(inout self, existing: Self):
        self.size = existing.size
        self.data = Tensor[DType.int32] (self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]

    fn __getitem__(self, i: Int) -> SIMD[DType.int32, 1]:
        return self.data[i]

    fn __setitem__(inout self, i: Int, value: SIMD[DType.int32, 1]):
        self.data[i] = value
    