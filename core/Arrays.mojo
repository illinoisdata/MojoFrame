from python import Python
from tensor import Tensor

@value
struct Float64Array(CollectionElement):
    var data: Tensor[DType.float64]
    var size: Int
    
    fn __init__(inout self, num_elements: Int) raises:
        self.size = num_elements
        self.data = Tensor[DType.float64] (self.size)
        
    fn __copyinit__(inout self, existing: Self):
        self.size = existing.size
        self.data = Tensor[DType.float64] (self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]

    fn __getitem__(self, i: Int) -> SIMD[DType.float64, 1]:
        return self.data[i]

    fn __setitem__(inout self, i: Int, value: SIMD[DType.float64, 1]):
        self.data[i] = value
