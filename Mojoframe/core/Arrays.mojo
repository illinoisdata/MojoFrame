from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb
from utils.index import Index
from memory import UnsafePointer, memcpy, memset_zero
from utils.numerics import neg_inf
from pathlib import Path
from python import Python

# Array wrapper classes
# DataFrame initialization requires defined type
# @value decorator used to generate Trait methods not used in Array

# @value
# struct Float64Array(Copyable, Movable):
#     var data: Tensor[DType.float64]
#     var size: Int
    
#     fn __init__(mut self, num_elements: Int) raises:
#         self.size = num_elements
#         self.data = Tensor[DType.float64] (self.size)
    
#     fn __init__(mut self, num_elements: Int, as_min: Bool) raises:
#         self.size = num_elements
#         self.data = Tensor[DType.float64] (self.size)
#         var neg_inf = neg_inf[DType.float64]()
#         if as_min:
#             for i in range(self.size):
#                 self.data[i] = neg_inf
                
#     fn __init__(mut self, file_path: String) raises:
#         self.data = Tensor[DType.float64].fromfile(Path(file_path))
#         self.size = self.data.num_elements()

#     fn __copyinit__(mut self, existing: Self):
#         self.size = existing.size
#         self.data = Tensor[DType.float64] (self.size)
#         for i in range(self.size):
#             self.data[i] = existing.data[i]

#     fn __getitem__(self, i: Int) -> SIMD[DType.float64, 1]:
#         return self.data[i]
    
#     fn __setitem__(mut self, i: Int, owned value: SIMD[DType.float64, 1]):
#         self.data[i] = value

# @value
# struct Float32Array(Copyable, Movable):
#     var data: Tensor[DType.float32]
#     var size: Int
    
#     fn __init__(mut self, num_elements: Int) raises:
#         self.size = num_elements
#         self.data = Tensor[DType.float32] (self.size)
        
#     fn __copyinit__(mut self, existing: Self):
#         self.size = existing.size
#         self.data = Tensor[DType.float32] (self.size)
#         for i in range(self.size):
#             self.data[i] = existing.data[i]

#     fn __getitem__(self, i: Int) -> SIMD[DType.float32, 1]:
#         return self.data[i]

#     fn __setitem__(mut self, i: Int, value: SIMD[DType.float32, 1]):
#         self.data[i] = value

# @value
# struct Int32Array(Copyable, Movable):
#     var data: Tensor[DType.int32]
#     var size: Int
    
#     fn __init__(mut self, num_elements: Int) raises:
#         self.size = num_elements
#         self.data = Tensor[DType.int32] (self.size)
        
#     fn __copyinit__(mut self, existing: Self):
#         self.size = existing.size
#         self.data = Tensor[DType.int32] (self.size)
#         for i in range(self.size):
#             self.data[i] = existing.data[i]

#     fn __getitem__(self, i: Int) -> SIMD[DType.int32, 1]:
#         return self.data[i]

#     fn __setitem__(mut self, i: Int, value: SIMD[DType.int32, 1]):
#         self.data[i] = value

struct Int32Array(Copyable, Movable):
    var data: LayoutTensor[mut=True, DType.int32, Layout.col_major(UNKNOWN_VALUE), MutableAnyOrigin]
    var storage: UnsafePointer[Int32]
    var size: Int
    
    fn __init__(out self, num_elements: Int) raises:
        self.size = num_elements
        self.storage = UnsafePointer[Int32].alloc(self.size)
        memset_zero(self.storage, self.size)
        
        alias static_layout = Layout.col_major(UNKNOWN_VALUE)
        var runtime_layout = RuntimeLayout[static_layout].col_major(Index(self.size))
        self.data = LayoutTensor[mut=True, DType.int32, static_layout, MutableAnyOrigin](
            self.storage, runtime_layout
        )
    
    fn __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.storage = UnsafePointer[Int32].alloc(self.size)
        
        alias static_layout = Layout.col_major(UNKNOWN_VALUE)
        var runtime_layout = RuntimeLayout[static_layout].col_major(Index(self.size))
        self.data = LayoutTensor[mut=True, DType.int32, static_layout, MutableAnyOrigin](
            self.storage, runtime_layout
        )
        
        memcpy(self.storage, existing.storage, self.size)


    fn __del__(owned self):
        self.storage.free()

    fn __getitem__(self, i: Int) -> Int32:
        return self.data[i][0]
    
    fn __setitem__(mut self, i: Int, value: Int32):
        self.data[i] = value

    fn unsafe_ptr(self) -> UnsafePointer[Int32]:
        return self.storage

struct Float64Array(Copyable, Movable):
    var data: LayoutTensor[mut=True, DType.float64, Layout.col_major(UNKNOWN_VALUE), MutableAnyOrigin]
    var storage: UnsafePointer[Float64]
    var size: Int
    
    fn __init__(out self, num_elements: Int) raises:
        self.size = num_elements
        self.storage = UnsafePointer[Float64].alloc(self.size)
        memset_zero(self.storage, self.size)
        
        alias static_layout = Layout.col_major(UNKNOWN_VALUE)
        var runtime_layout = RuntimeLayout[static_layout].col_major(Index(self.size))
        self.data = LayoutTensor[mut=True, DType.float64, static_layout](
            self.storage, runtime_layout
        )
    
    fn __init__(out self, file_path: String) raises:
        var file = open(file_path, "r")
        
        var size_bytes = file.read_bytes(8)
        
        # convert first 8 bytes to (array size)
        self.size = size_bytes.unsafe_ptr().bitcast[Int]()[0]
        
        # allocate storage for Float64 values
        self.storage = UnsafePointer[Float64].alloc(self.size)
        
        var data_size_bytes = self.size * 8
        print("data_size_bytes: ", data_size_bytes)
        var data_bytes = file.read_bytes(data_size_bytes)
        
        if len(data_bytes) != data_size_bytes:
            raise Error("Incomplete data in file")
        
        # copy raw bytes directly to Float64 storage
        memcpy(self.storage.bitcast[UInt8](), data_bytes.unsafe_ptr(), data_size_bytes)
        
        alias static_layout = Layout.col_major(UNKNOWN_VALUE)
        var runtime_layout = RuntimeLayout[static_layout].col_major(Index(self.size))
        self.data = LayoutTensor[mut=True, DType.float64, static_layout](
            self.storage, runtime_layout
        )
        
        file.close()
    
    fn __init__(out self, num_elements: Int, as_min: Bool) raises:
        self.size = num_elements
        self.storage = UnsafePointer[Float64].alloc(self.size)
        
        alias static_layout = Layout.col_major(UNKNOWN_VALUE)
        var runtime_layout = RuntimeLayout[static_layout].col_major(Index(self.size))
        self.data = LayoutTensor[mut=True, DType.float64, static_layout](
            self.storage, runtime_layout
        )
        
        if as_min:
            var neg_inf_val = neg_inf[DType.float64]()
            for i in range(self.size):
                self.data[i] = neg_inf_val
                

    fn __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.storage = UnsafePointer[Float64].alloc(self.size)
        
        alias static_layout = Layout.col_major(UNKNOWN_VALUE)
        var runtime_layout = RuntimeLayout[static_layout].col_major(Index(self.size))
        self.data = LayoutTensor[mut=True, DType.float64, static_layout](
            self.storage, runtime_layout
        )
        
        # for i in range(self.size):
        #     self.data[i] = existing.data[i]
        memcpy(self.storage, existing.storage, self.size)

    fn __del__(owned self):
        self.storage.free()

    fn __getitem__(self, i: Int) -> Float64:
        return self.data[i][0]

    fn __setitem__(mut self, i: Int, var value: Float64):
        self.data[i] = value

    fn unsafe_ptr(self) -> UnsafePointer[Float64]:
        return self.storage
    
struct StringArray:
    var data: List[String]
    var size: Int
    # var creation_memory_mb: Float64
    
    fn __init__(out self, file_path: String) raises:
        # var file = open(file_path, "r")
        
        # # read number of strings
        # var count_bytes = file.read_bytes(8)
        # if len(count_bytes) != 8:
        #     raise Error("Cannot read string count")
        
        # self.size = count_bytes.unsafe_ptr().bitcast[Int]()[0]
        # self.data = List[String]()
        
        # for i in range(self.size):
        #     # read each string length
        #     var len_bytes = file.read_bytes(8)
        #     if len(len_bytes) != 8:
        #         raise Error("Cannot read string length")
            
        #     var str_len = len_bytes.unsafe_ptr().bitcast[Int]()[0]
            
        #     # read string data
        #     var str_bytes = file.read_bytes(str_len)
        #     if len(str_bytes) != str_len:
        #         raise Error("Cannot read string data")
            
        #     # convert bytes to String
        #     var span = Span[SIMD[DType.uint8, 1]](ptr=str_bytes.unsafe_ptr(), length=len(str_bytes))
        #     var string_slice = StringSlice(unsafe_from_utf8=span)
        #     var string_val = String(string_slice)

        #     self.data.append(string_val)
        
        # file.close()

        var file = open(file_path, "r")
        var all_bytes = file.read_bytes()
        file.close()
        
        var offset = 0
        # first byte stores number of strings
        self.size = (all_bytes.unsafe_ptr() + offset).bitcast[Int]()[0]
        offset += 8
        
        self.data = List[String](capacity=self.size)
        print("StringArray size: ", self.size)
        
        for i in range(self.size):
            var str_len = (all_bytes.unsafe_ptr() + offset).bitcast[Int]()[0]
            offset += 8
            
            data_span = Span[UInt8](all_bytes.unsafe_ptr() + offset, str_len)
            # Most direct string construction
            var string_val = String(bytes=data_span)

            self.data.append(string_val)
            offset += str_len
    
