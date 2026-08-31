from std.sys.info import size_of
from std.memory import unsafe_memcpy


def _resolve_data_path(file_path: String) -> String:
    var prefix = "/datadrive/tpch_large/"
    if file_path.startswith(prefix):
        var fname = file_path[byte=len(prefix.bytes()):]
        return "Data/" + fname
    return file_path


struct Float64Array(ImplicitlyCopyable, Copyable, Movable):
    var data: List[Float64]
    var size: Int

    def __init__(out self):
        self.size = 0
        self.data = List[Float64]()

    def __init__(out self, num_elements: Int):
        self.size = num_elements
        self.data = List[Float64]()
        self.data.resize(num_elements, 0.0)

    def __init__(out self, num_elements: Int, as_min: Bool):
        self.size = num_elements
        self.data = List[Float64]()
        var val: Float64 = -Float64.MAX_FINITE if as_min else 0.0
        self.data.resize(num_elements, val)

    def __init__(out self, file_path: String) raises:
        var path = file_path
        try:
            with open(path, "r") as f:
                var bytes_list = f.read_bytes()
                var byte_len = len(bytes_list)
                self.size = byte_len // size_of[Float64]()
                self.data = List[Float64]()
                self.data.resize(self.size, 0.0)
                if self.size > 0:
                    var src_ptr = bytes_list.unsafe_ptr().unsafe_bitcast[Float64]()
                    var dst_ptr = self.data.unsafe_ptr()
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=self.size)
        except:
            path = _resolve_data_path(file_path)
            with open(path, "r") as f:
                var bytes_list = f.read_bytes()
                var byte_len = len(bytes_list)
                self.size = byte_len // size_of[Float64]()
                self.data = List[Float64]()
                self.data.resize(self.size, 0.0)
                if self.size > 0:
                    var src_ptr = bytes_list.unsafe_ptr().unsafe_bitcast[Float64]()
                    var dst_ptr = self.data.unsafe_ptr()
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=self.size)

    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.size = move.size
        self.data = move.data^

    def __getitem__(self, i: Int) -> Float64:
        return self.data[i]

    def __setitem__(mut self, i: Int, value: Float64):
        self.data[i] = value

    def num_elements(self) -> Int:
        return self.size

    def unsafe_ptr(self) -> Pointer[Float64, origin_of(self.data)]:
        return self.data.unsafe_ptr()

    def load[width: Int](self, idx: Int) -> SIMD[DType.float64, width]:
        return self.data.unsafe_ptr().unsafe_load[width=width](idx)

    def store[width: Int](mut self, idx: Int, val: SIMD[DType.float64, width]):
        self.data.unsafe_ptr().unsafe_store[width=width](idx, val)


struct Float32Array(ImplicitlyCopyable, Copyable, Movable):
    var data: List[Float32]
    var size: Int

    def __init__(out self):
        self.size = 0
        self.data = List[Float32]()

    def __init__(out self, num_elements: Int):
        self.size = num_elements
        self.data = List[Float32]()
        self.data.resize(num_elements, 0.0)

    def __init__(out self, num_elements: Int, as_min: Bool):
        self.size = num_elements
        self.data = List[Float32]()
        var val: Float32 = -Float32.MAX_FINITE if as_min else 0.0
        self.data.resize(num_elements, val)

    def __init__(out self, file_path: String) raises:
        var path = file_path
        try:
            with open(path, "r") as f:
                var bytes_list = f.read_bytes()
                var byte_len = len(bytes_list)
                self.size = byte_len // size_of[Float32]()
                self.data = List[Float32]()
                self.data.resize(self.size, 0.0)
                if self.size > 0:
                    var src_ptr = bytes_list.unsafe_ptr().unsafe_bitcast[Float32]()
                    var dst_ptr = self.data.unsafe_ptr()
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=self.size)
        except:
            path = _resolve_data_path(file_path)
            with open(path, "r") as f:
                var bytes_list = f.read_bytes()
                var byte_len = len(bytes_list)
                self.size = byte_len // size_of[Float32]()
                self.data = List[Float32]()
                self.data.resize(self.size, 0.0)
                if self.size > 0:
                    var src_ptr = bytes_list.unsafe_ptr().unsafe_bitcast[Float32]()
                    var dst_ptr = self.data.unsafe_ptr()
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=self.size)

    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.size = move.size
        self.data = move.data^

    def __getitem__(self, i: Int) -> Float32:
        return self.data[i]

    def __setitem__(mut self, i: Int, value: Float32):
        self.data[i] = value

    def num_elements(self) -> Int:
        return self.size

    def unsafe_ptr(self) -> Pointer[Float32, origin_of(self.data)]:
        return self.data.unsafe_ptr()

    def load[width: Int](self, idx: Int) -> SIMD[DType.float32, width]:
        return self.data.unsafe_ptr().unsafe_load[width=width](idx)

    def store[width: Int](mut self, idx: Int, val: SIMD[DType.float32, width]):
        self.data.unsafe_ptr().unsafe_store[width=width](idx, val)


struct Int32Array(ImplicitlyCopyable, Copyable, Movable):
    var data: List[Int32]
    var size: Int

    def __init__(out self):
        self.size = 0
        self.data = List[Int32]()

    def __init__(out self, num_elements: Int):
        self.size = num_elements
        self.data = List[Int32]()
        self.data.resize(num_elements, 0)

    def __init__(out self, file_path: String) raises:
        var path = file_path
        try:
            with open(path, "r") as f:
                var bytes_list = f.read_bytes()
                var byte_len = len(bytes_list)
                self.size = byte_len // size_of[Int32]()
                self.data = List[Int32]()
                self.data.resize(self.size, 0)
                if self.size > 0:
                    var src_ptr = bytes_list.unsafe_ptr().unsafe_bitcast[Int32]()
                    var dst_ptr = self.data.unsafe_ptr()
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=self.size)
        except:
            path = _resolve_data_path(file_path)
            with open(path, "r") as f:
                var bytes_list = f.read_bytes()
                var byte_len = len(bytes_list)
                self.size = byte_len // size_of[Int32]()
                self.data = List[Int32]()
                self.data.resize(self.size, 0)
                if self.size > 0:
                    var src_ptr = bytes_list.unsafe_ptr().unsafe_bitcast[Int32]()
                    var dst_ptr = self.data.unsafe_ptr()
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=self.size)

    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.size = move.size
        self.data = move.data^

    def __getitem__(self, i: Int) -> Int32:
        return self.data[i]

    def __setitem__(mut self, i: Int, value: Int32):
        self.data[i] = value

    def num_elements(self) -> Int:
        return self.size

    def unsafe_ptr(self) -> Pointer[Int32, origin_of(self.data)]:
        return self.data.unsafe_ptr()

    def load[width: Int](self, idx: Int) -> SIMD[DType.int32, width]:
        return self.data.unsafe_ptr().unsafe_load[width=width](idx)

    def store[width: Int](mut self, idx: Int, val: SIMD[DType.int32, width]):
        self.data.unsafe_ptr().unsafe_store[width=width](idx, val)