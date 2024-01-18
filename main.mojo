from core.Arrays import Float64Array
from core.Calculations import pairwise_sum
from core.DataFrame import DataFrame
from python import Python

fn main() raises:
    test_array_creation()
    print()
    test_array_vector_creation()
    print()
    test_pairwise_sum()
    print()
    test_df_creation()
    print()
    test_df_sum()

fn test_array_creation() raises:
    # Creating a small Float64 array with 2 elements
    var small_arr = Float64Array(2)
    small_arr[0] = 5
    small_arr[1] = 10

    print("Small array with 2 elements")
    print(small_arr[0], small_arr[1])

fn test_array_vector_creation() raises:
    # Creating a vector of two arrays
    var vector = DynamicVector[Float64Array]()
    var arr_ele1 = Float64Array(2)
    var arr_ele2 = Float64Array(2)

    arr_ele1[0] = 5
    arr_ele1[1] = 10
    arr_ele2[0] = 0.2
    arr_ele2[1] = 0.065

    vector.append(arr_ele1)
    vector.append(arr_ele2)

    print("Print elements from the vector of arrays")
    print(vector[0][1], vector[1][1])

fn test_pairwise_sum() raises:
    # Test that pairwise sum works and its compare its accuracy against naive, Numpy, and high precision sum
    let np = Python.import_module("numpy")
    let decimal = Python.import_module("decimal")

    let max_num = 1000
    let size = 10000000
    let small_float = 3.1415926585
    var np_arr = np.random.randint(0, max_num + 1, size)
    np_arr = np_arr.astype(np.float64)
    np_arr /= small_float

    # Use Decimal for high-precision sum
    # Set high precision
    decimal.getcontext().prec = 50

    var decimal_sum = decimal.Decimal('0')
    var naive_sum = SIMD[DType.float64, 1](0)
    var mojo_arr = Float64Array(size)
    let np_sum = np.sum(np_arr)

    for i in range(size):
        decimal_sum += decimal.Decimal(np_arr[i])
        naive_sum += np_arr[i].to_float64()
        mojo_arr[i] = np_arr[i].to_float64()
    
    let pairwise_sum = pairwise_sum(mojo_arr, size, 0, size)

    # Compare
    print("High precicion sum:", decimal_sum)
    print("Naive sum:", naive_sum)
    print("Numpy sum:", np_sum)
    print("Pairwise sum:", pairwise_sum)

fn test_df_creation() raises:
    var col1 = Float64Array(10000)
    var col2 = Float64Array(10000)
    col1[0] = 21
    col2[0] = 23

    var col_data = DynamicVector[Float64Array]()
    col_data.append(col1)
    col_data.append(col2)

    let col1_name = "Units Sold"
    let col2_name = "UID"
    var col_names = DynamicVector[String]()
    col_names.append(col1_name)
    col_names.append(col2_name)
    
    let df = DataFrame(col_data, col_names)

    let df_col1_using_index = df[0][0]
    let df_col2_using_name = df["UID"][0]

    print("DataFrame first column first element:", df_col1_using_index)
    print("DataFrame second column first element:", df_col2_using_name)

fn test_df_sum() raises:
    var col1 = Float64Array(3)
    var col2 = Float64Array(3)
    col1[0] = 1
    col1[1] = 2
    col1[2] = 3

    col2[0] = 3
    col2[1] = 4
    col2[2] = 5

    var col_data = DynamicVector[Float64Array]()
    col_data.append(col1)
    col_data.append(col2)

    let col1_name = "Units Sold"
    let col2_name = "Number of Customers"
    var col_names = DynamicVector[String]()
    col_names.append(col1_name)
    col_names.append(col2_name)
    
    let df = DataFrame(col_data, col_names)
    let df_sums = df.sum(0)

    print("DataFrame 1st column sum:", df_sums[0])
    print("DataFrame 2nd column sum:", df_sums[1])
