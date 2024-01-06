from core.Arrays import Float64Array
from core.Calculations import pairwise_sum
from core.DataFrame import DataFrame
from python import Python

fn main() raises:
    test_array_creation()
    test_array_vector_creation()

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

fn test_pairwise_sum():
    pass

fn test_df_creation():
    pass

fn test_df_sum():
    pass
