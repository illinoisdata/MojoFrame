fn pairwise_sum_f64(arr: Float64Array, n: Int, start: Int, stop: Int) -> SIMD[DType.float64, 1]:
    if n < 8:
        var res = SIMD[DType.float64, 1](0)
        for i in range(start, stop):
            res += arr[i]
        return res
    
    elif n <= 128:
        var r0 = arr[start+0]
        var r1 = arr[start+1]
        var r2 = arr[start+2]
        var r3 = arr[start+3]
        var r4 = arr[start+4]
        var r5 = arr[start+5]
        var r6 = arr[start+6]
        var r7 = arr[start+7]

        let m = stop - (stop % 8)
        for i in range(start + 8, m, 8):
            r0 += arr[i]
            r1 += arr[i+1]
            r2 += arr[i+2]
            r3 += arr[i+3]
            r4 += arr[i+4]
            r5 += arr[i+5]
            r6 += arr[i+6]
            r7 += arr[i+7]
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

        for i in range(m, stop):
            res += arr[i]

        return res
    else:
        var n2 = n // 2
        n2 -= (n2 % 8)
        let middle = start + n2
        return (pairwise_sum_f64(arr, n2, start, middle)
                + pairwise_sum_f64(arr, n - n2, middle, stop))
                

fn pairwise_sum_f32(arr: Float32Array, n: Int, start: Int, stop: Int) -> SIMD[DType.float32, 1]:
    if n < 8:
        var res = SIMD[DType.float32, 1](0)
        for i in range(start, stop):
            res += arr[i]
        return res
    
    elif n <= 128:
        var r0 = arr[start+0]
        var r1 = arr[start+1]
        var r2 = arr[start+2]
        var r3 = arr[start+3]
        var r4 = arr[start+4]
        var r5 = arr[start+5]
        var r6 = arr[start+6]
        var r7 = arr[start+7]

        let m = stop - (stop % 8)
        for i in range(start + 8, m, 8):
            r0 += arr[i]
            r1 += arr[i+1]
            r2 += arr[i+2]
            r3 += arr[i+3]
            r4 += arr[i+4]
            r5 += arr[i+5]
            r6 += arr[i+6]
            r7 += arr[i+7]
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

        for i in range(m, stop):
            res += arr[i]

        return res
    else:
        var n2 = n // 2
        n2 -= (n2 % 8)
        let middle = start + n2
        return (pairwise_sum_f32(arr, n2, start, middle)
                + pairwise_sum_f32(arr, n - n2, middle, stop))

fn pairwise_sum_i32(arr: Int32Array, n: Int, start: Int, stop: Int) -> SIMD[DType.int32, 1]:
    if n < 8:
        var res = SIMD[DType.int32, 1](0)
        for i in range(start, stop):
            res += arr[i]
        return res
    
    elif n <= 128:
        var r0 = arr[start+0]
        var r1 = arr[start+1]
        var r2 = arr[start+2]
        var r3 = arr[start+3]
        var r4 = arr[start+4]
        var r5 = arr[start+5]
        var r6 = arr[start+6]
        var r7 = arr[start+7]

        let m = stop - (stop % 8)
        for i in range(start + 8, m, 8):
            r0 += arr[i]
            r1 += arr[i+1]
            r2 += arr[i+2]
            r3 += arr[i+3]
            r4 += arr[i+4]
            r5 += arr[i+5]
            r6 += arr[i+6]
            r7 += arr[i+7]
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

        for i in range(m, stop):
            res += arr[i]

        return res
    else:
        var n2 = n // 2
        n2 -= (n2 % 8)
        let middle = start + n2
        return (pairwise_sum_i32(arr, n2, start, middle)
                + pairwise_sum_i32(arr, n - n2, middle, stop))
                