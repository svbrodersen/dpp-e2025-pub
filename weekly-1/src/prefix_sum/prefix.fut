def ilog2 (x:i64) = 63 - i64.i32 (i64.clz x)

def hillis_steele [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
  in loop xs = copy xs for d in iota m do 
    let pow_2 = 2**d
    in map (\i -> if i - pow_2 >= 0 then xs[i-(pow_2)] + xs[i] else xs[i]) (iota n)

def work_efficient [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
  let upswept = 
    loop xs = copy xs for d in iota m do
      let stride = 2 ** (d + 1)
      let offset = 2 ** d
      let num_idx = n / stride
      let idxs = map (\i -> (i + 1) * stride - 1) (iota num_idx)
      -- j = i + 2^(d + 1) - 1 => j - 2^d = i + 2^d - 1, from the Guy paper
      let vals = map (\j -> xs[j - offset] + xs[j]) idxs
      in scatter xs idxs vals
  let upswept[n-1] = 0
  let downswept = 
    loop xs = upswept for d in reverse (iota m) do
      let stride = 2**(d + 1)
      let offset = 2**d
      let num_idx = n / stride
      let iter_arr = iota (2 * num_idx)
      let all_idxs = map (\k ->
        let i = k / 2
        let is_left = k % 2 == 0
        let left_idx = i * stride + offset - 1
        let right_idx = left_idx + offset
        in if is_left then left_idx else right_idx
      ) iter_arr
      let all_vals = map (\k ->
        let i = k / 2
        let is_left = k % 2 == 0
        let left_idx = i * stride + offset - 1
        let right_idx = left_idx + offset
        in if is_left then xs[right_idx] else xs[left_idx] + xs[right_idx]
      ) iter_arr
      in scatter xs all_idxs all_vals
  in downswept
      
-- Hillis test
-- ==
-- entry: test_hillis
-- nobench input { [0, 0, 1, 0, 0, 0, 0, 0] }
-- output { [0, 0, 1, 1, 1, 1, 1, 1] }
-- nobench input { [0, 1, 1, 0, 0, 0, 0, 0] }
-- output { [0, 1, 2, 2, 2, 2, 2, 2] }
-- nobench input { [3, 1, 7, 0, 4, 1, 6, 3] }
-- output { [3, 4, 11, 11, 15, 16, 22, 25] }
-- notest random input { [100]i32 }
-- notest random input { [1000]i32 }
-- notest random input { [10000]i32 }
-- notest random input { [100000]i32 }
-- notest random input { [1000000]i32 }
-- notest random input { [10000000]i32 }
entry test_hillis = hillis_steele


-- Work efficient test
-- ==
-- entry: test_efficient
-- nobench input { [0, 0, 1, 0, 0, 0, 0, 0] }
-- output { [0, 0, 0, 1, 1, 1, 1, 1] }
-- nobench input { [0, 1, 1, 0, 0, 0, 0, 0] }
-- output { [0, 0, 1, 2, 2, 2, 2, 2] }
-- nobench input { [3, 1, 7, 0, 4, 1, 6, 3] }
-- output { [0, 3, 4, 11, 11, 15, 16, 22] }
-- notest random input { [100]i32}
-- notest random input { [1000]i32}
-- notest random input { [10000]i32}
-- notest random input { [100000]i32}
-- notest random input { [1000000]i32}
-- notest random input { [10000000]i32}
-- notest random input { [100000000]i32}
entry test_efficient = work_efficient
