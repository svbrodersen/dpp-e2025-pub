def ilog2 (x:i64) = 63 - i64.i32 (i64.clz x)

def hillis_steele [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
  in loop xs = copy xs for d in 0...m-1 do 
    let pow_2 = 1 << d
    in map (\i -> if i - pow_2 >= 0 then xs[i-(pow_2)] + xs[i] else xs[i]) (iota n)

def work_efficient [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
  let upswept = 
    loop xs = copy xs for d in 0...m-1 do
      let stride = 1 << (d + 1) 
      let offset = 1 << d
      let idxs = map (\i -> i + stride - 1) (0..stride...n-1)
      -- j = i + 2^(d + 1) - 1 => j - 2^d = i + 2^d - 1, from the Guy paper
      let vals = map (\j -> xs[j - offset] + xs[j]) idxs
      in scatter xs idxs vals
  let upswept[n-1] = 0
  let downswept = 
    loop xs = upswept for d in m-1..m-2...0 do
      let stride = 1 << (d + 1) 
      let offset = 1 << d
      let (left_idxs, right_idxs) = map (\i -> 
        let left_idx = i + offset - 1
        in (left_idx, left_idx + offset)) (0..stride...n-1) |> unzip
      let left_vals = map (\r -> xs[r]) right_idxs
      -- First item is t from paper
      let right_vals = map2 (\l r -> xs[l] + xs[r]) left_idxs right_idxs
      let all_idxs = left_idxs ++ right_idxs
      let all_vals = left_vals ++ right_vals
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
-- notest random input { [1024]i32 }
-- notest random input { [2048]i32 }
-- notest random input { [4096]i32 }
-- notest random input { [8192]i32 }
-- notest random input { [16384]i32 }
-- notest random input { [32768]i32 }
-- notest random input { [65536]i32 }
-- notest random input { [131072]i32 }
-- notest random input { [262144]i32 }
-- notest random input { [524288]i32 }
-- notest random input { [1048576]i32 }
-- notest random input { [2097152]i32 }
-- notest random input { [4194304]i32 }
-- notest random input { [8388608]i32 }
-- notest random input { [16777216]i32 }
-- notest random input { [33554432]i32 }
-- notest random input { [67108864]i32 }
-- notest random input { [134217728]i32 }
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
-- notest random input { [1024]i32 }
-- notest random input { [2048]i32 }
-- notest random input { [4096]i32 }
-- notest random input { [8192]i32 }
-- notest random input { [16384]i32 }
-- notest random input { [32768]i32 }
-- notest random input { [65536]i32 }
-- notest random input { [131072]i32 }
-- notest random input { [262144]i32 }
-- notest random input { [524288]i32 }
-- notest random input { [1048576]i32 }
-- notest random input { [2097152]i32 }
-- notest random input { [4194304]i32 }
-- notest random input { [8388608]i32 }
-- notest random input { [16777216]i32 }
-- notest random input { [33554432]i32 }
-- notest random input { [67108864]i32 }
-- notest random input { [134217728]i32 }
entry test_efficient = work_efficient

-- Built in scan
-- ==
-- entry: test_scan
-- notest random input { [1024]i32 }
-- notest random input { [2048]i32 }
-- notest random input { [4096]i32 }
-- notest random input { [8192]i32 }
-- notest random input { [16384]i32 }
-- notest random input { [32768]i32 }
-- notest random input { [65536]i32 }
-- notest random input { [131072]i32 }
-- notest random input { [262144]i32 }
-- notest random input { [524288]i32 }
-- notest random input { [1048576]i32 }
-- notest random input { [2097152]i32 }
-- notest random input { [4194304]i32 }
-- notest random input { [8388608]i32 }
-- notest random input { [16777216]i32 }
-- notest random input { [33554432]i32 }
-- notest random input { [67108864]i32 }
-- notest random input { [134217728]i32 }
entry test_scan = scan (+) 0 
