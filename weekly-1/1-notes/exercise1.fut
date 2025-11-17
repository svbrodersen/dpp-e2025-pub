-- ==
-- entry: test_process test_process_idx
-- random input { [100]i32 [100]i32 }
-- random input { [1000]i32 [1000]i32 }
-- random input { [10000]i32 [10000]i32 }
-- random input { [100000]i32 [100000]i32 }
-- random input { [1000000]i32 [1000000]i32 }
-- random input { [10000000]i32 [10000000]i32 }

def process (xs: []i32) (ys: []i32): i32 =
  reduce i32.max 0 (map i32.abs (map2 (-) xs ys))

entry test_process = process

def process_idx [n] (xs: [n]i32) (ys: [n]i32): (i32,i64) =
  let max (d1,i1) (d2,i2) =
        if      d1 > d2 then (d1,i1)
        else if d2 > d1 then (d2,i2)
        else if i1 > i2 then (d1,i1)
        else                 (d2,i2)
  in reduce_comm max (0, -1)
                 (zip (map i32.abs (map2 (-) xs ys))
                      (iota n))

entry test_process_idx = test_process
