def process [n] (xs: [n]i32) (ys: [n]i32) : i32 = 
  reduce i32.max 0 (map i32.abs (map2 (-) xs ys))

def process_idx [n] (xs: [n]i32) (ys: [n]i32) : (i32, i64) = 
  let max (d1, i1) (d2, i2) =
      if d1 > d2 then (d1, i1)
      else if  d1 < d2 then (d2, i2)
      else if i1 > i2 then (d1, i1)
      else (d2, i2)
    in reduce_comm max (0, -1) 
      (zip
        (map i32.abs (map2 (-) xs ys)) 
        (iota n)
      )

-- Process tests
-- ==
-- entry: test_process
-- nobench input { [23 ,45 , -23 ,44 ,23 ,54 ,23 ,12 ,34 ,54 ,7 ,2 , 4 ,67] [ -2 , 3 , 4 ,57 ,34 , 2 , 5 ,56 ,56 , 3 ,3 ,5 ,77 ,89] }
-- output { 73 }
-- nobench input { empty([0]i32) empty([0]i32) }
-- output { 0 }
-- notest random input { [100]i32 [100]i32 }
-- notest random input { [1000]i32 [1000]i32 }
-- notest random input { [10000]i32 [10000]i32 }
-- notest random input { [100000]i32 [100000]i32 }
-- notest random input { [1000000]i32 [1000000]i32 }
-- notest random input { [10000000]i32 [10000000]i32 }
entry test_process = process

-- Process idx tests
-- ==
-- entry: test_process_idx
-- nobench input { [23 ,45 , -23 ,44 ,23 ,54 ,23 ,12 ,34 ,54 ,7 ,2 , 4 ,67] [ -2 , 3 , 4 ,57 ,34 , 2 , 5 ,56 ,56 , 3 ,3 ,5 ,77 ,89] }
-- output { 73 12i64 }
-- nobench input { empty([0]i32) empty([0]i32) }
-- output { 0 -1i64 }
-- notest random input { [100]i32 [100]i32 }
-- notest random input { [1000]i32 [1000]i32 }
-- notest random input { [10000]i32 [10000]i32 }
-- notest random input { [100000]i32 [100000]i32 }
-- notest random input { [1000000]i32 [1000000]i32 }
-- notest random input { [10000000]i32 [10000000]i32 }
entry test_process_idx = process_idx


