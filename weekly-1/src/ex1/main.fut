def process [n] (xs: [n]i32) (ys: [n]i32) : i32 = 
  reduce i32.max 0 (map i32.abs (map2 (-) xs ys))

entry test_process = process

-- Process tests
-- ==
-- entry: test_process
-- nobench input { [23 ,45 , -23 ,44 ,23 ,54 ,23 ,12 ,34 ,54 ,7 ,2 , 4 ,67] [ -2 , 3 , 4 ,57 ,34 , 2 , 5 ,56 ,56 , 3 ,3 ,5 ,77 ,89] }
-- output { 73 }
-- nobench input { empty([0]i32) empty([0]i32) }
-- output { 0 }
-- random input { [100]i32 [100]i32 }
-- random input { [1000]i32 [1000]i32 }
-- random input { [10000]i32 [10000]i32 }
-- random input { [100000]i32 [100000]i32 }
-- random input { [1000000]i32 [1000000]i32 }
-- random input { [10000000]i32 [10000000]i32 }

