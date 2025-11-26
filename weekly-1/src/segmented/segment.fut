def segscan [n] 't (op: t -> t -> t) (ne: t) 
                   (arr: [n](t, bool)): [n]t = 
  let (res, _) = scan (\(v1, f1) (v2, f2) -> 
                        (if f2 then v2 else op v1 v2, f1 || f2)
                      ) (ne, false) arr 
                      |> unzip
  in res

def segreduce [n] 't (op: t -> t -> t) (ne: t) 
                     (arr: [n](t, bool)): []t = 
  let res = segscan op ne arr
  let (_, flags) = unzip arr
  -- exclusive scan 
  let keep = map (\i ->
    if i == n-1 then 1
    else if flags[i + 1] then 1 
    else 0
  ) (iota n)
  let offsets1 = scan (+) 0 keep
  in if (length offsets1) == 0
    then []
    else scatter (replicate (last offsets1) ne) 
                 (map2 (\i k -> if k == 1 then i-1 else -1) offsets1 keep) res

-- Segmented scan tests
-- ==
-- entry: test_segscan
-- nobench input { [1i32, 2i32, 3i32, 4i32, 5i32] 
--         [false, false, true, false, false] 
--         0i32 }
-- output { [1i32, 3i32, 3i32, 7i32, 12i32] }
-- nobench input { [5i32, 10i32, 15i32, 20i32] 
--         [false, false, false, false] 
--         0i32 }
-- output { [5i32, 15i32, 30i32, 50i32] }
-- nobench input { [1i32, 2i32, 3i32, 4i32] 
--         [true, true, true, true] 
--         0i32 }
-- output { [1i32, 2i32, 3i32, 4i32] }
-- nobench input { [10i32, 20i32, 30i32, 40i32, 50i32, 60i32] 
--         [false, false, true, false, true, false] 
--         0i32 }
-- output { [10i32, 30i32, 30i32, 70i32, 50i32, 110i32] }
entry test_segscan (inp: []i32) (flags: []bool) (ne: i32)  = segscan (+) ne (zip inp flags)

-- ==
-- entry: test_segreduce
-- nobench input { [1i32, 2i32, 3i32, 4i32, 5i32]
--         [false, false, true, false, false]
--         0i32 }
-- output { [3i32, 12i32] }
-- nobench input { [5i32, 10i32, 15i32, 20i32] 
--         [false, false, false, false] 
--         0i32 }
-- output { [50i32] }
-- nobench input { [1i32, 2i32, 3i32, 4i32] 
--         [true, true, true, true] 
--         0i32 }
-- output { [1i32, 2i32, 3i32, 4i32] }
-- nobench input { [10i32, 20i32, 30i32, 40i32, 50i32, 60i32] 
--         [false, false, true, false, true, false] 
--         0i32 }
-- output { [30i32, 70i32, 110i32] }
-- nobench input { [100i32] 
--         [false] 
--         0i32 }
-- output { [100i32] }
-- nobench input { [7i32, 8i32, 9i32] 
--         [false, true, false] 
--         0i32 }
-- output { [7i32, 17i32] }
entry test_segreduce (inp: []i32) (flags: []bool) (ne: i32)  = segreduce (+) ne (zip inp flags)


-- Segscan bench
-- ==
-- entry: bench_segscan
-- notest random input { [100]i32 [100]bool }
-- notest random input { [1000]i32 [1000]bool }
-- notest random input { [10000]i32 [10000]bool }
-- notest random input { [100000]i32 [100000]bool }
-- notest random input { [1000000]i32 [1000000]bool }
-- notest random input { [10000000]i32 [10000000]bool }
entry bench_segscan (inp: []i32) (flags: []bool) = segscan (+) 0 (zip inp flags)


-- scan bench
-- ==
-- entry: bench_scan
-- notest random input { [100]i32 }
-- notest random input { [1000]i32 }
-- notest random input { [10000]i32 }
-- notest random input { [100000]i32 }
-- notest random input { [1000000]i32 }
-- notest random input { [10000000]i32 }
entry bench_scan (inp: []i32) = scan (+) 0 inp 


-- Segscan bench
-- ==
-- entry: bench_segreduce
-- notest random input { [100]i32 [100]bool }
-- notest random input { [1000]i32 [1000]bool }
-- notest random input { [10000]i32 [10000]bool }
-- notest random input { [100000]i32 [100000]bool }
-- notest random input { [1000000]i32 [1000000]bool }
-- notest random input { [10000000]i32 [10000000]bool }
entry bench_segreduce (inp: []i32) (flags: []bool) = segreduce (+) 0 (zip inp flags)


-- scan bench
-- ==
-- entry: bench_reduce
-- notest random input { [100]i32 }
-- notest random input { [1000]i32 }
-- notest random input { [10000]i32 }
-- notest random input { [100000]i32 }
-- notest random input { [1000000]i32 }
-- notest random input { [10000000]i32 }
entry bench_reduce (inp: []i32) = reduce (+) 0 inp 
