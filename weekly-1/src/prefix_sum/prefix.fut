def ilog2 (x:i64) = 63 - i64.i32 (i64.clz x)

def hillis_steele [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
  in loop xs = copy xs for d in iota m do 
    let pow_2 = 2**d
    in map (\i -> if i - pow_2 >= 0 then xs[i-(pow_2)] + xs[i] else xs[i]) (iota n)

def work_efficient [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
  let upswept = 
    loop xs = copy xs for d in reverse (iota m) do
      let pow_2 = 2 ** (m - d - 1)
      in map (\i -> if (i+1) % (2**(m - d)) == 0 then xs[i - pow_2] + xs[i] else xs[i]) (iota n)
    let upswept[n-1] = 0
    in upswept

    

-- Hillis test
-- ==
-- entry: test_hillis
-- nobench input { [0, 0, 1, 0, 0, 0, 0, 0] }
-- output { [0, 0, 1, 1, 1, 1, 1, 1] }
-- nobench input { [0, 1, 1, 0, 0, 0, 0, 0] }
-- output { [0, 1, 2, 2, 2, 2, 2, 2] }
entry test_hillis = hillis_steele


-- Work efficient test
-- ==
-- entry: test_efficient
-- nobench input { [0, 0, 1, 0, 0, 0, 0, 0] }
-- output { [0, 0, 1, 1, 1, 1, 1, 1] }
-- nobench input { [0, 1, 1, 0, 0, 0, 0, 0] }
-- output { [0, 1, 2, 2, 2, 2, 2, 2] }
entry test_efficient = work_efficient
