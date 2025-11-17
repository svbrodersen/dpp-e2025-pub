def transpose [n][m] 't (xss: [n][m]t) : [m][n]t = 
  let inner = (\i -> map (\j -> xss[j][i]) (iota n))
    in map inner (iota m)

def concat [n] [m] 't (xs: [n]t) (ys: [m]t) : [n + m]t =
   map (\i -> if i < n then xs[i] else ys[i - n])  (iota (n + m))

entry mainTrans [n][m] (xss: [n][m]i32) : [m][n]i32 = 
  transpose xss

entry mainConcat [n][m] (xss: [n]i32) (yss: [m]i32) : [n + m]i32 = 
  concat xss yss

-- Transpose function
-- ==
-- entry: mainTrans
-- input { [[1, 2], [3, 4]] }
-- output { [[1, 3], [2, 4]] }

-- Concat
-- ==
-- entry: mainConcat
-- input { [1, 2] [3, 4] }
-- output { [1, 2, 3, 4] }
