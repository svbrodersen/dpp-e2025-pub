def ilog2 (x:i64) = 63 - i64.i32 (i64.clz x)

def hillis_steele [n] (xs: [n]i32) : [n]i32 = 
  let m = ilog2 n
