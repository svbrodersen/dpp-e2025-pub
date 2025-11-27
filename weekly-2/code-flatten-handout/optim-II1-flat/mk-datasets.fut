def mkDataset (m: i64) (q:i64) (p:i64) : ([m]u32, []f32, [m]u32, []f32, [m]f32, [m]i64) =
  let Sa = replicate m q |> map u32.i64
  let Da = tabulate_2d m q 
              (\ i j -> (3*i*i + 5*i*j + 7*j*j + 11*i + 13*j) % (m*q) |> f32.i64 )
        |> flatten 
  --
  let Sb = replicate m p |> map u32.i64
  let Db = tabulate_2d m p (\ i j -> (i*j + i + j) % 57 |> f32.i64)
        |> transpose |> flatten
  --
  let cs  = iota m |> map f32.i64 |> map (\x -> 1 / (x+1))
  --
  let inds= tabulate m (\ i -> (5*i*i*i + 7*i*i + 13*i + 17) % p)
  --
  in  (Sa, Da, Sb, Db, cs, inds)

entry mkResults (m: i64) (q:i64) (p:i64) : [m]f32 =
  let (Sa, Da, Sb, Db, cs, inds) = mkDataset m q p
  let arrEq as bs = map2 (==) as bs |> reduce (&&) true
  let ass = assert (arrEq Sa (replicate m (u32.i64 q))) (Da :> [m*q]f32) |> unflatten
  let bss = assert (arrEq Sb (replicate m (u32.i64 p))) (Db :> [m*p]f32) |> unflatten
  let inBds is = map (\i -> i >= 0 && i < p) is |> reduce (&&) true
  let inds = assert (inBds inds) inds
  in  map4 (\ as bs c ind ->
             let f a = (f32.sqrt a) * bs[ind] + c
             let tmp1 = map f as
             let ioti = iota q
             let iotf = map f32.i64 ioti
             let tmp2 = map2 (+) tmp1 iotf
             in  reduce f32.max f32.lowest tmp2
           ) ass bss cs inds

