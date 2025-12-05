-- # Tree operations.
--
-- We import a library so we don't have to write a segmented scan
-- ourselves. Remember to run `futhark pkg sync` to download it.
import "lib/github.com/diku-dk/segmented/segmented"

-- A traversal is an array of these steps.
type step = #u | #d i32

-- ## Input handling.
--
-- You do not have to modify this. The function 'input.steps' takes as
-- argument a string with steps as discussed in the assignment text,
-- and gives you back an array of type '[]step'.
--
-- Example:
--
-- ```
-- > input.steps "d0 d2 d3 u u d5 u"
-- [#d 0, #d 2, #d 3, #u, #u, #d 5, #u]
-- ```


type char = u8
type string [n] = [n]char
module input
  : {
      -- | Parse a string into an array of commands.
      val steps [n] : string [n] -> []step
    } = {
  def is_space (x: char) = x == ' ' || x == '\n'
  def isnt_space x = !(is_space x)

  def (&&&) f g = \x -> (f x, g x)

  def dtoi (c: char) : i32 = i32.u8 c - '0'

  def is_digit (c: char) = c >= '0' && c <= '9'

  def atoi [n] (s: string [n]) : i32 =
    let (sign, s) = if n > 0 && s[0] == '-' then (-1, drop 1 s) else (1, s)
    in sign
       * (loop (acc, i) = (0, 0)
          while i < length s do
            if is_digit s[i]
            then (acc * 10 + dtoi s[i], i + 1)
            else (acc, n)).0

  def to_step (s: []char) : step =
    match s[0]
    case 'u' -> #u
    case _ -> #d (atoi (drop 1 s))

  type slice = (i64, i64)

  def get 't ((start, end): slice) (xs: []t) =
    xs[start:end]

  def words [n] (s: string [n]) : []slice =
    segmented_scan (+) 0 (map is_space s) (map (isnt_space >-> i64.bool) s)
    |> (id &&& rotate 1)
    |> uncurry zip
    |> zip (indices s)
    |> filter (\(i, (x, y)) -> (i == n - 1 && x > 0) || x > y)
    |> map (\(i, (x, _)) -> (i - x + 1, i + 1))

  def steps [n] (s: string [n]) =
    map (\slice -> to_step (get slice s)) (words s)
}



-- ## Task 2.1
def depths (steps: []step) : [](i64, i32) =
  let (arr, values, keep) = unzip3 <| map (\s -> 
    match s
    case #u -> (-1, -1, false)
    case #d x -> (1, x, true)
  ) steps
  let inc_scan_arr = scan (+) 0 arr
  let exc_scan_arr = map2 (\a b -> a - b) inc_scan_arr arr
  let (depth, value, _) = zip3 exc_scan_arr values keep |> filter (\(_, _, b) -> b) |> unzip3
  in zip depth value

-- ## Task 2.2

-- for node i, then parent is the first node on our left, which has a lower value than ourselves.
def parents (D: []i64) : []i64 =
  let n = length D
  let depth_tuple = zip (D :> [n]i64) (iota n)
  let parents = loop parents = [0] for i' < n - 1 do 
    let i = i' + 1
    let search_d = D[i] - 1
    let search_ds = map (\i -> depth_tuple[i]) (iota i)
    let (_, p) = reduce_comm (\(d1, i1) (d2, i2) -> 
      if (d1 == d2) && (d1 == search_d) then 
        if i1 > i2 then (d1, i1)
        else (d2, i2)
      else if d1 == search_d then (d1, i1)
      else if d2 == search_d then (d2, i2)
      else (-1,-1)
    ) (0, -1) search_ds
    in parents ++ [p]
  in parents

-- ## Task 2.3

def subtree_sizes [n] (steps: [n]step) : []i64 =
  let (D, V) = depths steps |> unzip 
  let P = parents D
  let n = length D
  let max_depth = reduce i64.max 0 D
  let (res, _) = loop (res, cur_depth) = (copy V, max_depth) while cur_depth > 0 do
      let indices = filter (\i -> D[i] == cur_depth) (iota n)
      let values = map (\i -> res[i]) indices
      let parent_indices = map (\i -> P[i]) indices
      in trace <| (reduce_by_index res (+) (0) parent_indices values, cur_depth - 1)
  in map i64.i32 res


entry test_depth [n] (inp: string [n]) = input.steps inp |> depths
