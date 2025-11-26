import "lib/github.com/diku-dk/sorts/radix_sort"
import "segment"

def custom_reduce_by_index 'a [m] [n]  (dest: *[m]a)
                                (f: a -> a -> a) (ne: a)
                                (is: [n]i64) (as: [n]a) : *[m]a =
  let (sorted_is, sorted_as) = radix_sort_int_by_key (\(i, _) -> i) i64.num_bits i64.get_bit (zip is as)
                               |> unzip
  let prev_is = rotate (-1) sorted_is
  let flag_array = map2 (\i_curr i_prev -> i_curr != i_prev) sorted_is prev_is
  let seg_res = zip sorted_as flag_array |> segreduce f ne
  in if length seg_res >= m then map2 (\a b -> f a b) (take m seg_res) dest
     else 
     let pad = replicate (m - length seg_res) ne 
     let inp = seg_res ++ pad
     in map2 (\a b -> f a b) (inp :> [m]a) dest

-- ==
-- entry: test_reduce
-- nobench input { [0,0,0] [0i64, 1i64, 1i64, 2i64] [10, 10, 10, 10] }
-- output { true }
-- nobench input { [0,0,0] [0i64, 1i64, 1i64, 2i64, 2i64] [10, 10, 10, 10, 10] }
-- output { true }
-- nobench input { [0,0,0] [0i64, 1i64, 1i64, 2i64, 2i64] [10, 20, 30, 40, 50] }
-- output { true }
entry test_reduce dest is vs = 
	let builtin = reduce_by_index (copy dest) (+) 0i32 is vs
	let custom = custom_reduce_by_index (copy dest) (+) 0i32 is vs
	in foldl (&&) true <| map2 (\a b -> a == b) (builtin :> [3]i32) (custom :> [3]i32)

-- ==
-- entry: bench_custom_reduce_idx
-- notest random input { [100]i32 [1000]i64 [1000]i32 }
-- notest random input { [1000]i32 [10000]i64 [10000]i32 }
-- notest random input { [10000]i32 [100000]i64 [100000]i32 }
-- notest random input { [100000]i32 [1000000]i64 [1000000]i32 }
-- notest random input { [1000000]i32 [10000000]i64 [10000000]i32 }
-- notest random input { [10000000]i32 [100000000]i64 [100000000]i32 }
entry bench_custom_reduce_idx (dest: *[]i32) is vs = custom_reduce_by_index dest (+) 0i32 is vs 

-- ==
-- entry: bench_reduce_idx
-- notest random input { [100]i32 [1000]i64 [1000]i32 }
-- notest random input { [1000]i32 [10000]i64 [10000]i32 }
-- notest random input { [10000]i32 [100000]i64 [100000]i32 }
-- notest random input { [100000]i32 [1000000]i64 [1000000]i32 }
-- notest random input { [1000000]i32 [10000000]i64 [10000000]i32 }
-- notest random input { [10000000]i32 [100000000]i64 [100000000]i32 }
entry bench_reduce_idx (dest: *[]i32) is vs = reduce_by_index dest (+) 0i32 is vs 
