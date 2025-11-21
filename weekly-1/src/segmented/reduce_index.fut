import "lib/github.com/diku-dk/sorts/radix_sort"
import "segment"

def reduce_by_index 'a [m] [n]  (dest: *[m]a)
                                (f: a -> a -> a) (ne: a)
                                (is: [n]i64) (as: [n]a) : *[m]a =
  let (sorted_is, sorted_as) = radix_sort_int_by_key (\(i, _) -> i) i64.num_bits i64.get_bit (zip is as)
                               |> unzip
  let prev_is = rotate 1 sorted_is
  let flag_array = map2 (\i_curr i_prev -> i_curr != i_prev) sorted_is prev_is
  let seg_res = zip sorted_as flag_array |> segreduce f ne 
  in if length seg_res > m then map2 (\a b -> f a b) (take m seg_res) dest
     else 
     let pad = replicate (m - length seg_res) ne 
     let inp = seg_res ++ pad
     in map2 (\a b -> f a b) (inp :> [m]a) dest
