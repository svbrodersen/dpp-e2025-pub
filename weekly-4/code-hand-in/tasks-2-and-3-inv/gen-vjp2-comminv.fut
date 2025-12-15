def imap xs f = map f xs
def imap2 xs ys f = map2 f xs ys
def imap3 xs ys zs f = map3 f xs ys zs

-----------------------------------------------------------------
-- | Task 2:
--     Implement the generic, optimized differentiation
--       of reduce given that the operator is commutative
--       and invertible (i.e., the inverse operator is known).
--     Function arguments:
--       `op` is the commutative and associative original
--            operator of reduce,
--       `op_lft` is the lifted operator,
--       `ne` is the neutral element of `op_lft`
--       `cfwd` and `cbwd` convert between the original
--            type `t` and the lifted type `t_lft`
--       `op_inv` is the inverse operator
--
--       `as` is the input (to-be-reduced) array
--       `rs_bar` is the adjoint of reduce's result
--     Result:
--       the result of the primal tupled with
--       the adjoint of the input (`as`)
--     Meaning:
--       your code should be equivalent with:
--         `vjp2 (reduce op (cbwd ne)) as rs_bar`
--     Hint:
--       you probably would like to consult in file
--       `L7and8-AD.pdf` the slides entitled
--       "Special Case: Multiplication" and
--       "Generalization based on Invertible Operators"
--
def vjp2_red_inv 't 't_lft [n]
               (op : t -> t -> t)
               (op_lft : t_lft -> t_lft -> t_lft)
               (ne: t_lft)
               (cfwd : t -> t_lft)
               (cbwd : t_lft -> t)
               (op_inv : t_lft -> t -> t)
               (as : [n]t)
               (rs_bar: t)
             : (t, [n]t) =
  -- please replace the dummy code below with your correct implementation
  let t_prim = map cfwd as |> reduce_comm (op_lft) ne
  let y = cbwd t_prim
  let as_bar = map (\a ->
    let p' = op_inv t_prim a
    in vjp (\x -> op x  p') a rs_bar
  ) as
  in (y, as_bar)

----------------------------------------------------------------------
-- | Task 3:
--     Implement the generic, optimized differentiation
--       of reduce-by-index given that the operator is
--       invertible (i.e., the inverse operator is known).
-- 
--     Function arguments:
--       `op` is the commutative and associative original
--            operator of reduce-by-index,
--       `op_lft` is the lifted operator,
--       `ne` is the neutral element of `op_lft`
--       `cfwd` and `cbwd` convert between the original
--            type `t` and the lifted type `t_lft`
--       `op_inv` is the inverse operator
--
--       `dst` is the destination of the reduce-by-index
--       `ks`  records the bin indices corresponding to
--               each element of the input
--       `vs` is the input (data) array
--       `rs_bar` is the adjoint of reduce-by-index result
--
--     Result:
--       the result of the primal (original program) tupled with
--       the adjoint of the `dst` array (that is "overwritten")
--       the adjoint of the input (`vs`)
--
--     Meaning:
--       the original (to-be-differentiated) program is
--       semantically equivalent with:
--         `reduce_by_index op (cbwd ne) dst ks vs` 
--
--     Hint:
--       you probably would like to consult in file
--       `L7and8-AD.pdf` the slides entitled
--       "Differentiating General-Case Reduce-By-Index", and,
--       more importantly,
--       "Differentiating Red-by-Index with Invertible Operator"
--
def vjp_redbyind_inv 't 't_lft [m] [n]
               (op : t -> t -> t)
               (op_lft : t_lft -> t_lft -> t_lft)
               (ne: t_lft)
               (cfwd : t -> t_lft)
               (cbwd : t_lft -> t)
               (op_inv : t_lft -> t -> t)
            -- \^ function params
               (dst: [m]t) (ks: [n]i64) (vs: [n]t) (rs_bar: [m]t)
             : ([m]t, [m]t, [n]t) =
  let hsL = map cfwd vs |> hist op_lft ne m ks
  let hs = map cbwd hsL
  let rs = map2 op dst hs
  let g d h r_bar = vjp (\x -> op x d) h r_bar
  let hs_bar = map3 g dst hs rs_bar
  let dst_bar = map3 g hs dst rs_bar
  let f k v = let p' = op_inv hsL[k] v in vjp (\x -> op x p') v hs_bar[k]
  let vs_bar = map2 f ks vs
  in (rs, dst_bar, vs_bar)

