import "gen-vjp2-comminv"
import "ops"
import "red-adj-ppad"

def plus_tup (a1: f64, a2: f64) (b1: f64, b2: f64) = (a1+b1, a2+b2)
def zero_tup = (0f64, 0f64)

def equalEps [n] (eps: f64) (as: [n]f64) (bs: [n]f64) : bool =
  map2 (\ a b -> f64.abs (a-b) <= eps * (eps + f64.abs a) ) as bs
  |> reduce_comm (&&) true


---------------------------------------------------------------------
---                    Sum of Products                            ---
---------------------------------------------------------------------

-- Reduce with sum-of-products (SOP) operator: performance
-- ==
-- entry: primal_sop ppad_sop vjp2_sop vjp2_sop_inv 
-- compiled random input { [50000000]f64  [50000000]f64 f64 f64 }

-- compiled random input {[500000000]f64 [500000000]f64 f64 f64 }

def primal_red_sop [n] (xs: [n](f64,f64)) : (f64,f64) =
  reduce_comm sop sop_ne xs

entry primal_sop [n]
      (inp1 : [n]f64) (inp2 : [n]f64)
      (_adj1 : f64) (_adj2 : f64) : (f64, f64) =
  zip inp1 inp2 |> primal_red_sop

entry vjp2_sop [n] (inp1 : [n]f64) (inp2 : [n]f64)
                   (adj1 : f64) (adj2 : f64) 
                 : (f64, f64, [n]f64, [n]f64) =
  let ((r1,r2), inp_bar) =
      vjp2 primal_red_sop (zip inp1 inp2) (adj1, adj2)
  let (inp1_bar, inp2_bar) = unzip inp_bar
  in (r1, r2, inp1_bar, inp2_bar)

entry ppad_sop [n] (inp1 : [n]f64) (inp2 : [n]f64)
                   (adj1 : f64) (adj2 : f64)
                 : (f64, f64, [n]f64, [n]f64) =
  let ((r1,r2), inp_bar) =
    vjp2_red_ppad zero_tup plus_tup sop sop_ne (zip inp1 inp2) (adj1, adj2)
  let (inp1_bar, inp2_bar) = unzip inp_bar
  in  (r1, r2, inp1_bar, inp2_bar)

entry vjp2_sop_inv [n]
                   (inp1 : [n]f64) (inp2 : [n]f64)
                   (adj1 : f64) (adj2 : f64) 
                 : (f64, f64, [n]f64, [n]f64) =
  let ((r1,r2), inp_bar) =
    vjp2_red_inv sop sop sop_ne id id sop_inv (zip inp1 inp2) (adj1, adj2)
  let (inp1_bar, inp2_bar) = unzip inp_bar
  in  (r1, r2, inp1_bar, inp2_bar)

-- Reduce with SOP: validity
-- ==
-- entry: validate_sop
-- compiled random input {  [10000000]f64  [10000000]f64 f64 f64 } output {true}

entry validate_sop [n] (inp1 : [n]f64)
                    (inp2 : [n]f64)
                    (adj1 : f64)
                    (adj2 : f64) =
 let (_, ppad_bar_tup) =
    vjp2_red_ppad zero_tup plus_tup sop sop_ne (zip inp1 inp2) (adj1, adj2)
 let ppad_bar = unzip ppad_bar_tup
 let (_, orig_bar_tup) =
    vjp2 primal_red_sop (zip inp1 inp2) (adj1, adj2)
 let orig_bar = unzip orig_bar_tup
 let (_, optm_bar_tup) =
    vjp2_red_inv sop sop sop_ne id id sop_inv (zip inp1 inp2) (adj1, adj2)
 let optm_bar = unzip optm_bar_tup
 --
 let eps = 0.0000000000001f64
 in  equalEps eps ppad_bar.0 orig_bar.0 &&
     equalEps eps ppad_bar.1 orig_bar.1 &&
     equalEps eps orig_bar.0 optm_bar.0 &&
     equalEps eps orig_bar.1 optm_bar.1
     
---------------------------------------------------------------------
---                    Multiplication                             ---
---------------------------------------------------------------------

-- Reduce with sum-of-products (SOP) operator: performance
-- ==
-- entry: primal_mul ppad_mul vjp2_mul vjp2_mul_inv 
-- compiled random input { [50000000]f64 f64 }
-- compiled random input {[500000000]f64 f64 }

def primal_red_mul [n] (xs: [n]f64) : f64 =
  reduce_comm (*) 1.0 xs

entry primal_mul [n] (inp : [n]f64) (_adj1 : f64) : f64 =
  primal_red_mul inp

entry vjp2_mul [n] (inp : [n]f64) (adj : f64) : (f64, [n]f64) =
  vjp2 primal_red_mul inp adj

entry ppad_mul [n] (inp : [n]f64) (adj : f64) : (f64, [n]f64) =
  vjp2_red_ppad 0f64 (+) (*) 1.0f64 inp adj

entry vjp2_mul_inv [n] (inp : [n]f64) (adj : f64) : (f64, [n]f64) =
  vjp2_red_inv (*) mul_lft mul_lft_ne mul_cfwd mul_cbwd mul_inv inp adj

-- Reduce with SOP: validity
-- ==
-- entry: validate_mul
-- input { [2.0f64, 3.0f64, 0f64, 5.0f64, 4.0f64] 1.5f64 } output {true}
-- input { [2.0f64, 3.0f64, 0f64, 5.0f64, 4.0f64, 0.0f64] 1.5f64 } output {true}
-- compiled input@data/red-100K.in
-- output {true}

entry validate_mul [n] (inp : [n]f64) (adj : f64) : bool =
 --
 let (ppad_r, ppad_bar) = vjp2_red_ppad 0f64 (+) (*) 1.0f64 inp adj
 let (vjp2_r, vjp2_bar) = vjp2 primal_red_mul inp adj
 let (optm_r, optm_bar) = vjp2_red_inv (*) mul_lft mul_lft_ne mul_cfwd mul_cbwd mul_inv inp adj
 --
 let eps = 0.0000000000001f64
 in  f64.abs(ppad_r - vjp2_r) < eps * (eps + f64.abs vjp2_r) &&
     f64.abs(optm_r - vjp2_r) < eps * (eps + f64.abs vjp2_r) &&
     equalEps eps ppad_bar vjp2_bar &&
     equalEps eps vjp2_bar optm_bar
