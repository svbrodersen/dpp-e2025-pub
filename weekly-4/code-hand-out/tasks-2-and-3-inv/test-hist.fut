import "gen-vjp2-comminv"
import "ops"

def plus_tup (a1: f64, a2: f64) (b1: f64, b2: f64) = (a1+b1, a2+b2)
def zero_tup = (0f64, 0f64)

def equalEps [n] (eps: f64) (as: [n]f64) (bs: [n]f64) : bool =
  map2 (\ a b -> f64.abs (a-b) <= eps * (eps + f64.abs a) ) as bs
  |> reduce_comm (&&) true

---------------------------------------------------------------------
---                    Sum of Products                            ---
---------------------------------------------------------------------

-- Reduce-by-Index with sum-of-products (SOP) operator: performance
-- ==
-- entry: primal_sop vjp2_sop vjp2_sop_inv
-- compiled input@data/histo-31-10M-tup.in
-- compiled input@data/histo-401-10M-tup.in
-- compiled input@data/histo-50K-10M-tup.in


def primal_hist_sop [m][n] (dst: [m](f64,f64)) (ks: [n]i64) (xs: [n](f64,f64)) : [m](f64,f64) =
  let hs = hist sop sop_ne m ks xs
  let rs = map2 sop dst hs
  in  rs

entry primal_sop [m][n]
      (dst1 : [m]f64) (dst2 : [m]f64) (ks: [n]i64)
      (inp1 : [n]f64) (inp2 : [n]f64)
      (_adj1 : [m]f64) (_adj2 : [m]f64) : ([m]f64, [m]f64) =
  zip inp1 inp2 |> primal_hist_sop (zip dst1 dst2) ks |> unzip

entry vjp2_sop [m][n]
      (dst1 : [m]f64) (dst2 : [m]f64) (ks: [n]i64)
      (inp1 : [n]f64) (inp2 : [n]f64)
      (adj1 : [m]f64) (adj2 : [m]f64)
    : ([m]f64, [m]f64, [m]f64, [m]f64, [n]f64, [n]f64) =
  let (rs, inp_bar) =
      vjp2 (primal_hist_sop (zip dst1 dst2) ks) (zip inp1 inp2) (zip adj1 adj2)
  
  let (inp1_bar, inp2_bar) = unzip inp_bar
  let (r1, r2) = unzip rs
  
  let dst_bar = vjp (\ dst -> primal_hist_sop dst ks (zip inp1 inp2)) (zip dst1 dst2) (zip adj1 adj2) |> unzip
  
  in (r1, r2, dst_bar.0, dst_bar.1, inp1_bar, inp2_bar)

entry vjp2_sop_inv [m][n]
      (dst1 : [m]f64) (dst2 : [m]f64) (ks: [n]i64)
      (inp1 : [n]f64) (inp2 : [n]f64)
      (adj1 : [m]f64) (adj2 : [m]f64)
    : ([m]f64, [m]f64, [m]f64, [m]f64, [n]f64, [n]f64) =
  let (rs, dst_bar, inp_bar) =
    vjp_redbyind_inv sop sop_lft sop_ne id id sop_inv
                    (zip dst1 dst2) ks (zip inp1 inp2)
                    (zip adj1 adj2)
  --  
  let (r1, r2) = unzip rs
  let (dst1_bar, dst2_bar) = unzip dst_bar
  let (inp1_bar, inp2_bar) = unzip inp_bar
  --
  in  (r1, r2, dst1_bar, dst2_bar, inp1_bar, inp2_bar)

-- Reduce-by-Index with SOP: validity
-- ==
-- entry: validate_sop
-- compiled input@data/histo-31-10M-tup.in
-- output {true}
-- compiled input@data/histo-401-10M-tup.in
-- output {true}
-- compiled input@data/histo-50K-10M-tup.in
-- output {true}

entry validate_sop [m][n]
      (dst1 : [m]f64) (dst2 : [m]f64) (ks: [n]i64)
      (inp1 : [n]f64) (inp2 : [n]f64)
      (adj1 : [m]f64) (adj2 : [m]f64) : bool =
  --
  let (vjp2_rs, vjp2_inp_bar) =
      vjp2 (primal_hist_sop (zip dst1 dst2) ks) (zip inp1 inp2) (zip adj1 adj2)
  
  let (vjp2_inp1_bar, vjp2_inp2_bar) = unzip vjp2_inp_bar
  let (vjp2_r1, vjp2_r2) = unzip vjp2_rs
  
  let (vjp2_dst1_bar, vjp2_dst2_bar) =
      vjp (\ dst -> primal_hist_sop dst ks (zip inp1 inp2)) (zip dst1 dst2) (zip adj1 adj2) |> unzip
 
  let (optm_rs, optm_dst_bar, optm_inp_bar) =
      vjp_redbyind_inv sop sop_lft sop_ne id id sop_inv
                    (zip dst1 dst2) ks (zip inp1 inp2)
                    (zip adj1 adj2)
  let (optm_r1, optm_r2) = unzip optm_rs
  let (optm_dst1_bar, optm_dst2_bar) = unzip optm_dst_bar
  let (optm_inp1_bar, optm_inp2_bar) = unzip optm_inp_bar
 --
 let eps = 0.0000000000001f64
 in  equalEps eps vjp2_r1 optm_r1 &&
     equalEps eps vjp2_r2 optm_r2 &&
     equalEps eps vjp2_dst1_bar optm_dst1_bar &&
     equalEps eps vjp2_dst2_bar optm_dst2_bar &&   
     equalEps eps vjp2_inp1_bar optm_inp1_bar &&
     equalEps eps vjp2_inp2_bar optm_inp2_bar

---------------------------------------------------------------------
---                    Multiplication                             ---
---------------------------------------------------------------------

-- Reduce-by-Index with multiplication operator: performance
-- ==
-- entry: primal_mul vjp2_mul vjp2_mul_inv
-- compiled input@data/histo-31-10M.in
-- compiled input@data/histo-401-10M.in
-- compiled input@data/histo-50K-10M.in


def primal_hist_mul [m][n] (dst: [m]f64) (ks: [n]i64) (xs: [n]f64) : [m]f64 =
  let hs = hist (*) 1.0 m ks xs
  let rs = map2 (*) dst hs
  in  rs

entry primal_mul [m][n]
        (dst : [m]f64) (ks: [n]i64) (inp : [n]f64)
        (_adj : [m]f64) : [m]f64 =
  primal_hist_mul dst ks inp

entry vjp2_mul [m][n]
      (dst : [m]f64) (ks: [n]i64) (inp : [n]f64)
      (adj : [m]f64) :  ([m]f64, [m]f64, [n]f64) =
  --
  let (rs, inp_bar) = vjp2 (primal_hist_mul dst ks) inp adj
  --
  let dst_bar = vjp (\dst' -> primal_hist_mul dst' ks inp) dst adj
  --
  in (rs, dst_bar, inp_bar)

entry vjp2_mul_inv [m][n]
      (dst : [m]f64) (ks: [n]i64) (inp : [n]f64)
      (adj : [m]f64) : ([m]f64, [m]f64, [n]f64) =
  --
  let (rs, dst_bar, inp_bar) =
    vjp_redbyind_inv (*) mul_lft mul_lft_ne
                     mul_cfwd mul_cbwd mul_inv
                     dst ks inp adj
  --  
  in (rs, dst_bar, inp_bar)

-- Reduce-by-Index with multiplication: validity
-- ==
-- entry: validate_mul
-- compiled input@data/histo-31-1M.in
-- output {true}
-- compiled input@data/histo-401-1M.in
-- output {true}
-- compiled input@data/histo-50K-1M.in
-- output {true}

entry validate_mul [m][n]
      (dst : [m]f64) (ks: [n]i64) (inp : [n]f64) (adj : [m]f64) : bool =
  --
  let (rs, inp_bar) = vjp2 (primal_hist_mul dst ks) inp adj  
  let dst_bar = vjp (\dst' -> primal_hist_mul dst' ks inp) dst adj
 
  let (inv_rs, inv_dst_bar, inv_inp_bar) =
    vjp_redbyind_inv (*) mul_lft mul_lft_ne
                     mul_cfwd mul_cbwd mul_inv
                     dst ks inp adj
 --
 let eps = 0.0000000000001f64
 in  equalEps eps rs inv_rs &&
     equalEps eps dst_bar inv_dst_bar &&
     equalEps eps inp_bar inv_inp_bar
