-------------------------------
--- sum of product operator ---
-------------------------------

def sop_ne = (0f64, 0f64)

def sop_cfwd = id

def sop_cbwd = id

def sop (p1: f64, s1: f64) (p2: f64, s2: f64) =
  (p1 + p2 + s1*s2, s1+s2)

def sop_lft = sop

def sop_inv (acc_p: f64, acc_s: f64) (p1: f64, s1: f64) =
  let s2 = acc_s - s1
  let p2 = acc_p - p1 - s1*s2
  in  (p2, s2)

-------------------------------
--- multiplication operator ---
-------------------------------
def mul_lft_ne : (i64, f64) = (0, 1)

def mul_cfwd (a: f64) : (i64, f64) =
    if a == 0 then (1, 1.0) else (0, a)

def mul_cbwd (nz : i64, p : f64) : f64 =
    if nz == 0 then p else 0

def mul_lft (nz_1: i64, p_1: f64) (nz_2: i64, p_2: f64) : (i64, f64) =
    (nz_1 + nz_2, p_1 * p_2)

def mul_inv (nz: i64, p: f64) (a: f64) : f64 = 
  if nz == 1 && a == 0 then p
  else if nz == 0 then p / a  else 0



