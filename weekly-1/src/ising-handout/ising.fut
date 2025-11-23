-- We represent a spin as a single byte.  In principle, we need only
-- two values (-1 or 1), but Futhark represents booleans a a full byte
-- entirely, so using an i8 instead takes no more space, and makes the
-- arithmetic simpler.
type spin = i8

import "lib/github.com/diku-dk/cpprandom/random"

-- Pick an RNG engine and define random distributions for specific types.
module rng_engine = minstd_rand
module rand_f32 = uniform_real_distribution f32 rng_engine
module rand_i8 = uniform_int_distribution i8 rng_engine

-- We can create an few RNG state with 'rng_engine.rng_from_seed [x]',
-- where 'x' is some seed.  We can split one RNG state into many with
-- 'rng_engine.split_rng'.
--
-- For an RNG state 'r', we can generate random integers that are
-- either 0 or 1 by calling 'rand_i8.rand (0i8, 1i8) r'.
--
-- For an RNG state 'r', we can generate random floats in the range
-- (0,1) by calling 'rand_f32.rand (0f32, 1f32) r'.
--
-- Remember to consult
-- https://futhark-lang.org/pkgs/github.com/diku-dk/cpprandom/latest/

def rand = rand_f32.rand (0f32, 1f32)

-- Create a new grid of a given size.  Also produce an identically
-- sized array of RNG states.
def random_grid (seed: i32) (h: i64) (w: i64)
              : ([h][w]rng_engine.rng, [h][w]spin) =
  let initial = rng_engine.rng_from_seed [seed]
  let states_flat = rng_engine.split_rng (h * w) initial
  let (new_states, spins) = map (\r -> rand_i8.rand (-1i8, 1i8) r) states_flat |> unzip
  in (unflatten new_states, unflatten spins)


-- Compute $\Delta_e$ for each spin in the grid, using wraparound at
-- the edges.
def deltas [h][w] (spins: [h][w]spin): [h][w]i8 =
   let rot_down = rotate 1 spins |> flatten
   let rot_up = rotate (-1) spins |> flatten
   let rot_left = map (rotate (-1)) spins |> flatten
   let rot_right = map (rotate (1)) spins |> flatten
   -- for spins[i, j] then l is rot_right[i, j]
   let res = map5 (\c u d l r -> 2 * c * (u + d + l + r)) (flatten spins) rot_down rot_up rot_right rot_left
   in unflatten res


-- The sum of all deltas of a grid.  The result is a measure of how
-- ordered the grid is.
def delta_sum [h][w] (spins: [h][w]spin): i32 =
  deltas spins |> flatten |> map (i32.i8) |> reduce (+) 0

-- Take one step in the Ising 2D simulation.
def step [h][w] (abs_temp: f32) (samplerate: f32)
                (rngs: [h][w]rng_engine.rng) (spins: [h][w]spin)
              : ([h][w]rng_engine.rng, [h][w]spin) =
  let delta_es = deltas spins |> flatten |> map (f32.i8)
  let (new_states, new_spins) = map3 (\r1 c delta_e ->
    let neg_delta_e = - delta_e
    let (r2, a) = rand_f32.rand (0f32, 1f32) r1 
    let (new_r, b) = rand_f32.rand (0f32, 1f32) r2
    let pow_e = f32.exp (neg_delta_e / abs_temp)
    in
      if (a < samplerate) && ((delta_e < neg_delta_e) || (b < pow_e)) then 
        (new_r, -c)
      else 
        (new_r, c)
  ) (flatten rngs) (flatten spins) delta_es |> unzip
  in (unflatten new_states, unflatten new_spins)

-- | Just for benchmarking.
def main (abs_temp: f32) (samplerate: f32)
         (h: i64) (w: i64) (n: i32): [h][w]spin =
  (loop (rngs, spins) = random_grid 1337 h w for _i < n do
     step abs_temp samplerate rngs spins).1

-- ==
-- entry: main
-- input { 0.5f32 0.1f32 10i64 10i64 2 } auto output

-- The following definitions are for the visualisation and need not be modified.

type~ state = {cells: [][](rng_engine.rng, spin)}

entry tui_init seed h w : state =
  let (rngs, spins) = random_grid seed h w
  in {cells=map (uncurry zip) (zip rngs spins)}

entry tui_render (s: state) = map (map (.1)) s.cells

entry tui_step (abs_temp: f32) (samplerate: f32) (s: state) : state =
  let rngs = (map (map (.0)) s.cells)
  let spins = map (map (.1)) s.cells
  let (rngs', spins') = step abs_temp samplerate rngs spins
  in {cells=map (uncurry zip) (zip rngs' spins')}
