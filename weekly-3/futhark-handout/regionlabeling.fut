-- Start with some utility definitions for handling directions and positions.

-- | A cardinal direction, with `#c` being current location ("centre").

type dir = #n | #w | #e | #s

-- | Position in a grid.
type pos = (i64, i64)

-- | A representative for an invalid position.
def no_pos : pos = (-1, -1)

-- | Less-than-or-equal comparison of positions. Requires you to pass in the
-- grid width.
def pos_lte (w: i64) ((ax, ay): pos) ((bx, by): pos) : bool =
  ax * w + ay <= bx * w + by

-- | Move along direction.
def move (d: dir) ((i, j): pos) =
  match d
  case #n -> (i - 1, j)
  case #w -> (i, j - 1)
  case #e -> (i, j + 1)
  case #s -> (i + 1, j)

-- | Turn a position into a flat index, given a grid width.
def flat_pos (w: i64) ((x, y): pos) : i64 = x * w + y

-- | Turn a flat index into a position, given a grid width.
def unflat_pos (w: i64) (i: i64) : pos = (i // w, i %% w)

-- | Is this position in bounds in some grid?
def in_bounds [h] [w] 'a (_: [h][w]a) ((i, j): pos) =
  i >= 0 && i < h && j >= 0 && j < w

-- | Get element at position in grid.
def get 'a ((i, j): pos) (g: [][]a) =
  g[i, j]

-- > :img ($loadimg "regions-hard.png")

def region_label_naive [h] [w] (img: [h][w]u32) : [h][w]i64 =
  let labels_flat = flatten (tabulate_2d h w \i j -> flat_pos w (i, j))
  let (_, res) =
    loop (prev_labels, current_labels) = (replicate (h * w) (-1), labels_flat)
    while prev_labels != current_labels do
      let new_labels =
        map (\fpos ->
               let pos = unflat_pos w fpos
               let color = get pos img
               let neighbours =
                 map (\d ->
                        let npos = move d pos
                        in if in_bounds img npos then (npos, get npos img) else (no_pos, 0))
                     [#n, #w, #e, #s]
               let same_color_labels = map (\(npos, c) -> if c == color && npos != no_pos then current_labels[flat_pos w npos] else -1i64) neighbours
               in reduce i64.max current_labels[fpos] same_color_labels)
            labels_flat
      in (current_labels, new_labels)
  in unflatten (res :> [h * w]i64)

-- | Could be improved. This is unlikely to produce something very legible.
def colourise_regions [h] [w] (labels: [h][w]i64) : [h][w]u32 =
  let f l = u32.i64 l
  in map (map f) labels

-- > :img (colourise_regions (region_label_naive ($loadimg "regions-hard.png")))

type edge = (pos, pos)

-- | Normalise an edge such that it goes from the lesser index to the greater.
def norm_edge w ((a, b): edge) : edge =
  if pos_lte w a b then (a, b) else (a, b)

-- | Create normalised edges linking all neighbouring pixels with the same
-- colour.
def mk_edges [h] [w] (img: [h][w]u32) : ?[k].[k]edge =
  let edges =
    tabulate_2d h w (\r c ->
                       let pos = (r, c)
                       let color = get pos img
                       let e_neighbour = move #e pos
                       let s_neighbour = move #s pos
                       let e_edge = if in_bounds img e_neighbour && color == (get e_neighbour img) then (pos, e_neighbour) else (no_pos, no_pos)
                       let s_edge = if in_bounds img s_neighbour && color == (get s_neighbour img) then (pos, s_neighbour) else (no_pos, no_pos)
                       in [e_edge, s_edge])
    |> flatten_3d
  in filter (\(a, b) -> a != no_pos && b != no_pos) edges

def region_label_smarter [h] [w] (img: [h][w]u32) =
  -- Step 1: compute edges.
  let edges = map (\(a, b) -> (flat_pos w a, flat_pos w b)) (mk_edges img)
  -- Step 2: Initialise DAG.
  let forest = flatten (tabulate_2d h w \i j -> flat_pos w (i, j))
  let (forest', _) =
    loop (forest, edges) while length edges > 0 do
      let possible_edges = filter (\(u, _) -> forest[u] == u) edges
      let sources = map (.0) possible_edges
      let targets = map (.1) possible_edges
      let new_forest = reduce_by_index forest i64.max (-1) sources targets
      let new_edges = trace(map (\(a, b) -> (new_forest[a], new_forest[b])) edges)
      let non_inserted_edges = filter (\(a, b) -> a != b) new_edges
      in (new_forest, non_inserted_edges)
  let (_, labels_flat) =
    loop (prev_forest, cur_forest) = (replicate (h * w) (-1), forest')
    while prev_forest != cur_forest do
      (cur_forest, map (\i -> cur_forest[cur_forest[i]]) cur_forest)
  in unflatten labels_flat

-- > :img (colourise_regions (region_label_smarter ($loadimg "regions-hard.png")))
