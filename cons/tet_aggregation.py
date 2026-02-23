"""Tet-specialized multigrid aggregation using face-adjacency.

Produces prolongation operators (P matrices) in the same format as PyAMG,
so they can be used by both the CPU (mgpbd.py) and GPU (mgpbd_gpu.py) pipelines.
"""

import numpy as np
import scipy.sparse as sp


def build_tet_face_adjacency(t_i_np, n_tets):
  """Build face-adjacency graph: two tets sharing a triangular face are neighbors.

  A tet has 4 faces, each defined by 3 of its 4 vertices.  Two tets sharing
  a face are strongly coupled (full face contact), vs. vertex- or edge-sharing
  in generic AMG which represents weaker coupling.

  Returns:
      dict mapping tet_id -> set of face-adjacent tet_ids.
  """
  # Map each sorted face tuple -> list of tet ids containing that face
  face_to_tets = {}
  for t in range(n_tets):
    verts = [t_i_np[t * 4 + l] for l in range(4)]
    # 4 faces: each obtained by dropping one vertex
    for skip in range(4):
      face = tuple(sorted(verts[j] for j in range(4) if j != skip))
      if face not in face_to_tets:
        face_to_tets[face] = []
      face_to_tets[face].append(t)

  adj = {t: set() for t in range(n_tets)}
  for face, tets in face_to_tets.items():
    if len(tets) == 2:
      adj[tets[0]].add(tets[1])
      adj[tets[1]].add(tets[0])
  return adj


def tet_geometry_aggregation(t_i_np, v_p_np, n_tets, max_agg_size=8):
  """Geometry-aware greedy aggregation exploiting face adjacency.

  Strategy:
    1. Build face-adjacency graph.
    2. Greedy matching: pair face-adjacent tets starting from unmatched ones.
    3. Form aggregates from matched pairs.
    4. Remaining unmatched tets join the nearest aggregate (by centroid).

  Args:
      t_i_np: Flat tet index array (length 4*n_tets).
      v_p_np: Vertex positions (n_verts, 3).
      n_tets: Number of tetrahedra.
      max_agg_size: Maximum tets per aggregate (soft limit for greedy).

  Returns:
      agg_ids: np.ndarray of shape (n_tets,), aggregate assignment per tet.
      n_agg: Number of aggregates.
  """
  adj = build_tet_face_adjacency(t_i_np, n_tets)

  # Compute tet centroids for tie-breaking
  centroids = np.zeros((n_tets, 3), dtype=np.float64)
  for t in range(n_tets):
    c = np.zeros(3, dtype=np.float64)
    for l in range(4):
      c += v_p_np[t_i_np[t * 4 + l]]
    centroids[t] = c / 4.0

  agg_ids = -np.ones(n_tets, dtype=np.int32)
  agg_sizes = []  # size of each aggregate
  n_agg = 0

  # Pass 1: greedy face-adjacent matching
  visited = np.zeros(n_tets, dtype=bool)
  # Process tets sorted by ascending face-adjacency count (match sparse regions first)
  order = sorted(range(n_tets), key=lambda t: len(adj[t]))

  for seed in order:
    if visited[seed]:
      continue
    # Start new aggregate with seed
    agg_ids[seed] = n_agg
    visited[seed] = True
    current_size = 1

    # Greedily add face-adjacent unvisited neighbors
    candidates = sorted(adj[seed] - set(np.where(visited)[0]),
                        key=lambda t: len(adj[t]))
    for nb in candidates:
      if visited[nb]:
        continue
      if current_size >= max_agg_size:
        break
      agg_ids[nb] = n_agg
      visited[nb] = True
      current_size += 1
      # Also try to add neighbors-of-neighbor for larger aggregates
      if current_size < max_agg_size:
        for nb2 in adj[nb]:
          if not visited[nb2] and current_size < max_agg_size:
            agg_ids[nb2] = n_agg
            visited[nb2] = True
            current_size += 1

    agg_sizes.append(current_size)
    n_agg += 1

  # Pass 2: assign any remaining unmatched tets to nearest aggregate
  unmatched = np.where(agg_ids < 0)[0]
  if len(unmatched) > 0:
    # Compute aggregate centroids
    agg_centroids = np.zeros((n_agg, 3), dtype=np.float64)
    agg_counts = np.zeros(n_agg, dtype=np.float64)
    for t in range(n_tets):
      if agg_ids[t] >= 0:
        agg_centroids[agg_ids[t]] += centroids[t]
        agg_counts[agg_ids[t]] += 1.0
    for a in range(n_agg):
      if agg_counts[a] > 0:
        agg_centroids[a] /= agg_counts[a]

    for t in unmatched:
      dists = np.linalg.norm(agg_centroids - centroids[t], axis=1)
      agg_ids[t] = int(np.argmin(dists))

  assert np.all(agg_ids >= 0), "All tets must be assigned an aggregate"
  return agg_ids, n_agg


def build_prolongation_from_aggregates(agg_ids, n_tets, n_agg):
  """Build piecewise-constant prolongation matrix P from aggregate assignments.

  P is n_tets x n_agg, with P[i, agg_id[i]] = 1.

  Returns:
      P: scipy.sparse.csr_matrix
  """
  rows = np.arange(n_tets, dtype=np.int32)
  cols = agg_ids.astype(np.int32)
  data = np.ones(n_tets, dtype=np.float64)
  P = sp.csr_matrix((data, (rows, cols)), shape=(n_tets, n_agg))
  return P


def build_tet_hierarchy(t_i_np, v_p_np, n_tets, max_coarse=400, max_levels=4):
  """Build complete multigrid hierarchy using tet-specialized aggregation.

  Returns:
      Ps: list of scipy.sparse.csr_matrix prolongation operators,
          same format as PyAMG's ml.levels[i].P.
      sizes: list of level sizes for diagnostics.
  """
  Ps = []
  sizes = [n_tets]
  current_n = n_tets

  # For coarse levels, we create a "virtual" tet structure from aggregates.
  # The first level uses real tet geometry; subsequent levels use the
  # aggregate centroids as virtual vertex positions and aggregate adjacency.
  current_t_i = t_i_np
  current_v_p = v_p_np
  current_ntets = n_tets

  for level in range(max_levels):
    if current_ntets <= max_coarse:
      break

    # Determine aggregate size based on level (more aggressive on finer levels)
    max_agg = min(8, max(2, current_ntets // max(max_coarse, 1)))
    max_agg = max(max_agg, 2)

    agg_ids, n_agg = tet_geometry_aggregation(
        current_t_i, current_v_p, current_ntets, max_agg_size=max_agg)
    P = build_prolongation_from_aggregates(agg_ids, current_ntets, n_agg)
    Ps.append(P)
    sizes.append(n_agg)

    if n_agg >= current_ntets:
      # No coarsening progress — stop
      Ps.pop()
      sizes.pop()
      break

    # Build coarse-level virtual geometry for next level
    # Virtual vertices = aggregate centroids
    agg_centroids = np.zeros((n_agg, 3), dtype=np.float64)
    agg_counts = np.zeros(n_agg, dtype=np.float64)
    for t in range(current_ntets):
      c = np.zeros(3, dtype=np.float64)
      for l in range(4):
        c += current_v_p[current_t_i[t * 4 + l]]
      c /= 4.0
      agg_centroids[agg_ids[t]] += c
      agg_counts[agg_ids[t]] += 1.0
    for a in range(n_agg):
      if agg_counts[a] > 0:
        agg_centroids[a] /= agg_counts[a]

    # Virtual tets: each aggregate becomes a "tet" with 4 dummy vertices
    # (just itself repeated — the adjacency is what matters)
    virtual_t_i = np.zeros(n_agg * 4, dtype=np.int32)
    for a in range(n_agg):
      for l in range(4):
        virtual_t_i[a * 4 + l] = a

    current_t_i = virtual_t_i
    current_v_p = agg_centroids
    current_ntets = n_agg

  print(f"Tet hierarchy: {len(sizes)} levels, sizes={sizes}")
  return Ps, sizes
