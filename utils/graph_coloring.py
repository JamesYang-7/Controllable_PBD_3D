import numpy as np
from dataclasses import dataclass


@dataclass
class ColoringResult:
  """Result of a graph coloring operation.

  Attributes:
    ordered_indices: Element indices sorted by color group.
    color_offsets: color_offsets[i]:color_offsets[i+1] gives the index range
                   in ordered_indices for color group i.
    num_colors: Total number of colors used.
  """
  ordered_indices: np.ndarray
  color_offsets: list
  num_colors: int


# ---------------------------------------------------------------------------
# Adjacency builders
# ---------------------------------------------------------------------------

def build_element_adjacency(element_indices, n_elements, verts_per_element, n_vertices):
  """Build adjacency for elements that conflict when sharing a vertex.

  Args:
    element_indices: Flat array of vertex indices, length = n_elements * verts_per_element.
    n_elements: Number of elements (edges, tets, etc.).
    verts_per_element: Vertices per element (2 for edges, 4 for tets, etc.).
    n_vertices: Total number of vertices.

  Returns:
    List of sets, where adj[i] contains indices of elements adjacent to element i.
  """
  vert_to_elems = [[] for _ in range(n_vertices)]
  for e in range(n_elements):
    for k in range(verts_per_element):
      v = element_indices[e * verts_per_element + k]
      vert_to_elems[v].append(e)

  adj = [set() for _ in range(n_elements)]
  for e in range(n_elements):
    for k in range(verts_per_element):
      v = element_indices[e * verts_per_element + k]
      for neighbor in vert_to_elems[v]:
        if neighbor != e:
          adj[e].add(neighbor)
  return adj


def build_vertex_adjacency(element_indices, n_elements, verts_per_element, n_vertices):
  """Build adjacency for vertices that conflict when sharing an element.

  Args:
    element_indices: Flat array of vertex indices, length = n_elements * verts_per_element.
    n_elements: Number of elements.
    verts_per_element: Vertices per element.
    n_vertices: Total number of vertices.

  Returns:
    List of sets, where adj[i] contains indices of vertices adjacent to vertex i.
  """
  adj = [set() for _ in range(n_vertices)]
  for e in range(n_elements):
    verts = [element_indices[e * verts_per_element + k] for k in range(verts_per_element)]
    for i in range(verts_per_element):
      for j in range(i + 1, verts_per_element):
        adj[verts[i]].add(verts[j])
        adj[verts[j]].add(verts[i])
  return adj


# ---------------------------------------------------------------------------
# Coloring algorithms
# ---------------------------------------------------------------------------

def greedy_coloring(adjacency, n_nodes):
  """Greedy graph coloring.

  Assigns colors in node order, always picking the smallest available color.

  Args:
    adjacency: List of sets (or iterables), where adjacency[i] contains
               neighbors of node i.
    n_nodes: Number of nodes to color.

  Returns:
    ColoringResult with elements ordered by color group.
  """
  colors = [-1] * n_nodes
  for node in range(n_nodes):
    neighbor_colors = set()
    for neighbor in adjacency[node]:
      if colors[neighbor] != -1:
        neighbor_colors.add(colors[neighbor])
    c = 0
    while c in neighbor_colors:
      c += 1
    colors[node] = c

  num_colors = max(colors) + 1 if n_nodes > 0 else 0
  return _pack_result(colors, n_nodes, num_colors)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def color_edges(edge_indices, n_edges, n_vertices, algorithm='greedy'):
  """Color edges so no two edges sharing a vertex get the same color.

  Args:
    edge_indices: Flat array [v0,v1, v0,v1, ...], length = n_edges * 2.
    n_edges: Number of edges.
    n_vertices: Total number of vertices.
    algorithm: Coloring algorithm name (currently: 'greedy').

  Returns:
    ColoringResult.
  """
  adj = build_element_adjacency(edge_indices, n_edges, 2, n_vertices)
  return _dispatch(adj, n_edges, algorithm)


def color_elements(element_indices, n_elements, verts_per_element, n_vertices,
                   algorithm='greedy'):
  """Color elements so no two elements sharing a vertex get the same color.

  Args:
    element_indices: Flat array of vertex indices.
    n_elements: Number of elements.
    verts_per_element: Vertices per element (e.g. 4 for tets).
    n_vertices: Total number of vertices.
    algorithm: Coloring algorithm name.

  Returns:
    ColoringResult.
  """
  adj = build_element_adjacency(element_indices, n_elements, verts_per_element, n_vertices)
  return _dispatch(adj, n_elements, algorithm)


def color_vertices(element_indices, n_elements, verts_per_element, n_vertices,
                   algorithm='greedy'):
  """Color vertices so no two vertices sharing an element get the same color.

  Args:
    element_indices: Flat array of vertex indices.
    n_elements: Number of elements.
    verts_per_element: Vertices per element.
    n_vertices: Total number of vertices.
    algorithm: Coloring algorithm name.

  Returns:
    ColoringResult.
  """
  adj = build_vertex_adjacency(element_indices, n_elements, verts_per_element, n_vertices)
  return _dispatch(adj, n_vertices, algorithm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ALGORITHMS = {
    'greedy': greedy_coloring,
}


def _dispatch(adjacency, n_nodes, algorithm):
  if algorithm not in _ALGORITHMS:
    raise ValueError(f"Unknown coloring algorithm: '{algorithm}'. "
                     f"Available: {list(_ALGORITHMS.keys())}")
  return _ALGORITHMS[algorithm](adjacency, n_nodes)


def _pack_result(colors, n_nodes, num_colors):
  """Pack per-node colors into ordered indices + offsets."""
  groups = [[] for _ in range(num_colors)]
  for node, c in enumerate(colors):
    groups[c].append(node)

  ordered = []
  offsets = [0]
  for g in groups:
    ordered.extend(g)
    offsets.append(len(ordered))

  return ColoringResult(
      ordered_indices=np.array(ordered, dtype=np.int32),
      color_offsets=offsets,
      num_colors=num_colors,
  )
