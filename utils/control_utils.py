import numpy as np
from scipy.spatial import KDTree
import meshio
from sklearn.cluster import KMeans

def gen_control_points_uniformly(obj_path, tgf_out_path, weight_out_path, num_control_points=8):
    # Parse OBJ file
    mesh = meshio.read(obj_path)
    vertices = mesh.points
    vertices = np.array(vertices)
    
    # Create control points using k-means clustering
    kmeans = KMeans(n_clusters=num_control_points, random_state=0).fit(vertices)
    centroids = kmeans.cluster_centers_
    
    # For each centroid, find the nearest vertex in the original mesh
    control_points = np.zeros((num_control_points, 3))
    for i, centroid in enumerate(centroids):
        # Calculate distances from this centroid to all vertices
        distances = np.sum((vertices - centroid) ** 2, axis=1)
        # Find the index of the closest vertex
        closest_vertex_idx = np.argmin(distances)
        # Use the closest vertex as the control point
        control_points[i] = vertices[closest_vertex_idx]
    
    # print(f"Selected {num_control_points} control points using k-means clustering")
    
    # Write TGF file
    with open(tgf_out_path, 'w') as f:
        # Write nodes
        for i, p in enumerate(control_points):
            f.write(f"{i} {p[0]} {p[1]} {p[2]}\n")
        
        f.write("#\n")
        
        # # Generate edges (simplified - you might want a different connectivity)
        # for i in range(len(control_points)):
        #     for j in range(i+1, len(control_points)):
        #         f.write(f"{i} {j}\n")
    
    # Compute weights
    tree = KDTree(control_points)
    # Initialize weights matrix with zeros
    weights = np.zeros((len(vertices), num_control_points))
    
    for i, v in enumerate(vertices):
        # Find k nearest control points
        dists, indices = tree.query(v, k=4)
        # Calculate weights based on inverse distance
        w = 1.0 / (dists + 1e-10)
        w = w / np.sum(w)  # normalize
        
        # Fill the weights matrix
        for idx, weight in zip(indices, w):
            weights[i, idx] = weight
    
    # Save weights to a text file
    if weight_out_path.endswith('.npy'):
        np.save(weight_out_path, weights.astype(np.float32))
    else:
        with open(weight_out_path, 'w') as f:
            # First line: number of vertices and control points
            # f.write(f"{len(vertices)} {num_control_points}\n")
            
            # Write the full weight matrix
            for vertex_idx in range(len(vertices)):
                weight_str = ' '.join([f"{w:.6f}" for w in weights[vertex_idx]])
                f.write(f"{weight_str}\n")

def gen_weights_from_control_points(mesh_path, tgf_out_path, weight_out_path, control_points_idx, k=4):
    """
    Generate weights from control points using k-means clustering.

    Args:
        obj_path (str): Path to the OBJ file.
        tgf_out_path (str): Path to save the TGF file.
        weight_out_path (str): Path to save the weights file.
        control_points_idx (list): List of indices for control points in the OBJ file.
        k (int): Number of nearest neighbors to consider for weight calculation.
    """
    # Parse mesh file
    mesh = meshio.read(mesh_path)
    
    # Extract vertices and faces
    vertices = mesh.points
    vertices = np.array(vertices)
    num_control_points = len(control_points_idx)
    
    # Create control points using k-means clustering
    
    # # Apply k-means to vertices
    # kmeans = KMeans(n_clusters=num_control_points, random_state=0).fit(vertices)
    # centroids = kmeans.cluster_centers_
    
    # For each centroid, find the nearest vertex in the original mesh
    control_points = vertices[control_points_idx]
    
    # Write TGF file
    with open(tgf_out_path, 'w') as f:
        for i, p in enumerate(control_points):
            f.write(f"{i} {p[0]} {p[1]} {p[2]}\n")
        f.write("#\n")
    
    # Compute weights
    tree = KDTree(control_points)
    # Initialize weights matrix with zeros
    weights = np.zeros((len(vertices), num_control_points))
    
    for i, v in enumerate(vertices):
        # Find k nearest control points
        dists, indices = tree.query(v, k=k)
        # Calculate weights based on inverse distance
        w = (1.0 / (dists + 1e-10)) ** 1
        # w = np.exp(-dists * 10)  # Gaussian-like falloff
        w = w / np.sum(w)  # normalize
        
        # Fill the weights matrix
        if k == 1:
            weights[i, 0] = w
        else:
            for idx, weight in zip(indices, w):
                weights[i, idx] = weight
    
    # Save weights to a text file
    # np.savetxt(weight_out_path, weights, fmt='%.6f')
    if weight_out_path.endswith('.npy'):
        np.save(weight_out_path, weights.astype(np.float32))
    else:
        with open(weight_out_path, 'w') as f:
            # First line: number of vertices and control points
            # f.write(f"{len(vertices)} {num_control_points}\n")
            
            # Write the full weight matrix
            for vertex_idx in range(len(vertices)):
                weight_str = ' '.join([f"{w:.6f}" for w in weights[vertex_idx]])
                f.write(f"{weight_str}\n")

def gen_weights_with_influence_radius(obj_path, tgf_out_path, weight_out_path, control_points_idx, 
                                     influence_radius=0.3, falloff_factor=2.0, normalize_global=True):
    """
    Generate weights using influence radius and falloff factor for more localized control.

    Args:
        obj_path (str): Path to the OBJ file.
        tgf_out_path (str): Path to save the TGF file.
        weight_out_path (str): Path to save the weights file.
        control_points_idx (list): List of indices for control points in the OBJ file.
        influence_radius (float): Maximum distance for control point influence (relative to bounding box).
        falloff_factor (float): Controls how quickly influence falls off with distance (higher = more localized).
        normalize_global (bool): Whether to normalize weights globally to ensure they sum to 1.
    """
    # Parse OBJ file
    mesh = meshio.read(obj_path)
    vertices = mesh.points
    vertices = np.array(vertices)
    num_control_points = len(control_points_idx)
    
    # Get control points from specified indices
    control_points = vertices[control_points_idx]
    
    # Calculate bounding box to normalize influence radius
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    absolute_radius = influence_radius * bbox_size
    
    print(f"Bounding box size: {bbox_size:.3f}")
    print(f"Absolute influence radius: {absolute_radius:.3f}")
    print(f"Falloff factor: {falloff_factor}")
    
    # Write TGF file
    with open(tgf_out_path, 'w') as f:
        for i, p in enumerate(control_points):
            f.write(f"{i} {p[0]} {p[1]} {p[2]}\n")
        f.write("#\n")
    
    # Initialize weights matrix with zeros
    weights = np.zeros((len(vertices), num_control_points))
    
    # Calculate weights for each vertex
    for i, vertex in enumerate(vertices):
        vertex_weights = np.zeros(num_control_points)
        
        for j, control_point in enumerate(control_points):
            # Calculate distance from vertex to control point
            dist = np.linalg.norm(vertex - control_point)
            
            if dist <= absolute_radius:
                # Calculate weight based on distance with falloff
                # Use normalized distance (0 to 1 within influence radius)
                normalized_dist = dist / absolute_radius
                
                # Apply falloff: weight = (1 - normalized_dist)^falloff_factor
                weight = (1.0 - normalized_dist) ** falloff_factor
                vertex_weights[j] = weight
            else:
                # Outside influence radius, no weight
                vertex_weights[j] = 0.0
        
        # Normalize weights for this vertex (ensure they sum to 1)
        total_weight = np.sum(vertex_weights)
        if total_weight > 0:
            vertex_weights = vertex_weights / total_weight
        else:
            # If no control points are within influence radius, assign equal weights
            vertex_weights = np.ones(num_control_points) / num_control_points
        
        weights[i] = vertex_weights
    
    # # Optional: Global normalization to ensure all weights are well-distributed
    # if normalize_global:
    #     # Calculate average influence per control point
    #     avg_influence = np.mean(weights, axis=0)
    #     print(f"Average influence per control point: {avg_influence}")
        
    #     # Identify control points with very low influence
    #     min_influence_threshold = 0.01
    #     low_influence_points = avg_influence < min_influence_threshold
        
    #     if np.any(low_influence_points):
    #         print(f"Warning: Control points {np.where(low_influence_points)[0]} have very low influence")
    
    # Save weights
    if weight_out_path.endswith('.npy'):
        np.save(weight_out_path, weights.astype(np.float32))
    else:
        with open(weight_out_path, 'w') as f:
            for vertex_idx in range(len(vertices)):
                weight_str = ' '.join([f"{w:.6f}" for w in weights[vertex_idx]])
                f.write(f"{weight_str}\n")
    
    # Print statistics
    print(f"Weight statistics:")
    print(f"  Min weight: {np.min(weights):.6f}")
    print(f"  Max weight: {np.max(weights):.6f}")
    print(f"  Vertices with non-zero weights: {np.sum(np.any(weights > 0, axis=1))}/{len(vertices)}")

def gen_weights_adaptive_radius(obj_path, tgf_out_path, weight_out_path, control_points_idx, 
                               base_radius=0.2, overlap_factor=1.5, falloff_factor=2.0):
    """
    Generate weights with adaptive influence radius based on control point density.

    Args:
        obj_path (str): Path to the OBJ file.
        tgf_out_path (str): Path to save the TGF file.
        weight_out_path (str): Path to save the weights file.
        control_points_idx (list): List of indices for control points in the OBJ file.
        base_radius (float): Base influence radius (relative to bounding box).
        overlap_factor (float): Factor to ensure neighboring control points overlap.
        falloff_factor (float): Controls how quickly influence falls off with distance.
    """
    # Parse OBJ file
    mesh = meshio.read(obj_path)
    vertices = mesh.points
    vertices = np.array(vertices)
    num_control_points = len(control_points_idx)
    
    # Get control points from specified indices
    control_points = vertices[control_points_idx]
    
    # Calculate bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    # Build KD-tree for control points to find nearest neighbors
    from scipy.spatial import KDTree
    control_tree = KDTree(control_points)
    
    # Calculate adaptive radius for each control point
    adaptive_radii = np.zeros(num_control_points)
    for i, cp in enumerate(control_points):
        # Find distance to nearest neighbor control point
        dists, indices = control_tree.query(cp, k=2)  # k=2 to exclude self
        nearest_neighbor_dist = dists[1] if len(dists) > 1 else bbox_size * base_radius
        
        # Set radius to ensure overlap with nearest neighbor
        adaptive_radii[i] = min(nearest_neighbor_dist * overlap_factor, bbox_size * base_radius)
    
    print(f"Adaptive radii: {adaptive_radii}")
    
    # Write TGF file
    with open(tgf_out_path, 'w') as f:
        for i, p in enumerate(control_points):
            f.write(f"{i} {p[0]} {p[1]} {p[2]}\n")
        f.write("#\n")
    
    # Initialize weights matrix
    weights = np.zeros((len(vertices), num_control_points))
    
    # Calculate weights for each vertex
    for i, vertex in enumerate(vertices):
        vertex_weights = np.zeros(num_control_points)
        
        for j, control_point in enumerate(control_points):
            # Calculate distance from vertex to control point
            dist = np.linalg.norm(vertex - control_point)
            radius = adaptive_radii[j]
            
            if dist <= radius:
                # Calculate weight with adaptive radius
                normalized_dist = dist / radius
                weight = (1.0 - normalized_dist) ** falloff_factor
                vertex_weights[j] = weight
        
        # Normalize weights
        total_weight = np.sum(vertex_weights)
        if total_weight > 0:
            vertex_weights = vertex_weights / total_weight
        else:
            # Fallback: use nearest control point
            dists_to_all = [np.linalg.norm(vertex - cp) for cp in control_points]
            nearest_idx = np.argmin(dists_to_all)
            vertex_weights[nearest_idx] = 1.0
        
        weights[i] = vertex_weights
    
    # Save weights
    if weight_out_path.endswith('.npy'):
        np.save(weight_out_path, weights.astype(np.float32))
    else:
        with open(weight_out_path, 'w') as f:
            for vertex_idx in range(len(vertices)):
                weight_str = ' '.join([f"{w:.6f}" for w in weights[vertex_idx]])
                f.write(f"{weight_str}\n")

def process_sphere():
    obj_name = "sphere"
    obj_path = f"assets/{obj_name}/{obj_name}.obj"
    tgf_out_path = f"assets/{obj_name}/control_points.tgf"
    weight_out_path = f"assets/{obj_name}/weights.npy"
    control_idx = [2057, 2751, 1017, 2216, 3102, 3908, 3525, 500, 2516, 1508, 2544, 1534, 3554, 590]
    # num_control_points = len(control_idx)
    # gen_control_points_uniform(obj_path, tgf_out_path, weight_out_path, num_control_points)
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx)

def process_simple_sphere():
    obj_name = "sphere_simple"
    obj_path = f"assets/{obj_name}/{obj_name}.obj"
    tgf_out_path = f"assets/{obj_name}/control_points.tgf"
    weight_out_path = f"assets/{obj_name}/weights.npy"
    control_idx = [0, 11, 1, 5, 18, 20, 32, 36, 7, 8]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{obj_name}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_simple_sphere_with_radius():
    """Example usage of the new influence radius method"""
    obj_name = "sphere_simple3"
    obj_path = f"assets/{obj_name}/{obj_name}.obj"
    tgf_out_path = f"assets/{obj_name}/control_points.tgf"
    weight_out_path = f"assets/{obj_name}/weights.npy"
    control_idx = [0, 11, 1, 5, 18, 20, 32, 36, 7, 8]
    
    # Method 1: Fixed influence radius
    gen_weights_with_influence_radius(
        obj_path, tgf_out_path, weight_out_path, control_idx,
        influence_radius=0.5,
        falloff_factor=1,
        normalize_global=True
    )
    
    # # Method 2: Adaptive radius
    # weight_out_path_adaptive = f"assets/{obj_name}/weights_adaptive.npy"
    # gen_weights_adaptive_radius(
    #     obj_path, tgf_out_path, weight_out_path_adaptive, control_idx,
    #     base_radius=0.3,
    #     overlap_factor=1.2,
    #     falloff_factor=2.0
    # )
    weight_out_path_txt = f"assets/{obj_name}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_prostate():
    obj_name = "prostate"
    obj_path = f"assets/{obj_name}/{obj_name}.obj"
    tgf_out_path = f"assets/{obj_name}/control_points.tgf"
    weight_out_path = f"assets/{obj_name}/weights.npy"
    # control_idx = [14, 18, 190, 354, 441, 554, 557, 819, 823, 902, 926, 996, 1265, 1369, 1478, 1530, 1606, 1733, 1740, 1747, 1812, 1851, 1902, 2011, 2012, 2084, 2153, 2245, 2294, 2322, 2449, 2518, 2590, 2604, 2606, 2685, 2764, 2944, 2981, 3008, 3125, 3253, 3255, 3293, 3305, 3390, 3457, 3460, 3521, 3587]
    control_idx = [54, 228, 603, 653, 701, 822, 1011, 1474, 1636, 1711, 1731, 1857, 1963, 2448, 2491, 2597, 2633, 3236, 3303]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{obj_name}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_simple_sphere_tet():
    obj_name = "simple_sphere"
    mesh_path = f"assets/{obj_name}/{obj_name}.mesh"
    tgf_out_path = f"assets/{obj_name}/control_points.tgf"
    weight_out_path = f"assets/{obj_name}/weights.npy"
    control_idx = [0, 11, 1, 5, 18, 20, 32, 36, 7, 8]
    gen_weights_from_control_points(mesh_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{obj_name}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_prostate_tet():
    obj_name = "prostate_tet"
    obj_path = f"assets/{obj_name}/{obj_name}.mesh"
    tgf_out_path = f"assets/{obj_name}/control_points.tgf"
    weight_out_path = f"assets/{obj_name}/weights.npy"
    control_idx = [54, 228, 603, 653, 701, 822, 1011, 1474, 1636, 1711, 1731, 1857, 1963, 2448, 2491, 2597, 2633, 3236, 3303]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{obj_name}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_prostate_250626():
    dirname = "prostate_250626"
    obj_name = "prostate"
    obj_path = f"assets/{dirname}/{obj_name}.obj"
    tgf_out_path = f"assets/{dirname}/control_points.tgf"
    weight_out_path = f"assets/{dirname}/weights.npy"
    control_idx = [34, 105, 122, 357, 608, 780, 790, 1207, 1233, 1559, 1705, 1797, 1863, 2000, 2115, 2155, 2420, 2590, 2885, 2959, 3024, 3030, 3127, 3289, 3298, 3476, 3688, 
                   1399,1550,1558,1716,1728,1833,2147,2164,2323,1471,1474, 1606]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{dirname}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_prostate_tet_250626():
    dirname = "prostate_tet_250626"
    obj_name = "prostate_tet"
    obj_path = f"assets/{dirname}/{obj_name}.mesh"
    tgf_out_path = f"assets/{dirname}/control_points.tgf"
    weight_out_path = f"assets/{dirname}/weights.npy"
    control_idx = [34, 105, 122, 357, 608, 780, 790, 1207, 1233, 1559, 1705, 1797, 1863, 2000, 2115, 2155, 2420, 2590, 2885, 2959, 3024, 3030, 3127, 3289, 3298, 3476, 3688, 
                   1399,1550,1558,1716,1728,1833,2147,2164,2323,1471,1474, 1606]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{dirname}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_prostate_tet():
    dirname = "prostate_tet"
    obj_name = "prostate_tet"
    obj_path = f"assets/{dirname}/{obj_name}.mesh"
    tgf_out_path = f"assets/{dirname}/control_points.tgf"
    weight_out_path = f"assets/{dirname}/weights.npy"
    control_idx = [3671, 3473, 3421, 1121, 2898, 364, 1726]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{dirname}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

def process_prostate_260106():
    dirname = "prostate_260106"
    obj_name = "Prostate"
    obj_path = f"assets/{dirname}/{obj_name}.mesh"
    tgf_out_path = f"assets/{dirname}/control_points.tgf"
    weight_out_path = f"assets/{dirname}/weights.npy"
    control_idx = [1719, 924, 937, 1940, 1962, 1983, 1960, 7, 247, 570, 258]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx, k=4)
    weight_out_path_txt = f"assets/{dirname}/weights.txt"
    weights = np.load(weight_out_path)
    np.savetxt(weight_out_path_txt, weights, fmt='%.6f')

if __name__ == "__main__":
    process_prostate_260106()
