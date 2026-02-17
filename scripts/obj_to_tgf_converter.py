import numpy as np
from scipy.spatial import KDTree
import meshio
from sklearn.cluster import KMeans

def gen_control_points_uniform(obj_path, tgf_out_path, weight_out_path, num_control_points=8):
    # Parse OBJ file
    # Read the mesh using meshio
    mesh = meshio.read(obj_path)
    
    # Extract vertices and faces
    vertices = mesh.points
    
    # # Most meshio files use "triangle" cells for faces
    # if "triangle" in mesh.cells_dict:
    #     faces = mesh.cells_dict["triangle"]
    # else:
    #     # Fall back to the first cell type if triangles are not found
    #     cell_type = list(mesh.cells_dict.keys())[0] if mesh.cells_dict else None
    #     faces = mesh.cells_dict[cell_type] if cell_type else []
    
    vertices = np.array(vertices)
    
    # Create control points using k-means clustering
    
    # Apply k-means to vertices
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
    with open(weight_out_path, 'w') as f:
        # First line: number of vertices and control points
        # f.write(f"{len(vertices)} {num_control_points}\n")
        
        # Write the full weight matrix
        for vertex_idx in range(len(vertices)):
            weight_str = ' '.join([f"{w:.6f}" for w in weights[vertex_idx]])
            f.write(f"{weight_str}\n")

def gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_points_idx):
    # Parse OBJ file
    # Read the mesh using meshio
    mesh = meshio.read(obj_path)
    
    # Extract vertices and faces
    vertices = mesh.points
    vertices = np.array(vertices)
    
    # Create control points using k-means clustering
    
    # Apply k-means to vertices
    kmeans = KMeans(n_clusters=num_control_points, random_state=0).fit(vertices)
    centroids = kmeans.cluster_centers_
    
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
        dists, indices = tree.query(v, k=4)
        # Calculate weights based on inverse distance
        w = 1.0 / (dists + 1e-10)
        w = w / np.sum(w)  # normalize
        
        # Fill the weights matrix
        for idx, weight in zip(indices, w):
            weights[i, idx] = weight
    
    # Save weights to a text file
    np.savetxt(weight_out_path, weights, fmt='%.6f')

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) < 3:
    #     print("Usage: python obj_to_tgf_converter.py input.obj output.tgf output.weights")
    # else:
    #     obj_to_tgf(sys.argv[1], sys.argv[2], sys.argv[3])
    obj_path = 'assets/sphere/sphere.obj'
    tgf_out_path = 'assets/sphere/sphere.tgf'
    weight_out_path = 'assets/sphere/sphere_w.txt'
    num_control_points = 16
    # gen_control_points_uniform(obj_path, tgf_out_path, weight_out_path, num_control_points)
    control_idx = [2057, 2751, 1017, 2216, 3102, 3908, 3525, 500, 2516, 1508, 2544, 1534, 3554, 590]
    # control_idx = [2057, 2751, 1017, 4046, 4047, 4048, 3525, 500, 2516, 1508, 2544, 1534, 3554, 590]
    gen_weights_from_control_points(obj_path, tgf_out_path, weight_out_path, control_idx)
