import numpy as np
import trimesh
import os

def parse_obj(file_path):
    vertices = []
    face_normals = []
    faces = []
    using_vertex_normal = True

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('vn '):
                face_normals.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('f '):
                face = line.strip().split()[1:4]
                faces.append([int(face[i].split('//')[0]) - 1 for i in range(3)])
                vertex_id_equals_normal_id = [int(face[i].split('//')[0]) == int(face[i].split('//')[1]) for i in range(3)]
                if not all(vertex_id_equals_normal_id):
                    using_vertex_normal = False

    print(f"Parsed file {file_path}: {len(vertices)} vertices, {len(face_normals)} face normals, {len(faces)} faces")
    return np.array(vertices), np.array(face_normals), faces, using_vertex_normal

def calculate_vertex_normals(vertices, faces):
    vertex_normals = np.zeros(vertices.shape, dtype=np.float32)
    for face in faces:
        face_vertices = vertices[face]
        normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
        normal = normal / np.linalg.norm(normal)
        for vertex in face:
            vertex_normals[vertex] += normal
    
    vertex_normals = np.array([vn / np.linalg.norm(vn) for vn in vertex_normals])
    return vertex_normals

def replace_face_normals_with_vertex_normals(obj_path, new_vertex_normals, output_path):
    with open(obj_path, 'r') as file:
        lines = file.readlines()

    with open(output_path, 'w') as file:
        for line in lines:
            if line.startswith('vn '):
                continue  # Skip existing face normals
            elif line.startswith('f '):
                face_elements = line.strip().split()
                new_face_elements = []
                for element in face_elements[1:]:
                    vertex_index = int(element.split('//')[0]) - 1
                    vn_index = vertex_index + 1
                    new_face_elements.append(f"{element.split('//')[0]}//{vn_index}")
                file.write(f"f {' '.join(new_face_elements)}\n")
            else:
                file.write(line)
        
        for vn in new_vertex_normals:
            file.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")

def convert_to_twosided(filepath, output_path):
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices
    faces = mesh.faces
    faces = np.concatenate((faces, faces[:, [0, 2, 1]]))   # Reverse face winding order
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    new_mesh.export(output_path)
    print("Dual faces mesh saved to", output_path)

def face_normal_to_vertex_normal(filepath, output_path):
    '''
    Convert face normals to vertex normals
    Return True if the conversion is triggered.
    '''
    vertices, face_normals, faces, using_vertex_normal = parse_obj(filepath)
    if using_vertex_normal:
        print("The obj file already using vertex normals, skip the conversion")
    else:
        vertex_normals = calculate_vertex_normals(vertices, faces)
        replace_face_normals_with_vertex_normals(filepath, vertex_normals, output_path)
        print("Vertex normals replaced and saved to", output_path)
    return not using_vertex_normal

if __name__ == "__main__":
    dir_name = os.path.join("assets", "sphere_10k")
    filename = "sphere_10k.obj"
    input_path = os.path.join(dir_name, filename)
    output_path = os.path.join(dir_name, filename)
    face_normal_to_vertex_normal(input_path, output_path)