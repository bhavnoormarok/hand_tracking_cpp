import open3d as o3d
import numpy as np

import time 

mesh = o3d.geometry.TriangleMesh()

np_vertices = np.zeros((778, 3))
for i in range(778):
    for j in range(3):
        np_vertices[i,j] = float(input())

np_triangles = np.zeros((1538, 3)).astype(np.int32)

for i in range(1538):
    for j in range(3):
        np_triangles[i,j] = int(input())

mesh.vertices = o3d.utility.Vector3dVector(np_vertices)

mesh.triangles = o3d.utility.Vector3iVector(np_triangles)


print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices).shape)
print('Triangles:')
print(np.asarray(mesh.triangles).shape)


print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals).shape)


# d_near=600.0; d_far=1200.0;  fx=366.085;  fy=366.085;  cx=259.229;  cy=207.968

# left = -cx * d_near // fx;  # set u = 0 in above formula, left of image plane has u = 0
# top = cy * d_near // fy;  


# E = np.array([0.0, 0.0, 0.0])
# A = np.array([0.0, 0, 1.0])
# v = np.array([0.0, -1.0, 0.0])
# #o3d.visualization.draw_geometries([mesh], "Open3D", 512, 480, 50, 50, False, False, False,A,v,E, 1.0)
# o3d.visualization.draw_geometries([mesh])



# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualization
vis.add_geometry(mesh)

# Update and render the visualization
vis.poll_events()
vis.update_renderer()

time.sleep(10)
# Close the visualization window
vis.destroy_window()