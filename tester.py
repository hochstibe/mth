# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#

import pymesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.utils import load_from_pymesh
from trockenmauer.math_utils import Translation, tetra_volume
from trockenmauer.plot import set_axes_equal

# Distance to boundary






# ---------------------------------------------------------------
# Tetrahedralization
# s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0, scale=[1, 2])
s2 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])

mesh1 = pymesh.form_mesh(s1.mesh.vertices, s1.mesh.faces)
mesh2 = pymesh.form_mesh(s2.mesh.vertices, s2.mesh.faces)

intersection = pymesh.boolean(mesh1, mesh2, "intersection", engine='cork')
print(intersection.num_vertices, intersection.num_faces, intersection.num_voxels)
s3 = load_from_pymesh('geometry', intersection)
# print(pymesh.distance_to_mesh(mesh1, np.array([[0, 0, 0]])))
# print(pymesh.distance_to_mesh(mesh1, np.array([[1, 1, 1]])))
# mesh3 = pymesh.form_mesh(np.array([[10, 10, 10], [10, 11, 10], [11, 11, 11]]), np.array([[0, 1, 2]]))
# intersection = pymesh.boolean(mesh3, mesh1, "difference", engine='cgal')

t = Translation(np.array([2, 2, 2]))
s1.transform(t)
# merged = pymesh.merge_meshes([s1.mesh, s2.mesh])
merged = pymesh.boolean(s1.mesh, s2.mesh, 'union', 'cgal')
merged = load_from_pymesh('geometry', merged)

if not np.any(intersection.vertices):
    print('empty intersection')

print(s1.mesh.num_vertices, s1.mesh.num_faces, s1.mesh.num_voxels)
tetra = pymesh.tetrahedralize(s1.mesh, 20, engine='tetgen')
print(tetra.num_vertices, tetra.num_faces, tetra.num_voxels)
# print(tetra.voxels)
# print(tetra.attribute_names, tetra.get_attribute('voxel_volume'))
tetra = load_from_pymesh('geometry', tetra, 'tetra')
print('volume: .4 * .2 * .1 =', .4*.2*.1, np.linalg.norm(np.max(s1.mesh.vertices, axis=0) - np.min(s1.mesh.vertices, axis=0)))
tetra_volume(tetra.mesh.vertices.T, tetra.mesh.voxels)

distances, faces, points = pymesh.distance_to_mesh(tetra.mesh, [np.array([0, 0, 0]), np.array([.1, .1, .1])])
print('distances', distances)
print('faces', faces)
print('points', points)


# print('Closed meshes', mesh1.is_closed(), mesh2.is_closed(), intersection.is_closed(), mesh3.is_closed())

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# s1.add_shape_to_ax(ax)
# s2.add_shape_to_ax(ax)
# s3.add_shape_to_ax(ax, color='red')
# merged.add_shape_to_ax(ax, color='orange')
tetra.add_shape_to_ax(ax, color='cyan')

set_axes_equal(ax)
plt.show()
