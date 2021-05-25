# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#

from copy import copy

import pymesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from aabbtree import AABB

from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.utils import load_from_pymesh
from trockenmauer.math_utils import Translation, tetra_volume
from trockenmauer.plot import set_axes_equal

# footprint of bounding box
s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
print(s1.aabb, s1.aabb[2, 1])
print(np.vstack((s1.aabb_limits.T[:2, :], [0, 123])))
print(np.vstack((s1.aabb[:2, :], [0, 456])))
bb = AABB(np.vstack((s1.aabb_limits.T[:2, :], [0, 10])))
print(bb)
print(bb[:, 1] - bb[:, 0])
c = bb[:, 1] - bb[:, 0]
print(c[0] * c[1] * c[2])
print(bb.volume)


# Footprint of mesh
# s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
#
# vertices = copy(s1.mesh.vertices)
# print(vertices.shape)
# vertices[:, 2] = 0
# print(vertices.shape)
# # print(vertices)
#
# # convex hull of 2d problem
# mesh = pymesh.form_mesh(s1.mesh.vertices[:, :2], s1.mesh.faces)
# mesh_2, info = pymesh.remove_duplicated_vertices(mesh, tol=1e-3)
# tri = pymesh.triangle()
# tri.points = mesh_2.vertices
# tri.verbosity = 2
# tri.run()
# print('2d convex hull', len(tri.mesh.vertices), len(tri.mesh.faces))
# # convex hull of 3d problem
# bottom_vertices = np.vstack(mesh_2.vertices.T, np.zeros(3)).T
# print(s1.top_center[2] * np.ones(3))
# top_vertices = np.vstack(mesh_2.vertices.T, s1.top_center[2] * np.ones(3))
# vertices = np.vstack(bottom_vertices, top_vertices)
#
#
#
# # Try to get the outer hull --> triangles are doubled (one normal up, one down)
# # tri = pymesh.form_mesh(vertices, s1.mesh.faces)
# # print(len(tri.vertices), len(tri.faces))
# # tri = pymesh.compute_outer_hull(tri)
# # print('hull', len(tri.vertices), len(tri.faces))
# #
# # tri, info = pymesh.remove_duplicated_vertices(tri, tol=1e-3)
# # print(len(tri.vertices), len(tri.faces), info)
# # tri, info = pymesh.remove_degenerated_triangles(tri, 2)
# # print(len(tri.vertices), len(tri.faces), info)
# # print(tri.faces)
# # tri, info = pymesh.remove_duplicated_faces(tri) --> removes all faces
# # print(len(tri.vertices), len(tri.faces), info)
"""
tri = pymesh.triangle()
tri.points = vertices[:, :2]
tri.triangles = s1.mesh.faces
tri.verbosity = 2
tri.run()

floor = tri.mesh
print(floor.vertices)
print(floor.faces)


fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# s1.add_shape_to_ax(ax)
# s2.add_shape_to_ax(ax)
# s3.add_shape_to_ax(ax, color='red')
# merged.add_shape_to_ax(ax, color='orange')
tetra.add_shape_to_ax(ax, color='cyan')
tetra2.add_shape_to_ax(ax, color='brown')

set_axes_equal(ax)
plt.show()
"""
