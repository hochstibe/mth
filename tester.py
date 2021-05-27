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
# s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])

# Rotation matrix from v1 to v2
v_1 = np.array([[1], [0], [0]])
v_2 = np.array([[0], [1], [0]])


def rotation_align_vectors(v1, v2):
    """
    Calculates the rotation matrix from vector v1 vector to vector v2
    :param v1:
    :param v2:
    :return:
    """

    # normalize the length of the vectors
    v1 = (v1 / np.linalg.norm(v1)).flatten()
    v2 = (v2 / np.linalg.norm(v2)).flatten()

    if np.all(v1 == v2):
        return np.eye(3)
    elif np.all(-v1 == v2):
        return np.diag([-1, -1, -1])

    # cross product of the normalized vectors
    cross_p = np.cross(v1, v2)

    # rotation axis: normalized cross product
    u = cross_p / np.linalg.norm(cross_p)

    # the angle between vectors is the length of the cross product or the dot
    sin_phi = np.linalg.norm(np.linalg.norm(cross_p))
    cos_phi = np.dot(v1, v2)

    cross_p_matrix = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    r = cos_phi * np.eye(3) + sin_phi * cross_p_matrix + (1-cos_phi) * np.outer(u, u)

    return r
rot = rotation_align_vectors(v_1, v_2)
print('rot')
print(rot)
print('aligned vec')
print(rot @ v_1)
print(np.linalg.norm(rot @ v_1))



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
