# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#


import pymesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import random

from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.utils import load_from_pymesh
from trockenmauer.math_utils import Translation, tetra_volume
from trockenmauer.plot import set_axes_equal

# footprint of bounding box
# s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
scale = np.array([2, .5, 1])
alpha = .01 * scale  # .25
beta = 1  # 1
gamma = 1 / np.sqrt(scale)  # .97

position = np.array([0, 0, 0])
better_position = np.array([2, .5, 1])

distance = np.linalg.norm(position - better_position)
new_pos = position + beta*np.exp(-gamma*(distance**2)) * (better_position-position) + alpha*(random.uniform(-1, 1, 3))
print('alpha', alpha, 'beta', beta, 'gamma', gamma)
print(position, 'original distance', distance)
print(new_pos, 'moved distance', np.linalg.norm(new_pos - position))
print(beta*np.exp(-gamma*(distance**2)) * (better_position-position))
print(alpha*(random.uniform(0, 1)-0.5))


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
