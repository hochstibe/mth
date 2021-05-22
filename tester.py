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

s1 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
s2 = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])

mesh1 = pymesh.form_mesh(s1.mesh.vertices, s1.mesh.faces)
mesh2 = pymesh.form_mesh(s2.mesh.vertices, s2.mesh.faces)

intersection = pymesh.boolean(mesh1, mesh2, "intersection", engine='cork')
s3 = load_from_pymesh('geometry', intersection)
print(pymesh.distance_to_mesh(mesh1, np.array([[0, 0, 0]])))
print(pymesh.distance_to_mesh(mesh1, np.array([[1, 1, 1]])))
mesh3 = pymesh.form_mesh(np.array([[10, 10, 10], [10, 11, 10], [11, 11, 11]]), np.array([[0, 1, 2]]))
intersection = pymesh.boolean(mesh1, mesh3, "intersection", engine='cork')
if not np.any(intersection.vertices):
    print('empty intersection')

print('Closed meshes', mesh1.is_closed(), mesh2.is_closed(), intersection.is_closed(), mesh3.is_closed())

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
s1.add_shape_to_ax(ax)
s2.add_shape_to_ax(ax)
s3.add_shape_to_ax(ax, color='red')

plt.show()
