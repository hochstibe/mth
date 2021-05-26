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

class Test:
    _a = None
    _hidden = 'not settable'

    @property
    def hidden(self):
        return self._hidden

    @property
    def a(self):
        if not np.any(self._a):
            self._a = np.array([1, 2])
        return self._a

    @a.setter
    def a(self, arr):
        self._a = arr

    def b(self):
        print('Test')

class Tester(Test):
    def __init__(self):
        super().__init__()
    def b(self):
        super().b()
        print('Tester')

t = Test()
if np.any(t.a):
    print('setter worked')
print(t.a[0])
tt = Tester()
tt.b()
print(t.hidden)
t.hidden = 'set hidden attr'

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
