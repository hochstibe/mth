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

stones = [['s1', 's1'], ['s2', 's2', 's2'], ['s3', 's3', 's3', 's3']]
frames = len(stones) + sum([len(stone) for stone in stones])
print(frames, 'frames')

# lookup dictionary for the stones
lookup = {}
frame = 0
for i in range(len(stones)):
    lookup[i] = range(frame, frame + 1 + len(stones[i]))
    frame += len(stones[i]) + 1
print(lookup)

for frame in range(frames):
    # find the stone
    stone_index = [i for i in lookup if frame in lookup[i]][0]

    history_index = frame - lookup[stone_index].start
    # print('frame', frame, 'stone', stone_index, history_index, lookup[stone_index])
    if history_index in range(len(stones[stone_index])):
        print('frame', frame, 'stone', stone_index, 'history', history_index)
    else:
        print('frame', frame, 'stone', stone_index, 'plot stone')

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
