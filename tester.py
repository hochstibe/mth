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
from trockenmauer.math_utils import Translation, tetra_volume, Rotation
from trockenmauer.plot import set_axes_equal


def rz(phi):
    phi = np.pi * phi/180
    return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])


stones = [generate_regular_stone((.1, .3), (.075, .2), (.05, .15)) for _ in range(15)]

# order them by volume
stones.sort(key=lambda x: x.aabb_volume, reverse=True)
print([stone.aabb_area for stone in stones])
print([(stone.length, stone.width) for stone in stones])

stone_volume = np.sum([])

# add a rotation attribute
stone = stones[0]
nrows = 3
ncols = 3

fig = plt.figure(figsize=(ncols*3.1, nrows*3))
fig.suptitle('Rotation')

# original position
ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
stone.add_shape_to_ax(ax)
ax.set_title('original')

# 30°
stone.transform(Rotation(rz(30)))
ax = fig.add_subplot(nrows, ncols, 2, projection='3d')
stone.add_shape_to_ax(ax)
ax.set_title('30°')

# 30° + 30° = 60°
stone.transform(Rotation(rz(30)))
ax = fig.add_subplot(nrows, ncols, 3, projection='3d')
stone.add_shape_to_ax(ax)
ax.set_title('30+30°')

# 60-60 = 0
stone.transform(Rotation(rz(-60)))
ax = fig.add_subplot(nrows, ncols, 4, projection='3d')
stone.add_shape_to_ax(ax)
ax.set_title('60-600°')

# 0+(-30)T = 30
stone.transform(Rotation(rz(-30).T))
ax = fig.add_subplot(nrows, ncols, 5, projection='3d')
stone.add_shape_to_ax(ax)
ax.set_title('0+(-30)T')
# 30 + 60T = -30
stone.transform(Rotation(rz(60).T))
ax = fig.add_subplot(nrows, ncols, 6, projection='3d')
stone.add_shape_to_ax(ax)
ax.set_title('30+(60)T = -30')

# stone 2: 0°
stone2 = stones[1]
ax = fig.add_subplot(nrows, ncols, 7, projection='3d')
stone2.add_shape_to_ax(ax)
ax.set_title('s2: 0°')
# stone3: stone2 + 30°
stone3 = stones[2]
r = rz(30) @ stone2.current_rotation
stone3.transform(Rotation(r))
ax = fig.add_subplot(nrows, ncols, 8, projection='3d')
stone3.add_shape_to_ax(ax)
ax.set_title('s3: s2 + 30°')
# stone4: stone1 + 45° = 15°
stone4 = stones[3]
stone4.transform(Rotation(rz(45) @ stone.current_rotation))
ax = fig.add_subplot(nrows, ncols, 9, projection='3d')
stone4.add_shape_to_ax(ax)
ax.set_title('s3: s1 + 45° = 15°')




plt.show()


