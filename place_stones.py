# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, place_stones.py
#

from trockenmauer.stone import Stone, Boundary, Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.utils import set_axes_equal
from trockenmauer.math_utils import Translation

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import random


def find_placement(wall: Wall):
    # Find a placement within the given area of the wall

    x = random.uniform(0, wall.boundary.x)
    y = random.uniform(0, wall.boundary.y)
    z = 0  # on the ground

    for stoned in wall.stones:
        minimum = stoned.top.min(axis=0)
        maximum = stoned.top.max(axis=0)

        if minimum[0] < x < maximum[0] and minimum[1] < y < maximum[1]:
            # placement is on a stone
            z_temp = stoned.top_center[2]
            if z_temp > z:
                # print('on top of a stone', z, z_temp)
                z = z_temp

    return np.array([x, y, z])


def validate_placement(stone: Stone, wall: Wall):
    # validate
    # - stone is within the boundary -> intersection
    # - stone does not intersect with another -> intersection
    # - normal of the ground area at the placement is in the same direction as the stones bottom side -> rotation

    # optimization
    # - the closer to the boundary the better
    # - the closer to other stones the better (rotation, that the stones are aligned on on of their face?)
    # - less open space
    pass


boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
boundary.add_plot_to_ax(ax)

# place stones
for i in range(20):
    stone = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
    xyz = find_placement(wall)
    t = Translation(translation=xyz - stone.bottom_center)
    stone.transform(transformation=t)

    # add the stone to the wall
    wall.stones.append(stone)

    # add the stone to the plot
    stone.add_shape_to_ax(ax)

set_axes_equal(ax)
plt.show()

