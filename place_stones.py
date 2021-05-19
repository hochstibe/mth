# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, place_stones.py
#

from trockenmauer.stone import Stone, Boundary, Wall
from trockenmauer.generate_stones import generate_regular_stone

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import random


def find_placement(wall: Wall):
    # Find a placement within the given area of the wall

    x = random.uniform(0, wall.boundary.x)
    y = random.uniform(0, wall.boundary.y)
    z = 0  # on the ground

    counter = 0
    for stone in wall.stones:
        # print('stone', counter)
        counter += 1
        minimum = stone.top.min(axis=1)
        maximum = stone.top.max(axis=1)
        # print('minimum x', minimum[0][0], 'maximum x', maximum[0][0])
        if minimum[0][0] < x < maximum[0][0] and minimum[0][1] < y < maximum[0][1]:
            # placement is on a stone
            x_temp, y_temp, z_temp = stone.top.mean(axis=1)[0]
            # print(z_temp, z)
            if z_temp > z:
                print('on top of a stone', z, z_temp)
                x, y, z = x_temp, y_temp, z_temp

    return x, y, z


def validate_placement(stone: Stone, wall: Wall):
    # validate
    # - stone is within the boundary
    # - normal of the ground area at the placement is in the same direction as the stones bottom side
    # - the closer to the boundary the better
    pass


boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
boundary.add_plot_to_ax(ax)

# place stones
for i in range(1):
    stone = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
    x, y, z = find_placement(wall)
    if z > 0:
        print('transforming height')
    stone.transform(t=np.array([x, y, z]))

    # add the stone to the wall
    wall.stones.append(stone)

    # add the stone to the plot
    stone.add_shape_to_ax(ax)

plt.show()

