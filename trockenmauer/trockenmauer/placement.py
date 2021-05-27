# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, placement.py
#

import random
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from trockenmauer.wall import Wall


def random_init_fixed_z(xyz: np.ndarray, wall: 'Wall', **kwargs):
    """
    Calculates an initial z value for a given position: On the ground area or
    on top of the bounding box of a stone

    :param xyz: Random coordinates within the range [lower_boundary .. upper boundary]
    :param wall:
    :return:
    """
    x, y, z = xyz
    z = .0001

    for stone in wall.stones:
        minimum, maximum = stone.aabb_limits

        if minimum[0] < x < maximum[0] and minimum[1] < y < maximum[1]:
            # placement is on a stone
            # print('on a stone, maybe top stone')
            z_temp = maximum[2] + 0.0001
            if z_temp > z:
                # print('on top of a stone', z, z_temp)
                z = z_temp  # update z

    return np.array([x, y, z])


def find_placement(wall: 'Wall'):
    # Find a placement within the given area of the wall

    x = random.uniform(0, wall.boundary.x)
    y = random.uniform(0, wall.boundary.y)
    z = 0.001  # on the ground (1mm above for cleaner intersections

    for stone in wall.stones:
        minimum = stone.mesh.vertices[stone.top].min(axis=0)
        maximum = stone.mesh.vertices[stone.top].max(axis=0)

        if minimum[0] < x < maximum[0] and minimum[1] < y < maximum[1]:
            # placement is on a stone
            # print('on a stone, maybe top stone')
            z_temp = stone.top_center[2] + 0.001
            if z_temp > z:
                # print('on top of a stone', z, z_temp)
                z = z_temp  # update z

    return np.array([x, y, z])
