# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, placement.py
#

import random
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .stone import Wall


def find_placement(wall: 'Wall'):
    # Find a placement within the given area of the wall

    x = random.uniform(0, wall.boundary.x)
    y = random.uniform(0, wall.boundary.y)
    z = 0.001  # on the ground (1mm above for cleaner intersections

    for stone in wall.stones:
        minimum = stone.top.min(axis=0)
        maximum = stone.top.max(axis=0)

        if minimum[0] < x < maximum[0] and minimum[1] < y < maximum[1]:
            # placement is on a stone
            z_temp = stone.top_center[2]
            if z_temp > z:
                # print('on top of a stone', z, z_temp)
                z = z_temp

    return np.array([x, y, z])
