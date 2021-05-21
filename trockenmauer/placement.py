# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, placement.py
#

import random

import numpy as np

from trockenmauer.stone import Stone, Wall


class Validator:
    """
    Validation functions for a placement

    validate
    - stone is within the boundary -> intersection
    - stone does not intersect with another -> intersection
    - normal of the ground area at the placement is in the same direction as the stones bottom side -> rotation
    - ...

    optimization
    - the closer to the boundary the better
    - the closer to other stones the better (rotation, that the stones are aligned on on of their face?)
    - less open space
    - ...
    """

    def __init__(self, intersections=False):
        """
        The parameters control, whitch validations will be executed
        :param intersections:
        """
        self.intersections = intersections

    def validate_within_boundary(self, x, y, stone: Stone, wall: Wall):
        pass

    def validate_normals(self, stone: 'Stone', wall):
        """
        The normal of the stone should point in the same direction as the normal of the wall on the given placement

        :param stone:
        :param wall:
        :return:
        """
        pass


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
