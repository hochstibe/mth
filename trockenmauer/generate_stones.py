# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.2021
# generate_stones.py -

import random

import numpy as np

from trockenmauer.stone import Stone


def generate_regular_stone(x: float, y: float, z: float,  scale: float = 1,
                           x_noise: float = 0.1, y_noise: float = 0.05, z_noise: float = 0.05,
                           name: str = '') -> 'Stone':
    """
    Stone generator for target size (in meter), uniform noise

    :param x: target length in x-direction
    :param y: target length in y-direction
    :param z: target length in z-direction
    :param scale: apply a scaling-factor to all three directions
    :param x_noise:
    :param y_noise:
    :param z_noise:
    :return:
    """

    # Coordinates of one corner (first quadrant, all coordinates positive)
    x = scale * random.uniform(x - x_noise, x + x_noise) / 2
    y = scale * random.uniform(y - y_noise, y + y_noise) / 2
    z = scale * random.uniform(z - z_noise, z + z_noise) / 2

    v = np.array([[x, y, z], [-x, y, z], [-x, -y, z], [x, -y, z],  # upper 4 vertices
                  [x, y, -z], [-x, y, -z], [-x, -y, -z], [x, -y, -z]])  # lower 4 vertices
    return Stone(v, name)
