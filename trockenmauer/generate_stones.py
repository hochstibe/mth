# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.2021, generate_stones.py
#

import random
from typing import List

import numpy as np

from trockenmauer.stone import Stone


def generate_regular_stone(x: float, y: float, z: float,  scale: List[float] = (1, 1),
                           edge_noise: float = 0.5,
                           name: str = None) -> 'Stone':
    """
    Stone generator for target size (in meter), uniform noise

    :param x: target length in x-direction
    :param y: target length in y-direction
    :param z: target length in z-direction
    :param scale: apply a scaling-factor to all three directions
    :param edge_noise: uniform noise for the length of an edge, within ``+- e*edge_noise``
    :param name: Name of the stone
    :return:
    """
    scale = random.uniform(scale[0], scale[1])

    # Coordinates of one corner (first quadrant, all coordinates positive)
    x = scale * random.uniform(x / 2 - x*edge_noise, x / 2 + x*edge_noise)
    y = scale * random.uniform(y / 2 - y*edge_noise, y / 2 + y*edge_noise)
    z = scale * random.uniform(z / 2 - z*edge_noise, z / 2 + z*edge_noise)

    a = [0, 0, 0]
    b = [x, 0, 0]
    c = [x, y, 0]
    d = [0, y, 0]
    e = [0, 0, z]
    f = [x, 0, z]
    g = [x, y, z]
    h = [0, y, z]
    v = np.array([a, b, c, d,  # lower 4 vertices
                  e, f, g, h])  # upper 4 vertices

    # v = np.array([[x, y, z], [-x, y, z], [-x, -y, z], [x, -y, z],  # upper 4 vertices
    #               [x, y, -z], [-x, y, -z], [-x, -y, -z], [x, -y, -z]])  # lower 4 vertices
    if name:
        return Stone(v, name=name)
    return Stone(v)


def add_noise_to_vertices(vert, s=0.1):
    """
    Adds noise to the vertices (to each coordinate separately)

    :param vert:
    :param s:
    :return:
    """
    for v in vert:
        for c in v:
            c += random.gauss(0, s)

    return Stone(vert, name='Noisy vertices')
