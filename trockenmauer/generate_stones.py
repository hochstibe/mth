# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.2021, generate_stones.py
#

from typing import Union, Tuple

import numpy as np

from .stone import Stone


def generate_regular_stone(x: Union[Tuple[float, float], float],
                           y: Union[Tuple[float, float], float],
                           z: Union[Tuple[float, float], float],
                           random: 'np.random.Generator' = np.random.default_rng(),
                           edge_noise: float = 0.3,
                           name: str = None) -> 'Stone':
    """
    Stone generator for target size (in meter), uniform noise

    :param x: target length in x-direction or boundary (min, max)
    :param y: target length in y-direction or boundary (min, max)
    :param z: target length in z-direction or boundary (min, max)
    :param random: Generator for random numbers
    :param edge_noise: uniform noise for the length of an edge, within ``+- e*edge_noise``
    :param name: Name of the stone
    :return:
    """
    if isinstance(x, float) and isinstance(y, float) and isinstance(z, float):
        low = np.array([x, y, z]) - np.array([x, y, z]) * edge_noise
        high = np.array([x, y, z]) + np.array([x, y, z]) * edge_noise
    else:
        low = np.array([x[0], y[0], z[0]])
        high = np.array([x[1], y[1], z[1]])

    # Coordinates of one corner (first quadrant, all coordinates positive)
    x, y, z = random.uniform(low, high, 3)

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
