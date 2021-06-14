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
from trockenmauer.math_utils import Translation, tetra_volume
from trockenmauer.plot import set_axes_equal

stones = [generate_regular_stone((.15, .4), (.1, .2), (.5, .15)) for _ in range(10)]

# order them by volume
stones.sort(key=lambda x: x.aabb_volume, reverse=True)
print([stone.aabb_area for stone in stones])
print([(stone.length, stone.width) for stone in stones])

# add a rotation attribute
