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

stone = stones[0]
# 20% overlap
o = stone.aabb_area*.1
# same width -> reduce length
l = o/stone.width
target_l = stone.length - l
target_w = stone.width
if target_l < stone.width:
    print('rotate target stone')
    target_w = target_l
    target_l = stone.width

print('search stone with dimensions l <', target_l, 'w <', target_w)
# next works good, if the stones are sorted by their volume / area
match = next(s for s in stones if s.length < target_l and s.width < target_w)
print(stones.index(match), match.length, match.width)
matches = [s for s in stones if s.length < target_l and s.width < target_w]
print(len(matches), [m.aabb_area for m in matches], matches[0].length, matches[0].width)
