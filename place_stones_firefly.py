# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from time import time
from datetime import datetime
import random

import numpy as np

from trockenmauer.stone import Boundary
from trockenmauer.wall import Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.math_utils import Translation, Rotation, RotationTranslation, RZ_90
from trockenmauer.validation import Validator
from trockenmauer.placement import solve_placement


STONES = 10
FIREFLIES = 15
ITERATIONS = 30
# FILENAME = None
FILENAME = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{STONES}_stones_{FIREFLIES}_{ITERATIONS}_iterations'

boundary = Boundary(x=1, y=.5, z=1, batter=.1)
wall = Wall(boundary)

validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True, volume_below_stone=True,
                      distance2closest_stone=True
                      )
rz90 = Rotation(rotation=RZ_90, center=np.zeros(3))
# Place the first stone manually
stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(-1))
# 90° Rotation around z-axis, translation to a corner of the wall
t = np.array([stone.aabb_limits[1][1],
              stone.aabb_limits[1][0] + stone.aabb_limits[1][2] * boundary.batter,
              -stone.bottom_center[2]])
stone.transform(RotationTranslation(rotation=RZ_90, center=np.zeros(3), translation=t))
wall.add_stone(stone)

# generate stones

start = time()
for i in range(STONES):
    # Generate, optimize and plot a stone
    stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(i))
    rotation = random.choice([True, False])
    if rotation:
        stone.transform(rz90)
    # Find a placement
    res = solve_placement(wall, stone, FIREFLIES, ITERATIONS, validator)
    print(i, res.position, res.value)

    # Translate the stone to the optimal position and add to the wall
    t = Translation(translation=res.position - stone.bottom_center)
    stone.transform(transformation=t)
    wall.add_stone(stone)


# Stop criteria
stop = time()
m, s = divmod(stop-start, 60)
print(f"Successfully placed {len(wall.stones)} stones in {int(m)}'{round(s, 1)}''.")


wall.replay(fireflies=True, save=FILENAME)
