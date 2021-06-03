# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from time import time
from datetime import datetime

import numpy as np

from trockenmauer.stone import Boundary
from trockenmauer.wall import Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.math_utils import Translation, Rotation, RotationTranslation, RZ_90
from trockenmauer.validation import ValidatorNormal
from trockenmauer.placement import solve_placement, random_xy_on_current_building_level


STONES = 10
FIREFLIES = 30
ITERATIONS = 40
FILENAME = None
SEED = None
# FILENAME = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{STONES}_stones_{FIREFLIES}_{ITERATIONS}_iterations'

# random generator for comparable results
random = np.random.default_rng(SEED)
boundary = Boundary(x=1, y=.5, z=1, batter=.1)
wall = Wall(boundary)

validator = ValidatorNormal(intersection_boundary=True, intersection_stones=True,
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
    print(f'stone {i} -----------------------------')
    # Generate, optimize and plot a stone
    stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(i))
    rotation = random.choice([True, False])
    if rotation:
        stone.transform(rz90)
    # Find a placement
    print('free area:', wall.level_free*wall.level_area, 'stone area', stone.aabb_area)
    init_pos = np.array([random_xy_on_current_building_level(wall, random) for _ in range(FIREFLIES)])
    if np.all(init_pos[:, 2] == wall.level_h[wall.level][0]):
        print('all placement on the current building level', (init_pos[init_pos[:, 2] == wall.level_h[wall.level][0]])[:, 2].flatten())
    else:
        print('some placements on higher level after 5 tries')
        print(init_pos[init_pos[:, 2] == wall.level_h[wall.level][0]])
    # --> Fazit: Bringt nichts, entweder stimmt etwas nicht mit den verbesserten Startpositionen oder
    # -->        es müssen Startpositionen ohne Überschneidungen sein

    res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                          validator=validator, seed=random)
    # Todo: reduce the number of fireflies after 3 iterations (usually already converged
    # Maybe add a rotation in the first 3 iterations and use the better of 2 solutions per position
    # Maybe add the rotation only in the initial positions
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
