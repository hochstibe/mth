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
from trockenmauer.utils import pick_smaller_stone
from trockenmauer.math_utils import Translation, Rotation, RotationTranslation, RZ_90
from trockenmauer.validation import ValidatorNormal
from trockenmauer.placement import solve_placement, random_xy_on_current_building_level


STONES = 5
FIREFLIES = 5
ITERATIONS = 3
FILENAME = None
SEED = None
# FILENAME = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{STONES}_stones_{FIREFLIES}_{ITERATIONS}_iterations'

# random generator for comparable results
random = np.random.default_rng(SEED)
boundary = Boundary(x=.5, y=.5, z=1, batter=.1)
wall = Wall(boundary)

validator = ValidatorNormal(intersection_boundary=True, intersection_stones=True,
                            distance2boundary=True, volume_below_stone=True,
                            distance2closest_stone=True
                            )
rz90 = Rotation(rotation=RZ_90, center=np.zeros(3))

# generate all stones and sort them with decreasing volume
# .25x.15x.1 with +- *=0.5
wall.init_stones(STONES, (.1, .3), (.075, .2), (.05, .15), random)
# normal_stones = [generate_regular_stone((.1, .3), (.075, .2), (.05, .15), random) for _ in range(STONES)]
# normal_stones.sort(key=lambda x: x.aabb_volume, reverse=True)
# Place the first stone manually
stone = wall.normal_stones[0]
# 90° Rotation around z-axis, translation to a corner of the wall
t = np.array([stone.aabb_limits[1][1],
              stone.aabb_limits[1][0] + stone.aabb_limits[1][2] * boundary.batter,
              -stone.bottom_center[2]])
stone.transform(RotationTranslation(rotation=RZ_90, center=np.zeros(3), translation=t))
wall.add_stone(stone)
wall.normal_stones.pop(0)

start = time()
placed_stones = 1
# for i in range(15):
while placed_stones < STONES:
    i = placed_stones
    print(f'stone {i} -----------------------------')
    # Pick the stones from biggest to smallest
    stone = wall.normal_stones[0]
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

    res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                          validator=validator, seed=random)
    """
    if res.validation_result.intersection:
        # height is in z direction, if the stone is flat
        print('the stone intersects. stone area:', stone.aabb_area,
              'intersection area', res.validation_result.intersection_area)
        print('Try a smaller stone')
        i = pick_smaller_stone(stone, normal_stones, overlapping_area=res.validation_result.intersection_area)
        if not i:  # no smaller stone available
            normal_stones.pop(0)
        else:
            stone = normal_stones[i]
            init_pos = res.position * np.ones((int(FIREFLIES/2), 1))  # start with only half the fireflies
            res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                                  validator=validator, seed=random)
            if res.validation_result.intersection:
                print('smaller stone also intersects...')
                print('original stone removed')
                normal_stones.pop(0)
            else:
                print('smaller stone without intersection')
                print(i, res.position, res.value)
                t = Translation(translation=res.position - stone.bottom_center)
                stone.transform(transformation=t)
                wall.add_stone(stone)
                normal_stones.pop(i)
                placed_stones += 1
            # If the length is the limiting factor, pick a stone with equal width, but reduce the length

            # if the width is the limiting factor, pick a stone with equal length, but reduce width

            # if both are limiting: -> pick one with the needed area

            # Todo: reduce the number of fireflies after 3 iterations (usually already converged
            # Maybe add a rotation in the first 3 iterations and use the better of 2 solutions per position
            # Maybe add the rotation only in the initial positions
    else:
        print(i, res.position, res.value)

        # Translate the stone to the optimal position and add to the wall
        t = Translation(translation=res.position - stone.bottom_center)
        stone.transform(transformation=t)
        wall.add_stone(stone)
        normal_stones.pop(0)
        placed_stones += 1
    """
    print(i, res.position, res.value)
    wall.add_stone(stone)
    print(stone.bottom_center)
    wall.normal_stones.pop(0)
    placed_stones += 1

# Stop criteria
stop = time()
m, s = divmod(stop-start, 60)
print(f"Successfully placed {len(wall.stones)} stones in {int(m)}'{round(s, 1)}''.")


wall.replay(fireflies=False, save=FILENAME)
