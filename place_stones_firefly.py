# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from time import time
from copy import copy
from datetime import datetime

import numpy as np

from trockenmauer.stone import Boundary
from trockenmauer.wall import Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.utils import pick_smaller_stone
from trockenmauer.math_utils import Translation, Rotation, RotationTranslation, RZ_90
from trockenmauer.validation import ValidatorNormal, ValidatorFill
from trockenmauer.placement import solve_placement, random_xy_on_current_building_level, corner_placement, find_random_placement


STONES = 20  # available stones
STONES_FILL = 5
STONES_LIM = 10  # number of (normal) stones to place
FIREFLIES = 20
ITERATIONS = 20
FILENAME = None
SEED = None
LEVEL_COVERAGE = 0.25
# FILENAME = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{STONES}_stones_{FIREFLIES}_{ITERATIONS}_iterations'

# random generator for comparable results
random = np.random.default_rng(SEED)
boundary = Boundary(x=1, y=.5, z=.5, batter=.1)
wall = Wall(boundary)

validator_n = ValidatorNormal(intersection_boundary=True, intersection_stones=True,
                              distance2boundary=True, volume_below_stone=True,
                              distance2closest_stone=True, on_level=True
                              )
validator_f = ValidatorFill(intersection_boundary=True, intersection_stones=True,
                            volume_below_stone=True,
                            distance2closest_stone=True, on_level=True
                            )
rz90 = Rotation(rotation=RZ_90, center=np.zeros(3))

# generate all stones and sort them with decreasing volume
# .25x.15x.1 with +- *=0.5
wall.init_stones(STONES, (.1, .3), (.075, .2), (.05, .15), random)
# create filling stones
wall.filling_stones = [generate_regular_stone((.05, .15), (.05, .15), (.01, .075), random) for _ in range(STONES_FILL)]
# normal_stones = [generate_regular_stone((.1, .3), (.075, .2), (.05, .15), random) for _ in range(STONES)]
# normal_stones.sort(key=lambda x: x.aabb_volume, reverse=True)
# Place the first stone manually
corner_placement(wall, 'left')
corner_placement(wall, 'right')

start = time()
placed_stones = 2
invalid_level_counter = 0
# for i in range(15):
running = True
while placed_stones < STONES_LIM and wall.normal_stones and running:

    # print(f'stone {placed_stones} -----------------------------')
    # Pick the stones from biggest to smallest -> pick one of the 25% biggest stones
    stone_index = random.integers(0, np.max((1, np.round(len(wall.normal_stones)/4))))

    stone = copy(wall.normal_stones[stone_index])

    print(f'stone {placed_stones} - {stone.name} ----------------------------')
    rotation = random.choice([True, False])
    if rotation:
        stone.transform(rz90)
    # Find a placement
    print('free area:', wall.level_free, wall.level_free*wall.level_area, 'stone area', stone.aabb_area)
    init_pos = np.array([random_xy_on_current_building_level(wall, random) for _ in range(FIREFLIES)])
    # if np.all(init_pos[:, 2] == wall.level_h[wall.level][0]):
    #     print('all placement on the current building level', (init_pos[init_pos[:, 2] == wall.level_h[wall.level][0]])[:, 2].flatten())
    # else:
    #     print('some placements on higher level after 5 tries')
    #     print(init_pos[init_pos[:, 2] == wall.level_h[wall.level][0]])

    res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                          validator=validator_n, seed=random)
    stone.best_firefly = res

    if res.validation_result.intersection:
        # height is in z direction, if the stone is flat
        print('the stone intersects. stone area:', stone.aabb_area,
              'intersection area', res.validation_result.intersection_area)
        print('Try a smaller stone')
        # add the stone with intersection -> red
        wall.add_stone(stone, invalid_color='orange')

        stone_index_small, rot = pick_smaller_stone(
            stone, wall.normal_stones, overlapping_area=res.validation_result.intersection_area)

        if not stone_index_small:  # no smaller stone available
            print('no smaller stone available -> filler? -> next level?')
            invalid_level_counter += 1
            # wall.normal_stones.pop(stone_index)
        else:
            stone = copy(wall.normal_stones[stone_index_small])
            stone.transform(Rotation(rot))
            init_pos = res.position * np.ones((int(FIREFLIES/2), 1))  # start with only half the fireflies
            res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                                  validator=validator_n, seed=random)
            stone.best_firefly = res
            if res.validation_result.intersection:
                print('smaller stone also intersects...')
                # add the stone with intersection -> orange
                wall.add_stone(stone, invalid_color='red')
                # print('original stone removed')
                # wall.normal_stones.pop(stone_index)
                invalid_level_counter += 1

            else:
                invalid_level_counter = 0
                print('smaller stone without intersection')
                print(stone_index_small, res.position, res.value)
                t = Translation(translation=res.position - stone.bottom_center)
                stone.transform(transformation=t)
                wall.add_stone(stone)
                wall.normal_stones.pop(stone_index_small)
                placed_stones += 1
            # If the length is the limiting factor, pick a stone with equal width, but reduce the length

            # if the width is the limiting factor, pick a stone with equal length, but reduce width

            # if both are limiting: -> pick one with the needed area

            # Todo: reduce the number of fireflies after 3 iterations (usually already converged
            # Maybe add a rotation in the first 3 iterations and use the better of 2 solutions per position
            # Maybe add the rotation only in the initial positions
    else:  # No intersection
        if res.validation_result.on_level:
            invalid_level_counter = 0
            # yay, stone is on the current level
            print(placed_stones, res.position, res.value, res.validation_result.intersection)

            # Translate the stone to the optimal position and add to the wall
            # t = Translation(translation=res.position - stone.bottom_center)
            # stone.transform(transformation=t)
            wall.add_stone(stone)
            wall.normal_stones.pop(stone_index)
            placed_stones += 1
        else:
            # the stone is placed higher, retry or set new boundary limits
            print('!!! placement invalid (on a new level). retry', wall.level_h[wall.level], stone.aabb_limits[1][2])
            invalid_level_counter += 1

    # stopping criteria for building on the current level and building the wall in general
    if invalid_level_counter >= 3 or wall.level_free < LEVEL_COVERAGE or not wall.normal_stones:
        print(f'------------- place filling stones --- {invalid_level_counter} {wall.level_free} ---------------------')
        # filling stones
        invalid_level_counter = 0
        valid_counter = 0
        while invalid_level_counter < 3 and wall.filling_stones:
            stone_index = random.choice(range(len(wall.filling_stones)))
            stone = copy(wall.filling_stones[stone_index])
            # without improved starting positions -> often above level?
            init_pos = np.array([find_random_placement(wall, random, wall.level_h[wall.level][0]) for _ in range(FIREFLIES)])
            res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                                  validator=validator_f, seed=random)

            if res.validation_result.intersection:
                print(valid_counter, stone_index, stone.name, 'intersection')
                invalid_level_counter += 1
                wall.add_stone(stone, 'blue')
            elif not res.validation_result.on_level:
                print(valid_counter, stone_index, stone.name, 'above level', res.validation_result.on_level)
                print(wall.level_h[wall.level], stone.aabb_limits[1][2])
                invalid_level_counter += 1
                # stone.color = 'teal'
                # wall.add_stone(stone)

            else:
                valid_counter += 1
                print(valid_counter, stone_index, stone.name, placed_stones, res.position, res.value)
                stone.color = 'lime'
                wall.add_stone(stone)
                wall.filling_stones.pop(stone_index)

        # go to next level or
        running = wall.next_level()
        invalid_counter = 0
    """
    print(placed_stones, res.position, res.value, res.validation_result.intersection)
    wall.add_stone(stone)
    wall.normal_stones.pop(stone_index)
    placed_stones += 1
    """

# Stop criteria
stop = time()
m, s = divmod(stop-start, 60)
print(f"Successfully placed {len(wall.stones)} stones in {int(m)}'{round(s, 1)}''.")


wall.replay(fireflies=False, save=FILENAME)
