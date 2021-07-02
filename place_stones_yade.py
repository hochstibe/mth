# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 02.03.21, place_stones_yade.py
#

from time import time
from copy import copy
from datetime import datetime

from yade import qt, polyhedra_utils, utils
from yade.wrapper import (
    O, PolyhedraMat, ForceResetter, InsertionSortCollider, Bo1_Polyhedra_Aabb, Bo1_Wall_Aabb, Bo1_Facet_Aabb,
    InteractionLoop, Ig2_Wall_Polyhedra_PolyhedraGeom, Ig2_Polyhedra_Polyhedra_PolyhedraGeom,
    Ig2_Facet_Polyhedra_PolyhedraGeom, Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys,
    Law2_PolyhedraGeom_PolyhedraPhys_Volumetric, NewtonIntegrator, PyRunner
)
import numpy as np

from trockenmauer.stone import Boundary
from trockenmauer.wall import Wall
from trockenmauer.utils import pick_smaller_stone2, calc_smaller_stone_boundaries
from trockenmauer.math_utils import Rotation, RZ_90
from trockenmauer.fitness import FitnessNormal, FitnessFill
from trockenmauer.placement import solve_placement, random_xy_on_current_building_level, corner_placement


STONES = 30  # available stones
STONES_FILL = 0
STONES_LIM = 5  # number of (normal) stones to place
STONES_FILL_LIM = 0  # number of filling stones to place
FIREFLIES = 5
ITERATIONS = 5
FILENAME = None
SEED = 666
LEVEL_COVERAGE = 0.2
# FILENAME = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{STONES}_stones_{FIREFLIES}_{ITERATIONS}_iterations'

# random generator for comparable results
random = np.random.default_rng(SEED)
boundary = Boundary(x=1, y=.5, z=.5, batter=.1)
wall = Wall(boundary)

validator_n = FitnessNormal(intersection_boundary=True, intersection_stones=True,
                            distance2boundary=True, volume_below_stone=True,
                            distance2closest_stone=True, on_level=True
                            )
validator_f = FitnessFill(intersection_boundary=True, intersection_stones=True,
                          volume_below_stone=True,
                          distance2closest_stone=True, on_level=True,
                          delta_h=True)
rz90 = Rotation(rotation=RZ_90, center=np.zeros(3))

# generate all stones and sort them with decreasing volume
# .25x.15x.1 with +- *=0.5
wall.init_stones(STONES, (.1, .3), (.075, .2), (.05, .15), random, 'normal')
# create filling stones
wall.init_stones(STONES_FILL, (.05, .15), (.05, .15), (.01, .075), random, 'filling')
# Place the first stone manually
corner_placement(wall, 'left')
corner_placement(wall, 'right')

start = time()
placed_stones = 2
placed_filling_stones = 0
invalid_level_counter = 0  # counts failed placements on the current level (no smaller stone, intersections, not on lvl)
invalid_filling_counter = 0  # counts failed filling placements on the current level
invalid_level_update_counter = 0  # counts failed new levels (no stone placed on the current level). after 3, all stops
# for i in range(15):
running = True
while placed_stones < STONES_LIM and wall.normal_stones and running:

    stone_index = random.integers(0, np.max((1, np.round(len(wall.normal_stones)/4))))

    stone = copy(wall.normal_stones[stone_index])

    print(f'stone {placed_stones} - {stone.name} ----------------------------')
    rotation = random.choice([True, False])
    if rotation:
        stone.transform(rz90)
    # Find a placement
    print('free area:', wall.level_free, wall.level_free*wall.level_area, 'stone area', stone.aabb_area)
    init_pos = np.array([random_xy_on_current_building_level(wall, random) for _ in range(FIREFLIES)])

    res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                          validator=validator_n, seed=random)
    stone.best_firefly = res

    if res.validation_result.intersection:
        # height is in z direction, if the stone is flat
        print('the stone intersects. stone area:', stone.aabb_area,
              'intersection area', res.validation_result.intersection_area, res.validation_result.intersection_volume)
        print('Try a smaller stone')
        # add the stone with intersection -> orange
        wall.add_stone(stone, invalid_color='orange')
        intersections = []
        new = calc_smaller_stone_boundaries(stone, res.validation_result.intersection_boundary,
                                            res.validation_result.intersection_stones,
                                            # stone could be higher (up to the level limits) -- -> +
                                            stone.aabb_limits[1, 2] - wall.level_h[wall.level][1])
        stone_index_small, pos, rot = pick_smaller_stone2(wall.normal_stones, new)

        if not stone_index_small:  # no smaller stone available
            print('no smaller stone available -> filler? -> next level?')
            invalid_level_counter += 1
        else:
            stone = copy(wall.normal_stones[stone_index_small])
            print('smaller stone limits:', stone.aabb_limits[1] - stone.aabb_limits[0])
            if np.any(rot):
                print('rotate smaller stone')
                stone.transform(Rotation(rot))
            init_pos = np.array([pos[0], pos[1], 0]) * np.ones((int(FIREFLIES/2), 1))  # start with only half the fireflies
            res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                                  validator=validator_n, seed=random)
            stone.best_firefly = res
            if res.validation_result.intersection:
                print('smaller stone also intersects...')
                # add the stone with intersection -> red
                wall.add_stone(stone, invalid_color='red')
                invalid_level_counter += 1
            elif not res.validation_result.on_level:
                print('smaller stone above the level')

            else:
                invalid_level_counter = 0
                print('smaller stone without intersection')
                print(stone_index_small, res.position, res.value)
                wall.add_stone(stone)
                wall.normal_stones.pop(stone_index_small)
                placed_stones += 1
    else:  # No intersection
        if res.validation_result.on_level:
            invalid_level_counter = 0

            # Add to the wall (not as a valid stone
            wall.add_stone(stone)
            wall.normal_stones.pop(stone_index)
            placed_stones += 1
        else:
            # the stone is placed higher, retry or set new boundary limits
            print('!!! placement invalid (on a new level). retry', wall.level_h[wall.level], stone.aabb_limits[1][2])
            invalid_level_counter += 1

    # stopping criteria for building on the current level and building the wall in general
    if invalid_level_counter >= 3 or wall.level_free < LEVEL_COVERAGE \
            or not wall.normal_stones or placed_stones == STONES_LIM:
        print(f'------------- place filling stones --- {invalid_level_counter} {wall.level_free} ---------------------')
        # Update the level limits
        status = wall.update_level_limits()
        if status:  # a stone was placed on the level
            invalid_level_update_counter = 0
        else:  # no normal stones placed on the current level
            invalid_level_update_counter += 1
            if invalid_level_update_counter >= 3:
                running = False  # 3 times no stone placed on the level

        # filling stones
        invalid_filling_counter = 0
        valid_counter = 0
        while invalid_filling_counter < 3 and wall.filling_stones and placed_filling_stones < STONES_FILL_LIM:
            stone_index = random.integers(0, np.max((1, np.round(len(wall.filling_stones)/4))))
            stone = copy(wall.filling_stones[stone_index])
            print(f'stone {placed_filling_stones} - {stone.name} ----------------------------')
            init_pos = np.array([random_xy_on_current_building_level(wall, random) for _ in range(FIREFLIES)])
            res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                                  validator=validator_f, seed=random)
            stone.best_firefly = res

            if res.validation_result.intersection or res.validation_result.delta_h < 0:
                if res.validation_result.intersection:
                    print('intersection, value:', res.value, 'inters_volume (relative)',
                          res.validation_result.intersection_volume / stone.aabb_volume, stone.name)
                    print(res.validation_result.intersection_volume, res.validation_result.distance2closest_stone,
                          res.validation_result.delta_h)
                elif res.validation_result.delta_h < 0:
                    print('above stone', res.value, valid_counter, stone_index, stone.name, res.validation_result.delta_h)
                    print(wall.level_h[wall.level], stone.aabb_limits[1][2])
                    print(res.validation_result.distance2closest_stone, res.validation_result.delta_h)
                invalid_filling_counter += 1
                wall.add_stone(stone, 'teal')

                # smaller stone
                new_limits = calc_smaller_stone_boundaries(stone, res.validation_result.intersection_boundary,
                                                           res.validation_result.intersection_stones,
                                                           np.abs(res.validation_result.delta_h))
                stone_index_small, pos, rot = pick_smaller_stone2(wall.filling_stones, new_limits)

                if not stone_index_small:  # no smaller stone available
                    print('no smaller stone available')
                    invalid_filling_counter += 1
                else:
                    stone = copy(wall.filling_stones[stone_index_small])
                    print('smaller stone limits:', stone.aabb_limits[1] - stone.aabb_limits[0])
                    if np.any(rot):
                        print('rotate smaller stone')
                        stone.transform(Rotation(rot))
                    init_pos = np.array([pos[0], pos[1], 0]) * np.ones(
                        (int(FIREFLIES / 2), 1))  # start with only half the fireflies
                    res = solve_placement(wall, stone, n_fireflies=init_pos, n_iterations=ITERATIONS,
                                          validator=validator_f, seed=random)
                    stone.best_firefly = res

                    if res.validation_result.intersection or res.validation_result.delta_h < 0:
                        print('smaller stone also intersects or too high...')
                        # add the stone with intersection -> red
                        wall.add_stone(stone, invalid_color='red')
                        invalid_filling_counter += 1
                    elif res.validation_result.delta_h < 0:
                        print('smaller stone is above above stone', res.value, valid_counter, stone_index, stone.name,
                              res.validation_result.delta_h)

                    else:
                        invalid_filling_counter = 0
                        print('smaller stone without intersection')
                        print(stone_index_small, res.position, res.value)
                        wall.add_stone(stone)
                        wall.filling_stones.pop(stone_index_small)
                        placed_filling_stones += 1

            else:
                valid_counter += 1
                placed_filling_stones += 1
                print('valid filling stone')
                print(placed_filling_stones, res.position, res.value, res.validation_result.intersection)
                print(res.validation_result.intersection_volume, res.validation_result.distance2closest_stone,
                      res.validation_result.delta_h)
                stone.color = 'blue'
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

print('\nStopping criteria (main loop, AND):', placed_stones < STONES_LIM, running)
print('  Goto next lvl/filling, IF OR', invalid_level_counter >= 3,  wall.level_free < LEVEL_COVERAGE)
print('  Inner loop (filling AND):', invalid_filling_counter < 3)

# Placed stones:
print(f'{placed_stones} normal stones and {placed_filling_stones} filling stones placed')
print(f'  ({STONES} normal stones and {STONES_FILL} available)')
# Volume
wall_vol = wall.boundary.volume / wall.boundary.z * np.max([stone.aabb_limits[1][2] for stone in wall.stones])
print(f'Wall Volume: {wall_vol} (up to highest stones). Boundary Volume: {wall.boundary.volume} ')
stone_vol = np.sum([stone.aabb_volume for stone in wall.stones])
print(f'{len(wall.stones)} Stones Volume: {stone_vol}')
print(stone_vol / wall.boundary.volume, stone_vol / wall_vol)


# Stop criteria
stop = time()
m, s = divmod(stop-start, 60)
print(f"Successfully placed {len(wall.stones)} stones in {int(m)}'{round(s, 1)}''.")


wall.replay(fireflies=True, save=FILENAME)


# ----------------------------------------------------------------------------
# Physiksimulation

# add a concrete material
mat = PolyhedraMat()
mat.density = 2600  # kg/m^3
mat.young = 3E9  # Pa
mat.poisson = 0.2
mat.frictionAngle = 0.5  # rad
print(mat.dict())  # print all attributes

O.materials.append(mat)
# boundary: plane at z=0
floor = utils.wall(0, axis=2, sense=1, material=mat)
# stones
poly = [polyhedra_utils.polyhedra(mat, v=stone.mesh.vertices) for stone in wall.stones]

# Add the bodies to the scene
O.bodies.append(floor)
for i, p in enumerate(poly):
    p.state.pos = wall.stones[i].center
    O.bodies.append(p)
# O.bodies.append(stones[0])


# setup the animation
def checkUnbalanced():
    force = utils.unbalancedForce()
    if np.isnan(force) or force > len(poly) * 0.002:
        # at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
        # print("unbalanced forces = %.5f, position %f, %f, %f" % (
        # utils.unbalancedForce(), p.state.pos[0], p.state.pos[1], p.state.pos[2]))
        print('unbalanced forces =', force)
    else:
        print('no unalanced forces')


O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Polyhedra_Aabb(), Bo1_Wall_Aabb(), Bo1_Facet_Aabb()], verletDist=.001),
    InteractionLoop(
        [Ig2_Wall_Polyhedra_PolyhedraGeom(), Ig2_Polyhedra_Polyhedra_PolyhedraGeom(),
         Ig2_Facet_Polyhedra_PolyhedraGeom()],
        [Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys()],  # collision "physics"
        [Law2_PolyhedraGeom_PolyhedraPhys_Volumetric()]  # contact law -- apply forces
    ),
    # Add gravity
    # GravityEngine(gravity=(0, 0, -9.81)),
    NewtonIntegrator(damping=.9, gravity=(0, 0, -9.81)),  # damping=.5
    # Check for unbalanced forces every second
    PyRunner(command='checkUnbalanced()', realPeriod=1, label='checker')

]

O.dt = 0.0001
qt.Controller()
V = qt.View()

