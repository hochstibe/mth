# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, placement.py
#

from typing import TYPE_CHECKING, Union, Iterable

import numpy as np

from .firefly import FireflyProblem
from .math_utils import RZ_90, RotationTranslation

if TYPE_CHECKING:
    from . import Wall, Stone, Validator


def solve_placement(wall: 'Wall', stone: 'Stone', n_fireflies: Union[int, np.ndarray, Iterable],
                    n_iterations: int, validator: 'Validator',
                    seed: Union[np.random.Generator, any]):
    """
    Find the optimal placement
    """
    # Find a placement
    problem = FireflyProblem(n_fireflies, validator.fitness, wall.boundary.aabb_limits[0], wall.boundary.aabb_limits[1],
                             iteration_number=n_iterations, seed=seed,
                             pos_function=xy_fixed_z, stone=stone, wall=wall)
    res, history = problem.solve()
    stone.position_history = history
    # print(fitness, xyz)
    return res


def corner_placement(wall: 'Wall', corner='left'):
    stone = wall.normal_stones[0]
    bnd = wall.boundary
    lim = stone.aabb_limits
    # 90° Rotation around z-axis, translation to a corner of the wall
    if corner == 'left':
        t = np.array([lim[1][1],
                      lim[1][0] + lim[1][2] * bnd.batter,
                      -stone.bottom_center[2]])
    elif corner == 'right':
        t = np.array([bnd.x - lim[1][1],
                      bnd.y - lim[1][0] - lim[1][2] * bnd.batter,
                      -stone.bottom_center[2]])
    else:
        raise ValueError
    stone.transform(RotationTranslation(rotation=RZ_90, center=np.zeros(3), translation=t))
    stone.alpha = 1
    wall.add_stone(stone)
    wall.normal_stones.pop(0)


def random_xy_on_current_building_level(wall: 'Wall',
                                        random: 'np.random.Generator' = np.random.default_rng(), n_tries: int = 5):
    # try to generate a random coordinate for the free area
    h_min, h_max = wall.level_h[wall.level]
    # calculate empty space for the current level in x/y direction
    boundary = np.array([[0, wall.boundary.batter * h_min],
                         [wall.boundary.x, wall.boundary.y - wall.boundary.batter * h_min]])

    counter = 0
    xyz = np.array([])  # good solution
    xy = np.array([[]])  # random coordinates within boundary
    blocked = True

    while counter < n_tries and blocked:
        xy = random.uniform(boundary[0], boundary[1], 2)
        # print('counter', counter)

        blocked = blocked_area(np.array([*xy, h_min]), wall=wall)
        # print(' ', counter, n_tries, xy, blocked)
        if blocked:
            # print(' blocked area -> redo', counter)
            counter += 1
        else:
            # print(' not blocked, fine', counter)
            xyz = np.array([*xy, h_min])
            # print('  valid coordinates', xyz, np.any(xyz))

    if np.any(xyz):
        return xyz
    else:
        return np.array([*xy, h_min])


def updated_current_building_level(xyz: np.ndarray, stone: 'Stone', wall: 'Wall'):

    h_stone = xyz[2] + stone.aabb_limits[1][2] - stone.aabb_limits[0][2]
    h_min, h_max = wall.level_h[wall.level]

    if h_stone > h_max + (h_max - h_min) / 2:
        # top-z of the stone significantly higher than max-z of the current level
        # set a new range for the new level: min = previous max; max = new
        wall.level += 1
        wall.level_h.append([h_max, h_stone])

    else:
        # on the current level, set h_max of the current level (could increase a bit)
        wall.level_h[wall.level] = [h_min, np.max([h_max, h_stone])]


def blocked_area(xyz: np.ndarray, wall: 'Wall'):
    # The stones of a certain height limit the available free space to place a stone
    # xyz: coordinates for the bottom center of a new stone
    x, y, z = xyz

    # calculate empty space for the current level
    bb = np.array([[0, wall.boundary.batter * z, z],
                   [wall.boundary.x, wall.boundary.y - wall.boundary.batter * z, z+.01]])

    # All stones within one level
    hits = wall.r_tree.intersection(bb.flatten(), objects=True)  # -> list of indices in index
    limits = np.array([item.bbox for item in hits])  # bbox = (xmin, ymin, zmin, xmax, ymax, zmax)
    # print('blocked area, stones on level:', len(limits))
    # Total area of all hits
    blocked = np.sum(np.prod(limits[:, 3:5] - limits[:, :2], axis=1))
    # ratio of blocked area to total area
    # blocked = blocked / np.prod(bb[1][:2] - bb[2][:2])
    # Todo: only normal stones block the area

    # if the coordinate is within the xy limits of another stone (boolean index)
    in_stones = np.all((limits[:, :2] <= xyz[:2]) & (limits[:, 3:5] >= xyz[:2]), axis=1)
    # print('  in_stones', in_stones, np.any(in_stones))

    if np.any(in_stones):
        # print('  coordinates lie on a blocked area')
        return True  # Coordinates lie on a blocked are
    else:
        return False  # Coordinate lie in empty area


def xy_fixed_z(xyz: np.ndarray, wall: 'Wall', **kwargs):
    """
    Calculates an initial z value for a given position: On the ground area or
    on top of the bounding box of a stone

    :param xyz: Random coordinates within the range [lower_boundary .. upper boundary]
    :param wall:
    :return:
    """

    stones_limits = np.array([s.aabb_limits for s in wall.stones])
    mi = stones_limits[:, 0]  # np.min(stones_limits, axis=1)
    ma = stones_limits[:, 1]

    # if the coordinate is within the xy limits of another stone (boolean index)
    in_stones = np.all(((mi[:, :2] <= xyz[:2]) & (ma[:, :2] >= xyz[:2])), axis=1)

    if np.any(in_stones):  # at least one stone
        z = np.max(ma[in_stones], axis=0)[2] + .0001
    else:
        z = .0001

    xyz[2] = z

    return xyz


def find_random_placement(wall: 'Wall', random: 'np.random.Generator' = np.random.default_rng(), z: float = 0):
    # Find a placement within the given area of the wall

    x = random.uniform(0, wall.boundary.x)
    y = random.uniform(z * wall.boundary.batter, wall.boundary.y - z * wall.boundary.batter)
    z = 0.001  # on the ground (1mm above for cleaner intersections

    for stone in wall.stones:
        minimum = stone.mesh.vertices[stone.top].min(axis=0)
        maximum = stone.mesh.vertices[stone.top].max(axis=0)

        if minimum[0] < x < maximum[0] and minimum[1] < y < maximum[1]:
            # placement is on a stone
            # print('on a stone, maybe top stone')
            z_temp = stone.top_center[2] + 0.001
            if z_temp > z:
                # print('on top of a stone', z, z_temp)
                z = z_temp  # update z

    return np.array([x, y, z])
