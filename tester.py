# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#

from typing import List, Optional

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.math_utils import Translation, RZ_90

random = np.random.default_rng()


def aabb_overlap(limits: 'np.ndarray', other_limits: 'np.ndarray') -> Optional['np.ndarray']:
    """
    calculates the overlapping region. if no overlap, it returns None

    :param limits: aabb as [[x_min, y_min, ...], [x_max, y_max, ...]]
    :param other_limits: other aabb
    :return: limits of the intersection or None
    """

    # minimum of both maxima; maximum of both minima
    overlap_min = np.max(np.array([limits[0], other_limits[0]]), axis=0)
    overlap_max = np.min(np.array([limits[1], other_limits[1]]), axis=0)

    # difference is positive, if there is an overlap
    diff = overlap_max - overlap_min

    if np.any(diff <= 0):  # todo: ?
        return None

    return np.array([overlap_min, overlap_max])


def calc_smaller_stone_boundaries(invalid_stone: 'np.ndarray', intersections: List['np.ndarray']):
    """
    Calculate the max_bb for a stone and its intersections
    """
    x_min, y_min, x_max, y_max = invalid_stone.flatten()

    for n, i in enumerate(intersections):
        print('intersection', n)
        axs[n].axis('equal')
        axs[n].set_xlim(-.1, 2.1)
        axs[n].set_ylim(-1.1, 1.1)
        axs[n].add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='green'))
        ix_min, iy_min, ix_max, iy_max = i.flatten()
        axs[n].add_patch(Rectangle((ix_min, iy_min), ix_max - ix_min, iy_max - iy_min, color='red'))

        hit = aabb_overlap(np.array([[x_min, y_min], [x_max, y_max]]), i)

        if np.any(hit):
            print('  this stone still intersects')

            # List the possible reductions of the aabb limits
            reductions = list()  # [[name, amount, area after reduction], ...]

            # x_max
            if x_min < ix_min < x_max:
                reductions.append(['r_x_max', x_max - ix_min, (ix_min - x_min) * (y_max - y_min)])
                print('  reduce x_max')
            # y_max
            if y_min < iy_min < y_max:
                reductions.append(['r_y_max', y_max - iy_min, (iy_min - y_min) * (x_max - x_min)])
                print('  reduce y_max')
            # x_min
            if x_max > ix_max > x_min:
                reductions.append(['r_x_min', ix_max - x_min, (x_max - ix_max) * (y_max - y_min)])
                print('  reduce x_min')
            # y_min
            if y_max > iy_max > y_min:
                reductions.append(['r_y_min', iy_max - y_min, (y_max - iy_max) * (x_max - x_min)])
                print('  reduce y_min')

            print(reductions)

            red = reductions[0]
            for r in reductions:
                if r[2] > red[2]:
                    red = r
            print('  reduce', red)

            if red[0] == 'r_x_min':
                x_min += red[1]
            elif red[0] == 'r_y_min':
                y_min += red[1]
            elif red[0] == 'r_x_max':
                x_max -= red[1]
            elif red[0] == 'r_y_max':
                y_max -= red[1]

            axs[n].add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='blue'))
        else:
            print('  no intersection anymore')

    new = np.array([[x_min, y_min], [x_max, y_max]])

    print('calc_smaller_stone_boundaries')
    print('old:', invalid_stone[1] - invalid_stone[0])
    print('new:', new[1] - new[0])

    return new


def pick_smaller_stone(stones, new_lim):

    l, w = new_lim[1] - new_lim[0]

    target_l = l
    target_w = w
    rotation = None

    if w > l:
        rotation = RZ_90
        target_l = w
        target_w = l
    match = next(s for s in stones if s.length < target_l and s.width < target_w)
    print(match.length, match.width)

    # determine the rotation for l, w, h
    # - eigenvectors -> [[0, 1, 0][1, 0, 0][0, 0, 1]] for switching x/y axis --> not same as RZ90
    # while not in correct order: if z>y -> rotate x elif y>x -> rotate z

    return stones.index(match), rotation


invalid = np.array([[0, 0], [2, 1]])
intersect = [
    np.array([[0, 0], [1.2, .1]]),  # lower left corner -> x_min, y_min
    np.array([[0.9, 0.8], [2, 1]]),  # upper right corner -> x_max, y_max
    np.array([[-.1, .9], [2.1, 1]]),  # upper band
    np.array([[.5, .3], [2.1, .5]]),  # y_min and y_max within the stone
    np.array([[.55, .35], [1.85, .55]]),  # completely within
]

stones = [generate_regular_stone((.1, .9), (.1, .9), (.1, .9)) for _ in range(5)]
intersect = list()
for stone in stones:
    stone.transform(Translation(np.array([random.uniform(-.4, 2.4), random.uniform(-.3, 1.3), 0])))
    intersect.append(stone.aabb_limits[:, :2])
print(intersect)


fig, axs = plt.subplots(1, len(intersect)+1, figsize=(15, 4))
axs[-1].add_patch(Rectangle(invalid[0, :2], invalid[1, 0] - invalid[0, 0], invalid[1, 1] - invalid[0, 1], color='green'))
for i in intersect:
    axs[-1].add_patch(Rectangle(i[0, :2], i[1, 0] - i[0, 0], i[1, 1] - i[0, 1], color='red'))

axs[-1].axis('equal')
axs[-1].set_xlim(-.1, 2.1)
axs[-1].set_ylim(-1.1, 1.1)
new = calc_smaller_stone_boundaries(invalid, intersect)

# fetch a smaller stone
smaller_stones = [generate_regular_stone((new[0, 0] - .1, new[1, 0] + .1), (new[0, 1] - .1, new[1, 1] + .1), (.1, .1)) for _ in range(50)]
smaller_stones.sort(key=lambda x: x.aabb_volume, reverse=True)
index, rot = pick_smaller_stone(smaller_stones, new)
if np.any(rot):
    print('rotate the smaller stone!')

plt.show()
