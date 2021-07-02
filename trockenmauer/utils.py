# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, utils.py
#

from typing import TYPE_CHECKING, List, Tuple, Optional

import numpy as np

from .math_utils import RZ_90, aabb_overlap

if TYPE_CHECKING:
    from . import Stone, Intersection


def calc_smaller_stone_boundaries(invalid_stone: 'Stone',
                                  intersection_boundary: 'Intersection', intersection_stones: List['Intersection'],
                                  reduce_z: Optional[float] = None) -> 'np.ndarray':
    """
    Calculate the max_bb for a stone and its intersections (in x and y direction)
    Greedy calculation, reduces for each intersection -> no global calculation.
    If z_max is reduced by reduce_z
    """
    x_min, y_min, z_min, x_max, y_max, z_max = invalid_stone.aabb_limits.flatten()
    if intersection_boundary:
        x_min, y_min, z_min, x_max, y_max, z_max = intersection_boundary.valid_stone_aabb.flatten()

    for n, i in enumerate(intersection_stones):
        ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = i.aabb_limits.flatten()

        hit = aabb_overlap(np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]]), i.aabb_limits)

        if np.any(hit):
            # List the possible reductions of the aabb limits
            reductions = list()  # [[name, amount, area after reduction], ...]

            # x_max
            if x_min < ix_min < x_max:
                reductions.append(['r_x_max', x_max - ix_min, (ix_min - x_min) * (y_max - y_min)])
            # y_max
            if y_min < iy_min < y_max:
                reductions.append(['r_y_max', y_max - iy_min, (iy_min - y_min) * (x_max - x_min)])
            # x_min
            if x_max > ix_max > x_min:
                reductions.append(['r_x_min', ix_max - x_min, (x_max - ix_max) * (y_max - y_min)])
            # y_min
            if y_max > iy_max > y_min:
                reductions.append(['r_y_min', iy_max - y_min, (y_max - iy_max) * (x_max - x_min)])

            red = reductions[0]
            for r in reductions:
                if r[2] > red[2]:
                    red = r

            if red[0] == 'r_x_min':
                x_min += red[1]
            elif red[0] == 'r_y_min':
                y_min += red[1]
            elif red[0] == 'r_x_max':
                x_max -= red[1]
            elif red[0] == 'r_y_max':
                y_max -= red[1]

        else:
            pass

    if reduce_z:
        z_max -= reduce_z

    new = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])

    return new


def pick_smaller_stone2(stones: List['Stone'], new_lim: 'np.ndarray') -> Tuple[Optional[int], 'np.ndarray', 'np.ndarray']:
    # return the index to the list of stones for the smaller stone, the initial position and the rotation
    # stones are ordered by their volume or area --> the first match is good
    x, y, z = new_lim[1] - new_lim[0]

    target_l = x
    target_w = y
    target_h = z
    position = np.mean(new_lim, axis=0)
    rotation = None

    if y > x:
        rotation = RZ_90
        target_l = y
        target_w = x

    try:
        match = next(s for s in stones if s.length <= target_l and s.width <= target_w and s.height <= target_h)
        print(match.length, match.width)
    except StopIteration:
        print('no smaller stone in the list')
        return None, position, rotation

    return stones.index(match), position, rotation
