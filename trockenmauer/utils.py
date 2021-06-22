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
                                  intersection_boundary: 'Intersection', intersection_stones: List['Intersection']):
    """
    Calculate the max_bb for a stone and its intersections (in x and y direction)
    Greedy calculation, reduces for each intersection -> no global calculation
    """
    x_min, y_min, z_min, x_max, y_max, z_max = invalid_stone.aabb_limits.flatten()
    if intersection_boundary:
        # hit = aabb_overlap(invalid_stone.aabb_limits, intersection_boundary.aabb_limits)
        # if np.any(hit):
        #     x_min, y_min, z_min, x_max, y_max, z_max = hit.flatten()
        #     print('  this boundary still intersects')
        x_min, y_min, z_min, x_max, y_max, z_max = intersection_boundary.valid_stone_aabb.flatten()
        print('boundary intersection: valid part of the stone:')
        print(intersection_boundary.valid_stone_aabb)

    print(len(intersection_stones), 'intersection stones')
    for n, i in enumerate(intersection_stones):
        print('intersection', n)
        ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = i.aabb_limits.flatten()

        hit = aabb_overlap(np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]]), i.aabb_limits)

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

        else:
            print('  no intersection anymore')

    new = np.array([[x_min, y_min], [x_max, y_max]])

    print('calc_smaller_stone_boundaries')
    print('old:', invalid_stone.aabb_limits[1] - invalid_stone.aabb_limits[0])
    print(invalid_stone.aabb_limits)
    print('new:', new[1] - new[0])
    print(new)

    return new


def pick_smaller_stone2(stones: List['Stone'], new_lim: 'np.ndarray') -> Tuple[Optional[int], 'np.ndarray', 'np.ndarray']:
    # return the index to the list of stones for the smaller stone, the initial position and the rotation
    # stones are ordered by their volume or area --> the first match is good
    x, y = new_lim[1] - new_lim[0]

    target_l = x
    target_w = y
    position = np.mean(new_lim, axis=0)
    rotation = None

    if y > x:
        rotation = RZ_90
        target_l = y
        target_w = x

    try:
        match = next(s for s in stones if s.length <= target_l and s.width <= target_w)
        print(match.length, match.width)
    except StopIteration:
        print('no smaller stone in the list')
        return None, position, rotation

    # determine the rotation for l, w, h
    # - eigenvectors -> [[0, 1, 0][1, 0, 0][0, 0, 1]] for switching x/y axis --> not same as RZ90
    # -> https://math.stackexchange.com/questions/1896124/align-rectangle-to-coordinate-frame-x-and-y-axes-3d
    # - while not in correct order: if z>y -> rotate x elif y>x -> rotate z
    # - order of xyz should be 210 -> switch numbers = rotation around other axis

    return stones.index(match), position, rotation


# def pick_smaller_stone(invalid_stone: 'Stone', stones: List['Stone'],
#                        overlapping_area: float, keep: str = 'width') -> Tuple[Optional[int], 'np.ndarray']:
#     # return the index to the list of stones for the smaller stone
#     # stones are ordered by their volume or area --> the first match is good
#
#     rotate = False
#     if keep == 'width':
#         # same width -> reduce length
#         overlap_l = overlapping_area / invalid_stone.width
#         target_l = invalid_stone.length - overlap_l
#         target_w = invalid_stone.width
#         if target_l < target_w:
#             print('rotate target stone')
#             rotate = True
#             target_w = target_l
#             target_l = invalid_stone.width
#     elif keep == 'length':
#         overlap_w = overlapping_area / invalid_stone.length
#         target_l = invalid_stone.length
#         target_w = invalid_stone.width - overlap_w
#         if target_l < target_w:
#             print('rotate target stone')
#             rotate = True
#             target_l = target_w
#             target_w = invalid_stone.length
#     else:
#         raise ValueError(f"'keep' must be one of 'width' or 'length' ('{keep}' given.")
#
#     try:
#         match = next(s for s in stones if s.length < target_l and s.width < target_w)
#     except StopIteration:
#         print('no smaller stone in the list')
#         return None, invalid_stone.current_rotation
#     # Todo: rotate, if not initial fireflies try both
#     if rotate:
#         # rotation matrix from origin position to the target rotation (given: relative to the big stone)
#         rotation = RZ_90 @ invalid_stone.current_rotation
#     else:
#         rotation = invalid_stone.current_rotation
#
#     return stones.index(match), rotation


# def load_from_pymesh(geom_type: str, mesh: 'Mesh', name: str = None
#                      ) -> Union['Geometry', 'Stone', 'Boundary']:
#     """
#     Load a geometry from a pymesh object
#
#     :param geom_type: 'geometry', 'stone', 'boundary', 'intersection'
#     :param mesh: pymesh object with vertices and faces
#     :param name: optional name of the stone
#     :return: Stone
#     """
#
#     if geom_type.lower() == 'geometry':
#         return Geometry(mesh, name)
#     elif geom_type.lower() == 'stone':
#         return Stone(mesh.vertices, mesh.faces, name)
#     elif geom_type.lower() == 'boundary':
#         return Boundary(mesh.vertices, mesh.faces, name)
#     elif geom_type.lower() == 'intersection':
#         return Intersection(mesh, name)
#     else:
#         raise ValueError("Type must be one of 'geometry', 'stone', 'boundary', 'intersection'")
