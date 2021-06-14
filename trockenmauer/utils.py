# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, utils.py
#

from typing import TYPE_CHECKING, Union, List, Tuple, Optional

import numpy as np

from .stone import Geometry, Stone, Boundary, Intersection
from .math_utils import RZ_90

if TYPE_CHECKING:
    from pymesh import Mesh


def pick_smaller_stone(invalid_stone: 'Stone', stones: List['Stone'],
                       overlapping_area: float, keep: str = 'width') -> Tuple[Optional[int], 'np.ndarray']:
    # return the index to the list of stones for the smaller stone
    # stones are ordered by their volume or area --> the first match is good

    rotate = False
    if keep == 'width':
        # same width -> reduce length
        overlap_l = overlapping_area / invalid_stone.width
        target_l = invalid_stone.length - overlap_l
        target_w = invalid_stone.width
        if target_l < target_w:
            print('rotate target stone')
            rotate = True
            target_w = target_l
            target_l = invalid_stone.width
    elif keep == 'length':
        overlap_w = overlapping_area / invalid_stone.length
        target_l = invalid_stone.length
        target_w = invalid_stone.width - overlap_w
        if target_l < target_w:
            print('rotate target stone')
            rotate = True
            target_l = target_w
            target_w = invalid_stone.length
    else:
        raise ValueError(f"'keep' must be one of 'width' or 'length' ('{keep}' given.")

    try:
        match = next(s for s in stones if s.length < target_l and s.width < target_w)
    except StopIteration:
        print('no smaller stone in the list')
        return None, invalid_stone.current_rotation
    # Todo: rotate, if not initial fireflies try both
    if rotate:
        # rotation matrix from origin position to the target rotation (given: relative to the big stone)
        rotation = RZ_90 @ invalid_stone.current_rotation
    else:
        rotation = invalid_stone.current_rotation

    return stones.index(match), rotation


def load_from_pymesh(geom_type: str, mesh: 'Mesh', name: str = None
                     ) -> Union['Geometry', 'Stone', 'Boundary']:
    """
    Load a geometry from a pymesh object

    :param geom_type: 'geometry', 'stone', 'boundary', 'intersection'
    :param mesh: pymesh object with vertices and faces
    :param name: optional name of the stone
    :return: Stone
    """

    if geom_type.lower() == 'geometry':
        return Geometry(mesh, name)
    elif geom_type.lower() == 'stone':
        return Stone(mesh.vertices, mesh.faces, name)
    elif geom_type.lower() == 'boundary':
        return Boundary(mesh.vertices, mesh.faces, name)
    elif geom_type.lower() == 'intersection':
        return Intersection(mesh, name)
    else:
        raise ValueError("Type must be one of 'geometry', 'stone', 'boundary', 'intersection'")
