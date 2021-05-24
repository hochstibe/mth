# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 22.05.21, validation.py
#

from typing import TYPE_CHECKING, Tuple, Union, List
from dataclasses import dataclass

import numpy as np
import pymesh

from .utils import load_from_pymesh
from .stone import Intersection

if TYPE_CHECKING:
    from .stone import Geometry, Stone, Wall
    from aabbtree import AABB, AABBTree


class Validator:
    """
    Validation functions for a placement

    validate
    - stone is within the boundary -> intersection
    - stone does not intersect with another -> intersection
    - normal of the ground area at the placement is in the same direction as the stones bottom side -> rotation
    - ...

    optimization
    - the closer to the boundary the better
    - the closer to other stones the better (rotation, that the stones are aligned on on of their face?)
    - less open space
    - ...
    """

    def __init__(self, intersection_boundary=False, intersection_stones=False):
        """
        The parameters control, witch validations will be executed

        :param intersection_boundary: validate intersections with the boundary
        :param intersection_stones: validate intersections with the previously placed stones
        """
        self.intersection_boundary = intersection_boundary
        self.intersection_stones = intersection_stones

    @staticmethod
    def _bb_intersections(tree: 'AABBTree', bb: 'AABB') -> List['Intersection']:
        # for i, box in zip(tree.overlap_values(bb), tree.overlap_aabbs(bb)):
        #     print(i, bb.overlap_volume(box))

        # return for each intersection the index in walls.stone (name in tree) and the volume
        return [Intersection(bb_volume=bb.overlap_volume(box), name=i)
                for i, box in zip(tree.overlap_values(bb), tree.overlap_aabbs(bb))]
        # slow? overlap is calculated 2x, maybe only overlap_values and use the index in the list of stones
        # wall instead of tree would be needed

    @staticmethod
    def _mesh_intersection(mesh1: 'pymesh.Mesh', mesh2: 'pymesh.Mesh') -> 'pymesh.Mesh':
        # engine: igl (auto) fails on empty intersection, cgal requires closed meshes
        # cork always works, but is more experimental
        intersection = pymesh.boolean(mesh1, mesh2, "intersection", engine='cgal')

        return intersection

    def _intersection_boundary(self, stone: 'Stone', wall: 'Wall'
                               ) -> Tuple[bool, Union[None, 'Geometry']]:
        """
        Intersects a stone with the boundary.

        :param stone: Stone object
        :param wall: Boundary object
        :return: Status: True (there is an intersection) / False (no intersection; intersection mesh
        """
        intersection = self._mesh_intersection(stone.mesh, wall.boundary.mesh_solid)
        if np.any(intersection.vertices):
            intersection = load_from_pymesh('geometry', intersection, 'boundary intersection')
            return True, intersection
        else:
            return False, None

    def _intersection_stones(self, stone: 'Stone', wall: 'Wall'
                             ) -> Tuple[bool, Union[None, List['Intersection']]]:
        """
        Intersects a stone with the previously placed stones

        :param stone: Stone object
        :param wall: Wall object with the mesh of all previously placed stones
        :return:
        """
        # Todo: at the moment: only intersect the bounding boxes
        if not wall.mesh:
            # No stone was placed yet
            return False, None

        # calculate bb intersections
        intersections = self._bb_intersections(tree=wall.tree, bb=stone.aabb)

        # calculate mesh intersections for all bb-intersections
        # intersections = [self._mesh_intersection(stone.mesh, wall_stone.mesh) for wall_stone in wall.stones]
        # intersections = [load_from_pymesh('geometry', i, 'stone intersection') for i in intersections if np.any(i.vertices)]  # remove empty intersections
        if intersections:
            return True, intersections
        else:
            return False, None
        # intersection = self._mesh_intersection(stone.mesh, wall.mesh)
        # if np.any(intersection.vertices):
        #     intersection = load_from_pymesh('geometry', intersection, 'stone intersection')
        #     return True, intersection
        # else:
        #     return False, None

    def _validate_normals(self, stone: 'Stone', wall: 'Wall'):
        """
        The normal of the stone should point in the same direction as the normal of the wall on the given placement
        """
        pass

    def validate(self, stone: 'Stone', wall: 'Wall') -> Tuple[bool, 'ValidationError']:
        """
        Validates the new stone to the built wall. All validation functions are used according to the
        attributes set in the validator initialisation.
        If the validation is passed, it returns True and en empty ValidationError-object.
        If the validation fails, it returns False and the details in the ValidationError-object

        :param stone:
        :param wall:
        :return: passed, errors
        """
        passed = True
        errors = ValidationError()

        if self.intersection_boundary:
            intersects, details = self._intersection_boundary(stone, wall)
            if intersects:
                passed = False
                errors.intersection_boundary = details

        if self.intersection_stones:
            intersects, details = self._intersection_stones(stone, wall)
            if intersects:
                passed = False
                errors.intersection_stones = details

        return passed, errors


@dataclass
class ValidationError:
    intersection_boundary: 'Geometry' = False
    intersection_stones: List['Intersection'] = False
