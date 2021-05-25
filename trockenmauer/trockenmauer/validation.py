# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 22.05.21, validation.py
#

from typing import TYPE_CHECKING, Tuple, Union, List
from dataclasses import dataclass

import numpy as np
import pymesh
from aabbtree import AABB

from .stone import Intersection
from .math_utils import Translation

if TYPE_CHECKING:
    from .stone import Stone, Boundary, Wall
    from aabbtree import AABBTree


class Validator:
    """
    Validation functions for a placement

    validate
    - stone is within the boundary -> minimize intersection volume
    - stone does not intersect with another -> minimize intersection volume
    - distance to the boundary -> minimize distance
    - normal of the ground area at the placement is in the same direction as the stones bottom side -> rotation
    - ...

    optimization
    - the closer to the boundary the better
    - the closer to other stones the better (rotation, that the stones are aligned on on of their face?)
    - less open space
    - ...
    """

    def __init__(self, intersection_boundary=False, intersection_stones=False,
                 distance2boundary=False, volume_below_stone=False):
        """
        The parameters control, witch validations will be executed

        :param intersection_boundary: validate intersections with the boundary
        :param intersection_stones: validate intersections with the previously placed stones
        """
        self.intersection_boundary = intersection_boundary
        self.intersection_stones = intersection_stones
        self.distance2boundary = distance2boundary
        self.volume_below_stone = volume_below_stone

        # initialize tetgen for tetrahedralization
        self.tetgen = pymesh.tetgen()
        self.tetgen.max_tet_volume = 2
        self.tetgen.verbosity = 0  # no output
        self.tetgen.split_boundary = False

    @staticmethod
    def _bb_intersections(tree: 'AABBTree', bb: 'AABB') -> List['Intersection']:
        # for i, box in zip(tree.overlap_values(bb), tree.overlap_aabbs(bb)):
        #     print(i, bb.overlap_volume(box))

        # return for each intersection the index in walls.stone (name in tree) and the bb (+ volume)
        return [Intersection(bb=bb.overlap_aabb(box), name=i)
                for i, box in zip(tree.overlap_values(bb), tree.overlap_aabbs(bb))]
        # slow? overlap is calculated 2x, maybe only overlap_values and use the index in the list of stones
        # wall instead of tree would be needed

    @staticmethod
    def _mesh_intersection(mesh1: 'pymesh.Mesh', mesh2: 'pymesh.Mesh') -> Union[None, 'pymesh.Mesh']:
        # engine: igl (auto) fails on empty intersection, cgal requires closed meshes
        # cork always works, but is more experimental
        try:
            intersection = pymesh.boolean(mesh1, mesh2, "intersection", engine='cgal')
        except RuntimeError as err:
            # print(err)
            intersection = None

        return intersection

    def _intersection_boundary(self, stone: 'Stone', wall: 'Wall'
                               ) -> Tuple[bool, Union[None, 'Intersection']]:
        """
        Intersects a stone with the boundary.

        :param stone: Stone object
        :param wall: Boundary object
        :return: Status: True (there is an intersection) / False (no intersection; intersection mesh
        """
        intersection = self._mesh_intersection(stone.mesh, wall.boundary.mesh_solid_sides)
        if intersection and np.any(intersection.vertices):
            # run tetrahedalization
            self.tetgen.points = intersection.vertices
            self.tetgen.triangles = intersection.faces
            self.tetgen.run()
            intersection = self.tetgen.mesh
            intersection = Intersection(mesh=intersection)
            # intersection = pymesh.tetrahedralize(intersection, 20, engine='tetgen')
            # intersection = load_from_pymesh('intersection', intersection, 'boundary intersection')
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

    @staticmethod
    def _distance2mesh(points: List[np.ndarray], mesh: 'pymesh.Mesh') -> List:
        # for each point: [squared distance, closest face, closest point]
        distances, faces, closest_points = pymesh.distance_to_mesh(mesh, points, engine='cgal')
        return distances

    def _min_distance2boundary(self, stone: 'Stone', boundary: 'Boundary'):
        distances = self._distance2mesh(stone.sides_center, boundary.mesh_solid_sides)
        return np.sqrt(np.min(distances))

    def _volume_below_stone(self, stone: 'Stone', wall: 'Wall'):
        # Approximation with bounding box -> with mesh, it is harder to have the outer hull footprint
        # print(stone.aabb_limits)
        # bounding box of the volume below the stone
        bb = AABB(np.vstack((stone.aabb[:2, :], [0, stone.aabb[2, 0]])))

        # subtract the volume of intersection stones below
        inter = self._bb_intersections(wall.tree, bb)
        vol_inter = np.sum([i.aabb.volume for i in inter])

        return bb.volume - vol_inter

    def validate(self, stone: 'Stone', wall: 'Wall') -> Tuple[bool, 'ValidationResult']:
        """
        Validates the new stone to the built wall. All validation functions are used according to the
        attributes set in the validator initialisation.
        If the validation is passed, it returns True and en empty ValidationError-object.
        If the validation fails, it returns False and the details in the ValidationError-object

        :param stone:
        :param wall:
        :return: passed, results
        """
        passed = True
        results = ValidationResult()

        if self.intersection_boundary:
            intersects, details = self._intersection_boundary(stone, wall)
            if intersects:
                passed = False
                results.intersection_boundary = details

        if self.intersection_stones:
            intersects, details = self._intersection_stones(stone, wall)
            if intersects:
                passed = False
                results.intersection_stones = details

        if self.distance2boundary:
            d = self._min_distance2boundary(stone, wall.boundary)
            results.distance2boundary = d

        if self.volume_below_stone:
            v = self._volume_below_stone(stone, wall)
            results.volume_below_stone = v

        return passed, results

    def fitness(self, firefly: 'np.ndarray', stone: 'Stone', wall: 'Wall', **kwargs):
        """
        Calculates the fitness of a placement for a stone.

        :param firefly: Firefly with the three coordinates as genes
        :param stone:
        :param wall:
        :return:
        """
        # move the bottom center of the stone to the given position
        t = Translation(translation=firefly - stone.bottom_center)
        stone.transform(transformation=t)
        passed, res = self.validate(stone, wall)

        # balance the separate terms:
        # - volume: fraction of the volume of the stone [0-1]
        # - distance2boundary: wall width: 0.5 -> max_distance ~ 0.2 -> d*5 -> [0-1]
        # - volume below the stone
        score = 0
        if self.intersection_boundary or self.intersection_stones:
            score += 1 * res.intersection_volume / stone.mesh_volume
        if self.distance2boundary:
            score += 5*res.distance2boundary
        if self.volume_below_stone:
            score += res.volume_below_stone / stone.mesh_volume
        return score


@dataclass
class ValidationResult:
    """
    Storing all validation results. The total intersection volume gets automatically updated,
    if a boundary intersection or stones intersection is set.
    """
    _intersection_boundary: 'Intersection' = False
    _intersection_stones: List['Intersection'] = False
    intersection_volume: float = 0
    distance2boundary: float = None
    volume_below_stone: float = None

    @property
    def intersection_boundary(self) -> 'Intersection':
        return self._intersection_boundary

    @property
    def intersection_stones(self) -> List['Intersection']:
        return self._intersection_stones

    @intersection_boundary.setter
    def intersection_boundary(self, new_intersection: 'Intersection'):
        # print('set boundary volume')
        self._intersection_boundary = new_intersection
        self.update_total_volume(new_intersection)

    @intersection_stones.setter
    def intersection_stones(self, new_intersections: List['Intersection']):
        # print('set intersection volume: old vol', self.intersection_volume)
        self._intersection_stones = new_intersections
        for i in new_intersections:
            self.update_total_volume(i)

    def update_total_volume(self, new_intersection: 'Intersection'):
        if new_intersection.mesh_volume:
            self.intersection_volume += new_intersection.mesh_volume
            # print('mesh volume updated')
        elif new_intersection.aabb_volume:
            self.intersection_volume += new_intersection.aabb_volume
            # print('aabb volume updated')
        else:
            print('intersection but no volume added')
