# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 22.05.21, validation.py
#

from typing import TYPE_CHECKING, Tuple, Union, List
from dataclasses import dataclass

import numpy as np
import pymesh

from .stone import Intersection
from .math_utils import Translation

if TYPE_CHECKING:
    from .stone import Stone, Boundary
    from trockenmauer.wall import Wall


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

    # initialize tetgen for tetrahedralization
    tetgen = pymesh.tetgen()
    tetgen.max_tet_volume = 2
    tetgen.verbosity = 0  # no output
    tetgen.split_boundary = False

    @staticmethod
    def _bb_intersections(wall: 'Wall', bb: 'np.ndarray') -> List['Intersection']:

        hits = wall.r_tree.intersection(bb.flatten(), objects=False)  # -> list of indices in index

        intersections = list()
        for i in hits:
            limits = wall.stones[i].aabb_overlap(bb)
            if np.any(limits):  # rtree returns a hit for adjacent boxes
                intersections.append(Intersection(bb_limits=limits))

        return intersections

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
        if not wall.stones:
            # No stone was placed yet
            return False, None

        # calculate bb intersections
        intersections = self._bb_intersections(wall, stone.aabb_limits)

        if intersections:
            return True, intersections
        else:
            return False, None

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
        # bounding box of the volume below the stone
        # [[1, 2, 3], [4, 5, 6]] -> [[1, 2, 0], [4, 5, 3]]
        bb = np.hstack((stone.aabb_limits[:, :2], [[0], [stone.aabb_limits[0, 2]]]))
        bb_vol = np.prod(bb[1] - bb[0])

        # subtract the volume of intersection stones below
        intersections = self._bb_intersections(wall, bb)
        vol_inter = np.sum(i.aabb_volume for i in intersections)

        return bb_vol - vol_inter

    @staticmethod
    def _closest_stone(stone: 'Stone', wall: 'Wall'):

        hits = wall.r_tree.nearest(stone.aabb_limits.flatten(), num_results=5, objects=True)

        shortest_distances = list()
        for h in hits:
            # upper corner of the hit has to be higher than the stones lower corner
            if h.bbox[-1] > stone.aabb_limits[0, 2]:  # only use stones on the same level or higher
                overlap_min = np.max(np.array([stone.aabb_limits[0], h.bbox[:3]]), axis=0)
                overlap_max = np.min(np.array([stone.aabb_limits[1], h.bbox[3:]]), axis=0)
                diff = overlap_max - overlap_min
                d = np.linalg.norm(diff[diff <= 0])
                # negative --> in this direction is the shortest distance
                shortest_distances.append(d)

        if shortest_distances:
            return shortest_distances[0]
        else:
            return 0.

    def validate(self, stone: 'Stone', wall: 'Wall') -> Tuple[bool, 'ValidationResult']:
        """
        Dummy function, replaced in subclasses.
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

        return passed, results

    def fitness(self, firefly: 'np.ndarray', stone: 'Stone', wall: 'Wall', **kwargs) -> float:
        """
        Dummy function, replaced in subclasses.
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

        if passed:
            score = 0
        else:
            score = 1
        return score


class ValidatorNormal(Validator):

    def __init__(self, intersection_boundary=False, intersection_stones=False,
                 distance2boundary=False, volume_below_stone=False, distance2closest_stone=False):
        """
        The parameters control, witch validations will be executed

        :param intersection_boundary: validate intersections with the boundary
        :param intersection_stones: validate intersections with the previously placed stones
        """
        # Init tetgen from super()
        super().__init__()

        self.intersection_boundary = intersection_boundary
        self.intersection_stones = intersection_stones
        self.distance2boundary = distance2boundary
        self.volume_below_stone = volume_below_stone
        self.distance2closest_stone = distance2closest_stone

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

        if self.distance2closest_stone:
            results.distance2closest_stone = self._closest_stone(stone, wall)

        return passed, results

    def fitness(self, firefly: 'np.ndarray', stone: 'Stone', wall: 'Wall', **kwargs) -> float:
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
        # intersection volume
        # if res.intersection:  # penalty any intersection with 1 and add 10* scaled volume
        #     score += 1 + res.intersection_volume / stone.mesh_volume
        if res.intersection_boundary:
            score += .5 + res.intersection_volume_b / stone.mesh_volume
        if res.intersection_stones:
            score += 1 + res.intersection_volume_s / stone.mesh_volume
        # Distance to boundary
        if self.distance2boundary:
            score += (1 + 5*res.distance2boundary)**2 - 1
        # free volume below the stone
        if self.volume_below_stone:
            score += res.volume_below_stone / stone.mesh_volume
        # distance to the closest stone on the same level
        if self.distance2closest_stone:
            # no penalty, if there is no stone on the same level
            score += 5 * res.distance2closest_stone

        # Punish stones that are not on the current level
        if not wall.in_current_level(stone):
            score += 2 + stone.aabb_limits[1][2]
        return score


class ValidatorFill(Validator):

    def __init__(self):
        super().__init__()

    def validate(self, stone: 'Stone', wall: 'Wall') -> Tuple[bool, 'ValidationResult']:

        passed = True
        results = ValidationResult()

        return passed, results

    def fitness(self, firefly: 'np.ndarray', stone: 'Stone', wall: 'Wall', **kwargs) -> float:

        # move the bottom center of the stone to the given position
        t = Translation(translation=firefly - stone.bottom_center)
        stone.transform(transformation=t)
        passed, res = self.validate(stone, wall)
        
        if passed:
            return 0
        else:
            return 1


@dataclass
class ValidationResult:
    """
    Storing all validation results. The total intersection volume gets automatically updated,
    if a boundary intersection or stones intersection is set.
    """
    __intersection_boundary: 'Intersection' = False  # Intersection with the boundary
    __intersection_stones: List['Intersection'] = False  # All Intersections with other stones
    _intersection: bool = False  # whether there is any intersection or not
    _intersection_volume: float = 0  # total intersection volume
    _intersection_volume_b: float = 0
    _intersection_volume_s: float = 0
    distance2boundary: float = 0  # minimal distance to the boundary
    volume_below_stone: float = 0  # total free volume below the stone
    distance2closest_stone: float = 0

    @property
    def intersection_volume_b(self):
        return self._intersection_volume_b

    @property
    def intersection_volume_s(self):
        return self._intersection_volume_b

    @property
    def intersection_boundary(self) -> 'Intersection':
        return self.__intersection_boundary

    @property
    def intersection_stones(self) -> List['Intersection']:
        return self.__intersection_stones

    @property
    def intersection(self) -> bool:
        return self._intersection

    @property
    def intersection_volume(self) -> float:
        return self._intersection_volume

    @intersection_boundary.setter
    def intersection_boundary(self, new_intersection: 'Intersection'):
        self.__intersection_boundary = new_intersection
        self._intersection = True
        self.update_total_volume(new_intersection)
        self._intersection_volume_b = new_intersection.mesh_volume

    @intersection_stones.setter
    def intersection_stones(self, new_intersections: List['Intersection']):
        self.__intersection_stones = new_intersections
        self._intersection = True
        map(self.update_total_volume, new_intersections)
        self._intersection_volume_s += np.sum([i.aabb_volume for i in new_intersections])

    def update_total_volume(self, new_intersection: 'Intersection'):
        # update the volume: try first the exact mesh volume. if not available, use aabb volume
        if new_intersection.mesh_volume:
            self._intersection_volume += new_intersection.mesh_volume
        elif new_intersection.aabb_volume:
            self._intersection_volume += new_intersection.aabb_volume
        else:
            print('intersection but no volume added')
