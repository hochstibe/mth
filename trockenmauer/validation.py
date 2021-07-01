# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 22.05.21, validation.py
#

from typing import TYPE_CHECKING, Tuple, Union, List, Optional
from dataclasses import dataclass

import numpy as np
import pymesh

from .stone import Intersection, NormalStone
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
        """
        Intersects a bounding box ((2, 3) with the bounding boxes within the tree

        :param wall: Boundary object
        :param bb: Bounding box to intersect
        :return: Status: True (there is an intersection) / False (no intersection; intersection mesh
        """

        hits = wall.r_tree.intersection(bb.flatten(), objects=False)  # -> list of indices in index

        intersections = list()
        for i in hits:
            limits = wall.stones[i].aabb_overlap(bb)
            if np.any(limits):  # rtree returns a hit for adjacent boxes
                intersections.append(Intersection(bb_limits=limits))

        return intersections

    @staticmethod
    def _mesh_intersection(mesh1: 'pymesh.Mesh', mesh2: 'pymesh.Mesh'
                           ) -> Tuple[Union[None, 'pymesh.Mesh'], Union[None, 'pymesh.Mesh']]:
        """
        Intersects a mesh with another mesh

        :param mesh1: first mesh
        :param mesh2: second mesh
        :return: intersecting mesh + valid mesh or None, None
        """
        # engine: igl (auto) fails on empty intersection, cgal requires closed meshes
        # cork always works, but is more experimental
        try:
            intersection = pymesh.boolean(mesh1, mesh2, "intersection", engine='cgal')
            valid = pymesh.boolean(mesh1, mesh2, "difference", engine='cgal')
        except RuntimeError:
            intersection, valid = None, None

        return intersection, valid

    def _intersection_boundary(self, stone: 'Stone', wall: 'Wall'
                               ) -> Tuple[bool, Optional['Intersection']]:
        """
        Intersects a stone with the boundary.

        :param stone: Stone object
        :param wall: Boundary object
        :return: Status: True (there is an intersection) / False (no intersection; intersection mesh
        """
        intersection, valid_stone = self._mesh_intersection(stone.mesh, wall.boundary.mesh_solid_sides)
        if intersection and np.any(intersection.vertices):
            # run tetrahedalization
            self.tetgen.points = intersection.vertices
            self.tetgen.triangles = intersection.faces
            self.tetgen.run()
            intersection = self.tetgen.mesh

            valid = np.array([np.min(valid_stone.vertices, axis=0), np.max(valid_stone.vertices, axis=0)])
            intersection = Intersection(mesh=intersection, valid_stone_aabb=valid)
            return True, intersection

        else:
            return False, None

    def _intersection_stones(self, stone: 'Stone', wall: 'Wall'
                             ) -> Tuple[bool, Union[None, List['Intersection']]]:
        """
        Intersects a stone with the previously placed stones

        :param stone: Stone object
        :param wall: Wall object with the mesh of all previously placed stones
        :return: Status, List of intersection objects
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
    def _distance2mesh(points: np.ndarray, mesh: 'pymesh.Mesh') -> List:
        """
        Calculates the shortest distances from a set of points to a mesh
        
        :param points: Set of points (n, 3)
        :param mesh: mesh
        :return: list of squared distances
        """
        # for each point: [squared distance, closest face, closest point]
        distances, faces, closest_points = pymesh.distance_to_mesh(mesh, points, engine='cgal')
        return distances

    def _min_distance2boundary(self, stone: 'Stone', boundary: 'Boundary') -> float:
        """
        Calculates the minimal distance to the boundary

        :param stone: stone object
        :param boundary: boundary object
        :return: minimal distance
        """
        distances = self._distance2mesh(stone.sides_center, boundary.mesh_solid_sides)
        return np.sqrt(np.min(distances))

    def _volume_below_stone(self, stone: 'Stone', wall: 'Wall') -> float:
        """
        Calculates the empty volume below the stone

        :param stone: stone object
        :param wall: wall objects with the already placed stones
        :return: volume
        """
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
    def _closest_stone(stone: 'Stone', wall: 'Wall', n: int = 1) -> List['Stone']:
        """
        Calculates the n closest stones on the same height or higher
        (other stone z_max is higher than new stone's z_min)

        :param stone: new stone object
        :param wall: wall objects with the already placed stones
        :return: list with stones
        """
        hits = wall.r_tree.nearest(stone.aabb_limits.flatten(), num_results=np.min([2*n, 5]), objects=False)

        # upper corner of the hit has to be higher than the stones lower corner
        # only use stones on the same level or higher
        shortest_stones = [wall.stones[i] for i in hits if wall.stones[i].aabb_limits[1, 2] > stone.aabb_limits[0, 2]]

        return shortest_stones

    def _distance2closest_stone(self, stone: 'Stone', wall: 'Wall', n: int = 1) -> List[float]:
        """
        Calculates the distances to n closest stones on the same height
        (other stone z_max is higher than new stone's z_min)

        :param stone: new stone object
        :param wall: wall objects with the already placed stones
        :return: list with distances
        """
        # get the list of distances to the closest stones for num stones
        shortest_stones = self._closest_stone(stone, wall, n)

        shortest_distances = []
        for hit in shortest_stones:
            overlap_min = np.max(np.array([stone.aabb_limits[0], hit.aabb_limits[0]]), axis=0)
            overlap_max = np.min(np.array([stone.aabb_limits[1], hit.aabb_limits[1]]), axis=0)
            diff = overlap_max - overlap_min
            d = np.linalg.norm(diff[diff <= 0])
            # negative --> in this direction is the shortest distance
            shortest_distances.append(d)

        # hits = wall.r_tree.nearest(stone.aabb_limits.flatten(), num_results=4*n, objects=True)
        # shortest_distances = list()
        # for h in hits:
        #     # upper corner of the hit has to be higher than the stones lower corner
        #     if h.bbox[-1] > stone.aabb_limits[0, 2]:  # only use stones on the same level or higher
        #         overlap_min = np.max(np.array([stone.aabb_limits[0], h.bbox[:3]]), axis=0)
        #         overlap_max = np.min(np.array([stone.aabb_limits[1], h.bbox[3:]]), axis=0)
        #         diff = overlap_max - overlap_min
        #         d = np.linalg.norm(diff[diff <= 0])
        #         # negative --> in this direction is the shortest distance
        #         shortest_distances.append(d)

        if shortest_distances:
            return shortest_distances[:n]
        else:
            return [0., ]

    @staticmethod
    def _stone_on_level(stone: 'Stone', wall: 'Wall') -> bool:
        """
        Checks, if the stone is on the current building level or lower

        :param stone: new stone object
        :param wall: wall objects with the already placed stones
        :return: True, if on the current level or lower
        """
        # If the stone is on the current building level or lower
        level = wall.get_stone_level(stone)
        # print(f'  {level}, {wall.level}, {level <= wall.level}')
        if level <= wall.level:
            on_level = True
        else:
            on_level = False
        return on_level

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
                 distance2boundary=False, volume_below_stone=False, distance2closest_stone=False,
                 on_level=False):
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
        self.on_level = on_level

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
            intersects, b_intersection = self._intersection_boundary(stone, wall)
            if intersects:
                passed = False
                results.intersection_boundary = b_intersection

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
            results.distance2closest_stone = self._distance2closest_stone(stone, wall, 1)

        if self.on_level:
            results.on_level = self._stone_on_level(stone, wall)

        return passed, results

    def fitness(self, firefly: 'np.ndarray', stone: 'Stone', wall: 'Wall', **kwargs) -> 'ValidationResult':
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
            score += 5 * res.distance2closest_stone[0]

        # Punish stones that are not on the current level (or lower=
        if not res.on_level:
            score += 2 + stone.aabb_limits[1][2]  # 2 +
        res.score = score

        return res


class ValidatorFill(Validator):

    def __init__(self, intersection_boundary=False, intersection_stones=False,
                 volume_below_stone=False, distance2closest_stone=False,
                 on_level=False, delta_h=False):
        """
        The parameters control, witch validations will be executed

        :param intersection_boundary: validate intersections with the boundary
        :param intersection_stones: validate intersections with the previously placed stones
        """
        # Init tetgen from super()
        super().__init__()

        self.intersection_boundary = intersection_boundary
        self.intersection_stones = intersection_stones
        self.volume_below_stone = volume_below_stone
        self.distance2closest_stone = distance2closest_stone  # minimize distance to 2 stones
        self.on_level = on_level
        self.delta_h = delta_h
        
    def _h_closest_normal_stone(self, stone: 'Stone', wall: 'Wall') -> float:
        # Get the height difference to the closest normal stone
        # Higher than the stone -> bad
        # lower than the stone -> ok, but equal height would be best
        close_st = self._closest_stone(stone, wall, 5)
        normal_stones = [st for st in close_st if isinstance(st, NormalStone)]

        if normal_stones:
            delta_h = normal_stones[0].aabb_limits[1, 2] - stone.aabb_limits[1, 2]
            # print('delta_h', delta_h)
        else:
            # print('!!! NO CLOSE NORMAL STONE FOUND !!!', [st.__class__.__name__ for st in close_st], len(close_st), len(normal_stones))
            delta_h = -1
            # Todo: same result as 2 equally high stones --> return -1?

        return delta_h

    def validate(self, stone: 'Stone', wall: 'Wall') -> Tuple[bool, 'ValidationResult']:

        passed = True
        results = ValidationResult()

        if self.intersection_boundary:
            intersects, b_intersection = self._intersection_boundary(stone, wall)
            if intersects:
                passed = False
                results.intersection_boundary = b_intersection

        if self.intersection_stones:
            intersects, details = self._intersection_stones(stone, wall)
            if intersects:
                passed = False
                results.intersection_stones = details

        if self.volume_below_stone:
            v = self._volume_below_stone(stone, wall)
            results.volume_below_stone = v

        if self.distance2closest_stone:
            # Todo: If no closest stone -> no stone on the same height --> bad for filling stones?
            results.distance2closest_stone = self._distance2closest_stone(stone, wall, 2)

        if self.on_level:
            # Needed?
            results.on_level = self._stone_on_level(stone, wall)

        if self.delta_h:
            results.delta_h = self._h_closest_normal_stone(stone, wall)

        return passed, results

    def fitness(self, firefly: 'np.ndarray', stone: 'Stone', wall: 'Wall', **kwargs) -> 'ValidationResult':

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

        # free volume below the stone
        if self.volume_below_stone:
            score += res.volume_below_stone / stone.mesh_volume
        # distance to the closest stone on the same level
        if self.distance2closest_stone:
            # no penalty, if there is no stone on the same level
            # for dist in res.distance2closest_stone:
            #     score += (1 + 5 * dist)**2 - 1
            # score += (d1² - d2) if the difference is not too high (dont count far away stones
            d2c = res.distance2closest_stone
            if len(d2c) == 1 or d2c[0] - d2c[1] > .05:
                score += (1 + d2c[0] * 5)**2 - 1
            else:
                score += (1 + d2c[0] * 5)**2 - 1 - d2c[1] * 5
                # Distance: 2² - 4¹ = 0 --> same as if both distances == 0
                # if both are added separately -> only one distance is better than having 2...

        # Punish stones that are not on the current level (or lower=
        # if not res.on_level:
        #     score += 2 + stone.aabb_limits[1][2]  # 2 +
        # res.score = score

        if self.delta_h:
            if res.delta_h > 0:  # new stone is lower
                # score += res.delta_h
                pass
            else:  # new stone is higher than the closest normal stone (delta_h negative)
                score += 1 - res.delta_h
                # smallest stone: 1cm -> 2cm too high -> score += 1
            # if res.delta_h < 0:  # new stone is higher than the closest normal stone
            #     res.score += 2
            # else:
            #     res.score += res.delta_h*10

        res.score = score
        return res


@dataclass
class ValidationResult:
    """
    Storing all validation results. The total intersection volume gets automatically updated,
    if a boundary intersection or stones intersection is set.
    """
    __intersection_boundary: 'Intersection' = None  # Intersection with the boundary
    __intersection_stones: List['Intersection'] = ()  # All Intersections with other stones
    _intersection: bool = False  # whether there is any intersection or not
    _intersection_volume: float = 0  # total intersection volume
    _intersection_area: float = 0  # total intersection area
    _intersection_volume_b: float = 0
    _intersection_volume_s: float = 0
    distance2boundary: float = 0  # minimal distance to the boundary
    volume_below_stone: float = 0  # total free volume below the stone
    distance2closest_stone: List[float] = (0, )
    on_level: bool = True
    delta_h: float = 0

    score: float = 0

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

    @property
    def intersection_area(self) -> float:
        return self._intersection_area

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
        for i in new_intersections:
            self.update_total_volume(i)
        # map(self.update_total_volume, new_intersections)
        self._intersection_volume_s += np.sum([i.aabb_volume for i in new_intersections])

    def update_total_volume(self, new_intersection: 'Intersection'):
        # update the volume: try first the exact mesh volume. if not available, use aabb volume
        if new_intersection.mesh_volume:
            self._intersection_volume += new_intersection.mesh_volume
        elif new_intersection.aabb_volume:
            self._intersection_volume += new_intersection.aabb_volume
        else:
            print('intersection but no volume added')
        # update intersection area
        self._intersection_area += new_intersection.aabb_area
