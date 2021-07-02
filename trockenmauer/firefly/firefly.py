# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

# pylint: disable=too-many-arguments


from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..fitness import ValidationResult


class Firefly:
    def __init__(self, alpha, beta, gamma, lower_boundary, upper_boundary, function,
                 bit_generator, pos_function, coord=None, **kwargs):

        scale = upper_boundary - lower_boundary
        self.__alpha = alpha * scale
        self.__beta = beta
        self.__gamma = gamma / np.sqrt(scale)

        # initiate a random coordinate
        self.__lower_boundary = lower_boundary
        self.__upper_boundary = upper_boundary
        # self._random = bit_generator
        self._random = bit_generator
        self._function = function
        self.pos_function = pos_function
        self._kwargs = kwargs  # additional arguments fur the function

        self.__position = None
        self.__value = None
        self.__validation_result = None

        if np.any(coord):
            self.__position = coord
            # move with no distance -> random walk
            self.move_towards(coord)

        else:
            self._initialize()

    def move_towards(self, better_position):
        # euclidean distance
        distance = np.linalg.norm(self._position - better_position)

        # update position
        self._position = self._position + \
            self.__beta*np.exp(-self.__gamma*(distance**2)) * (better_position-self._position) + \
            self.__alpha*self._random.normal(0, 1, 3)
        # self.__alpha*self._random.uniform(-1, 1, 3)
        # Todo: Random Walk: uniform [-1:1] -> Gauss [mu=0, sigma=1]

    def random_walk(self, area):
        self._position = self._random.uniform(self._position-area, self._position+area, 3)

    def _initialize(self) -> None:
        """
        Initialize a new random position and its value
        """
        self._position = self._random.uniform(self.__lower_boundary, self.__upper_boundary, 3)

    @property
    def position(self) -> np.ndarray:
        """
        Get the coordinate's position

        Returns:
            numpy.ndarray: the Position
        """
        return self._position

    # Internal Getter
    @property
    def _position(self) -> np.ndarray:
        return self.__position

    # Internal Setter for automatic position clipping and value update
    @_position.setter
    def _position(self, new_pos: np.ndarray) -> None:
        """
        Set the coordinate's new position.
        Also updates checks whether the position is within the set boundaries
        and updates the coordinate's value.

        Args:
            new_pos (numpy.ndarray): The new coordinate position
        """
        self.__position = np.clip(new_pos, a_min=self.__lower_boundary, a_max=self.__upper_boundary)
        if self.pos_function:
            self.__position = self.pos_function(self.__position, **self._kwargs)
        self.__validation_result = self._function(self.__position, **self._kwargs)
        self.__value = self.__validation_result.score

    @property
    def value(self) -> float:
        return self.__value

    @property
    def validation_result(self) -> 'ValidationResult':
        return self.__validation_result

    def __eq__(self, other) -> bool:
        return self.__value == other.value

    def __ne__(self, other) -> bool:
        return self.__value != other.value

    def __lt__(self, other) -> bool:
        return self.__value < other.value

    def __le__(self, other) -> bool:
        return self.__value <= other.value

    def __gt__(self, other) -> bool:
        return self.__value > other.value

    def __ge__(self, other) -> bool:
        return self.__value >= other.value

    def __str__(self):
        return self.__value
