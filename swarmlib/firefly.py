# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

# pylint: disable=too-many-arguments

import numpy as np


class Firefly:
    def __init__(self, alpha, beta, gamma, lower_boundary, upper_boundary, function,
                 bit_generator, init_function, **kwargs):

        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

        # initiate a random coordinate
        self.__lower_boundary = lower_boundary
        self.__upper_boundary = upper_boundary
        # self._random = bit_generator
        self._random = bit_generator
        self._function = function
        self._init_function = init_function
        self._kwargs = kwargs  # additional arguments fur the function

        self.__value = None
        self.__position = None
        self._initialize()

    def move_towards(self, better_position):
        # euclidean distance
        distance = np.linalg.norm(self._position - better_position)

        # update position
        self._position = self._position + \
            self.__beta*np.exp(-self.__gamma*(distance**2)) * (better_position-self._position) + \
            self.__alpha*(self._random.uniform(0, 1)-0.5)

    def random_walk(self, area):
        self._position = np.array([self._random.uniform(cord-area, cord+area) for cord in self._position])

    def _initialize(self) -> None:
        """
        Initialize a new random position and its value
        """
        position = self._random.uniform(self.__lower_boundary, self.__upper_boundary, 3)
        if self._init_function:
            self._position = self._init_function(position, **self._kwargs)

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
        self.__value = self._function(self.__position, **self._kwargs)

    @property
    def value(self) -> float:
        return self.__value

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
