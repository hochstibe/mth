# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

from copy import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .firefly import Firefly

# LOGGER = logging.getLogger(__name__)


class FireflyProblem:
    def __init__(self, firefly_number, function, lower_boundary=0, upper_boundary=1, alpha=0.01, beta=1, gamma=1,
                 iteration_number=100, seed=None, init_function=None, random_walk_area=.1, **kwargs):
        """Initializes a new instance of the `FireflyProblem` class.

        Keyword arguments:  \r
        `firefly_number`   -- Number of fireflies used for solving
        `function`         -- The 2D evaluation function. Its input is a 2D numpy.array  \r
        `upper_boundary`   -- Upper boundary of the function (default 4)  \r
        `lower_boundary`   -- Lower boundary of the function (default 0)  \r
        `alpha`            -- Randomization parameter (default 0.25)  \r
        `beta`             -- Attractiveness at distance=0 (default 1)  \r
        `gamma`            -- Characterizes the variation of the attractiveness. (default 0.97) \r
        `iteration_number` -- Number of iterations to execute (default 100)  \r
        `interval`         -- Interval between two animation frames in ms (default 500)  \r
        `continuous`       -- Indicates whether the algorithm should run continuously (default False)
        """

        self._random = np.random.default_rng(seed)
        self.__iteration_number = iteration_number
        self._random_walk_area = random_walk_area
        # Create fireflies
        self.__fireflies = [
            Firefly(alpha, beta, gamma, lower_boundary, upper_boundary, function,
                    bit_generator=self._random, init_function=init_function,  **kwargs)
            for _ in range(firefly_number)
        ]

        self.__fireflies = np.sort(self.__fireflies)

        # Initialize visualizer for plotting
        # self._visualizer = BaseVisualizer(**kwargs)
        # self._visualizer.add_data(positions=[firefly.position for firefly in self.__fireflies])

    def solve(self) -> Tuple[Firefly, List['Iteration']]:
        """Solve the problem."""
        best = self.__fireflies[0]  # best firefly, sorted in __init__
        history = list()  # history of firefly positions
        # Add initial data for visualization
        history.append(Iteration(np.array([firefly.position for firefly in self.__fireflies]),
                                 np.array([firefly.value for firefly in self.__fireflies])))

        # keep track of iterations and updates
        no_update = 0
        iteration = 0

        while no_update < 10 and iteration < self.__iteration_number:
            for i in self.__fireflies:
                for j in self.__fireflies:
                    if j < i:
                        i.move_towards(j.position)
            iteration += 1

            self.__fireflies = np.sort(self.__fireflies)
            current_best = self.__fireflies[0]
            if not best or current_best < best:
                # update the best solution
                # best = deepcopy(current_best)
                best = copy(current_best)
                no_update = 0

            else:
                # no update of the best solution
                no_update += 1

            # LOGGER.info('Current best value: %s, Overall best value: %s', current_best.value, best.value)

            # randomly walk the best firefly (was not moved during algorithm)
            current_best.move_towards(current_best.position)

            # Add data for visualization
            history.append(Iteration(np.array([firefly.position for firefly in self.__fireflies]),
                                     np.array([firefly.value for firefly in self.__fireflies])))
            # self._visualizer.add_data(positions=[firefly.position for firefly in self.__fireflies])
        print(iteration, 'iterations, no update for', no_update)


        return best, history


@dataclass
class Iteration:
    positions: 'np.ndarray'
    values: 'np.ndarray'
    velocities: 'np.ndarray' = None

    # @property
    # def velocities(self):
    #     if not self._velocities:
    #         vel = [self.positions[i + 1] - self.positions[i] for i in range(len(self.positions) - 1)]
    #         vel.insert(0, np.zeros(3))
    #         self._velocities = vel
#
    #     return self._velocities
