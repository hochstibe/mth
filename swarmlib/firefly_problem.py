# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

from copy import copy  # deepcopy
import logging

from numpy.random import default_rng

from swarmlib.firefly import Firefly

LOGGER = logging.getLogger(__name__)


class FireflyProblem:
    def __init__(self, firefly_number, function, lower_boundary=0, upper_boundary=1, alpha=0.25, beta=1, gamma=0.97,
                 iteration_number=100, seed=None, **kwargs):
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

        self._random = default_rng(seed)
        self.__iteration_number = iteration_number
        # Create fireflies
        self.__fireflies = [
            Firefly(alpha, beta, gamma, lower_boundary, upper_boundary, function,
                    bit_generator=self._random,  **kwargs)
            for _ in range(firefly_number)
        ]

        # Initialize visualizer for plotting
        # self._visualizer = BaseVisualizer(**kwargs)
        # self._visualizer.add_data(positions=[firefly.position for firefly in self.__fireflies])

    def solve(self) -> Firefly:
        """Solve the problem."""
        best = None
        for _ in range(self.__iteration_number):
            for i in self.__fireflies:
                for j in self.__fireflies:
                    if j < i:
                        i.move_towards(j.position)

            current_best = min(self.__fireflies)
            if not best or current_best < best:
                # best = deepcopy(current_best)
                best = copy(current_best)

            LOGGER.info('Current best value: %s, Overall best value: %s', current_best.value, best.value)

            # randomly walk the best firefly
            current_best.random_walk(0.1)

            # Add data for visualization
            # self._visualizer.add_data(positions=[firefly.position for firefly in self.__fireflies])

        return best
