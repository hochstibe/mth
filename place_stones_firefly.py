# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from time import time
from datetime import datetime

from trockenmauer.stone import Boundary
from trockenmauer.wall import Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.math_utils import Translation
from trockenmauer.validation import Validator
from trockenmauer.placement import random_init_fixed_z
from swarmlib.firefly_problem import FireflyProblem


STONES = 10
FIREFLIES = 10
ITERATIONS = 30

filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{STONES}_stones_{FIREFLIES}_{ITERATIONS}_iterations'

boundary = Boundary()
wall = Wall(boundary)

validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True, volume_below_stone=True,
                      distance2closest_stone=True
                      )

start = time()
for i in range(STONES):
    # Generate, optimize and plot a stone
    stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(i))
    # Find a placement
    problem = FireflyProblem(FIREFLIES, validator.fitness, boundary.aabb_limits[0], boundary.aabb_limits[1],
                             iteration_number=ITERATIONS, init_function=random_init_fixed_z, stone=stone, wall=wall)
    res, history = problem.solve()
    stone.position_history = history
    # print(fitness, xyz)
    print(i, res.position, res.value)
    t = Translation(translation=res.position - stone.bottom_center)
    stone.transform(transformation=t)
    wall.add_stone(stone)


# Stop criteria
stop = time()
m, s = divmod(stop-start, 60)
print(f"Successfully placed {len(wall.stones)} stones in {int(m)}'{round(s, 1)}''.")


wall.replay(fireflies=True, save=filename)
