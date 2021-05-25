# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from time import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trockenmauer.stone import Boundary, Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import set_axes_equal
from trockenmauer.math_utils import Translation
from trockenmauer.placement import find_placement
from trockenmauer.validation import Validator
from swarmlib.firefly_problem import FireflyProblem

boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
boundary.add_shape_to_ax(ax)

validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True)

# place stones
start = time()
for i in range(20):
    stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(i))
    # Find a placement
    problem = FireflyProblem(5, validator.fitness, boundary.aabb_limits[0], boundary.aabb_limits[1],
                             iteration_number=20, stone=stone, wall=wall)
    res = problem.solve()
    # print(fitness, xyz)
    print(i, res.position, res.value)
    t = Translation(translation=res.position - stone.bottom_center)
    stone.transform(transformation=t)
    wall.add_stone(stone)
    stone.add_shape_to_ax(ax)

stop = time()
m, s = divmod(stop-start, 60)
print(f"Successfully placed {len(wall.stones)} stones in {m}'{round(s, 1)}''.")

set_axes_equal(ax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
