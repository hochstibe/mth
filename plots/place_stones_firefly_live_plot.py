# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from time import time

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from trockenmauer.stone import Boundary
from trockenmauer.wall import Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import set_axes_equal
from trockenmauer.math_utils import Translation
from trockenmauer.validation import Validator
from trockenmauer.placement import xy_fixed_z
from swarmlib.firefly_problem import FireflyProblem


STONES = 10

boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure(figsize=(12, 9))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
start = time()

validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True, volume_below_stone=True,
                      # distance2closest_stone=True
                      )


def init_func():
    boundary.add_shape_to_ax(ax)
    # Set plot properties
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.draw()


def func(i):
    # Generate, optimize and plot a stone

    stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(i))
    # Find a placement
    problem = FireflyProblem(10, validator.fitness, boundary.aabb_limits[0], boundary.aabb_limits[1],
                             iteration_number=40, init_function=xy_fixed_z, stone=stone, wall=wall)
    res = problem.solve()
    # print(fitness, xyz)
    print(i, res.position, res.value)
    t = Translation(translation=res.position - stone.bottom_center)
    stone.transform(transformation=t)
    wall.add_stone(stone)
    stone.add_shape_to_ax(ax)
    plt.draw()

    # Stop criteria
    if i == STONES - 1:
        stop = time()
        m, s = divmod(stop-start, 60)
        print(f"Successfully placed {len(wall.stones)} stones in {int(m)}'{round(s, 1)}''.")


# Run the animation
ani = FuncAnimation(fig, func, frames=STONES, interval=1, repeat=False, init_func=init_func)

plt.show()
