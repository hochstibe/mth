# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, place_stones_randomly.py
#

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from trockenmauer.stone import Boundary, Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import set_axes_equal
from trockenmauer.math_utils import Translation
from trockenmauer.placement import find_placement
from trockenmauer.validation import Validator


STONES = 30

boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True,
                      volume_below_stone=True
                      )


def init_func():
    boundary.add_shape_to_ax(ax)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.draw()


def func(i):
    # Generate, validate and plot a stone

    stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(i))
    # Find a placement
    xyz = find_placement(wall)
    t = Translation(translation=xyz - stone.bottom_center)
    stone.transform(transformation=t)

    # Validate the placement
    passed, errors = validator.validate(stone, wall)

    print(errors)
    if passed:
        # add the stone to the wall and to the plot
        wall.add_stone(stone)
        stone.add_shape_to_ax(ax)
        print(' passed')
    else:
        if errors.intersection_boundary:
            # Add the intersection to the plot
            # errors.intersection_boundary.add_shape_to_ax(ax, color='red')
            print(' boundary')
            pass
        if errors.intersection_stones:
            # Add the intersection to the plot
            # for inter in errors.intersection_stones:
                # inter.add_shape_to_ax(ax, color='orange')
            print(' stone')
            pass

    if i == STONES - 1:
        print(f'Successfully placed {len(wall.stones)} stones.')

# Run the animation
ani = FuncAnimation(fig, func, frames=STONES, interval=1, repeat=False, init_func=init_func)

plt.show()
