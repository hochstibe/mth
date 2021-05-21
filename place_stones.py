# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 19.05.2021, place_stones.py
#

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trockenmauer.stone import Boundary, Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.utils import set_axes_equal
from trockenmauer.math_utils import Translation
from trockenmauer.placement import find_placement

boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
boundary.add_plot_to_ax(ax)

# place stones
for i in range(20):
    stone = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2])
    xyz = find_placement(wall)
    t = Translation(translation=xyz - stone.bottom_center)
    stone.transform(transformation=t)

    # add the stone to the wall
    wall.stones.append(stone)

    # add the stone to the plot
    stone.add_shape_to_ax(ax)

set_axes_equal(ax)
plt.show()
