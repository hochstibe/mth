# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.05.21, place_stones_firefly.py
#

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trockenmauer.stone import Boundary, Wall
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import set_axes_equal
from trockenmauer.math_utils import Translation
from trockenmauer.placement import find_placement
from trockenmauer.validation import Validator
from FireflyAlgorithm import FireflyAlgorithm

boundary = Boundary()
wall = Wall(boundary)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
boundary.add_shape_to_ax(ax)

validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True)
# place stones

for i in range(20):
    stone = generate_regular_stone(.4, 0.2, 0.1, edge_noise=0.5, scale=[1, 2], name=str(i))
    # Find a placement
    # xyz = find_placement(wall)
    print(stone.name, stone.mesh.is_closed(), stone.mesh.is_manifold(), stone.mesh.is_oriented())
    firefly = FireflyAlgorithm(3, 2, 20, 0.5, 0.2, 1.0, 0.0, .5, validator.fitness, stone=stone, wall=wall)
    fitness, xyz = firefly.Run()
    print(fitness, xyz)
    t = Translation(translation=xyz - stone.bottom_center)
    stone.transform(transformation=t)
    wall.add_stone(stone)
    stone.add_shape_to_ax(ax)

print(f'Successfully placed {len(wall.stones)} stones.')

set_axes_equal(ax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
