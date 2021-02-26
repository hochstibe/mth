# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21
# tester.py -

import random

from matplotlib import pyplot as plt
import numpy as np

from trockenmauer.generate_stones import generate_regular_stone, add_noise_to_vertices
from trockenmauer.stone import Stone


def random_points(n):
    points = list()
    for i in range(n):
        points.append([random.random(), random.random(), random.random()])
    return np.array(points)

# stone = generate_regular_stone(0.2, 0.2, 0.2, name='Toni')
# print(stone)
# stone = add_noise_to_vertices(stone.vertices)
# print(stone)
stone = Stone(random_points(10), 'random points')

fig = plt.figure()
fig.suptitle('Collection of stones')

ax = fig.add_subplot(121, projection='3d')
stone.add_plot_to_ax(ax, orig=True)
ax.set_title('orig')
ax = fig.add_subplot(122, projection='3d')
stone.add_plot_to_ax(ax)
ax.set_title('rotated')
plt.show()
