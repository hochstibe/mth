# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#

import random

from matplotlib import pyplot as plt
import numpy as np

from trockenmauer.stone import Stone

# Test the rotation with random points


def random_points(n):
    points = list()
    for i in range(n):
        points.append([3*(random.random()+1), 2*(random.random()+2), 1*(random.random()+3)])
    return np.array(points)


stone = Stone(random_points(10), 'random points')

fig = plt.figure()
fig.suptitle('Collection of stones')

ax = fig.add_subplot(121, projection='3d')
stone.add_plot_to_ax(ax, orig=True, positive_eigenvec=False)
ax.set_title('orig')
ax = fig.add_subplot(122, projection='3d')
stone.add_plot_to_ax(ax)
ax.set_title('rotated')
plt.show()
