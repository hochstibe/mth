# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#

import random

from matplotlib import pyplot as plt
import numpy as np
from scipy import spatial

from trockenmauer.stone import Stone
from trockenmauer.generate_stones import generate_regular_stone

# Test the rotation with random points


def random_points(n):
    points = list()
    for i in range(n):
        points.append([1*(random.random()+1), 1*(random.random()+2), 1*(random.random()+3)])
    return np.array(points)


def collect_edges(tri):
    edges = set()

    def sorted_tuple(a,b):
        return (a, b) if a < b else (b, a)
    # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    for (i0, i1, i2, i3) in tri.simplices:
        edges.add(sorted_tuple(i0, i1))
        edges.add(sorted_tuple(i0, i2))
        edges.add(sorted_tuple(i0, i3))
        edges.add(sorted_tuple(i1, i2))
        edges.add(sorted_tuple(i1, i3))
        edges.add(sorted_tuple(i2, i3))
    return edges


def plot_tri_simple(ax, points, tri):
    print(len(tri.simplices), 'simplices (tetrahedrons)')
    print(len(tri.convex_hull), 'triangles in convex hull')
    c = 0
    colors = ['g', 'r', 'b', 'c', 'y', 'orange']
    # for tr in tri.convex_hull:  # triangles
    for tr in tri.simplices:  # quad

        pts = points[tr, :]

        ax.plot3D(pts[[0, 1], 0], pts[[0, 1], 1], pts[[0, 1], 2], color=colors[c], lw='0.3')
        ax.plot3D(pts[[0, 2], 0], pts[[0, 2], 1], pts[[0, 2], 2], color=colors[c], lw='0.3')
        ax.plot3D(pts[[0, 3], 0], pts[[0, 3], 1], pts[[0, 3], 2], color=colors[c], lw='0.3')  # not in hull
        ax.plot3D(pts[[1, 2], 0], pts[[1, 2], 1], pts[[1, 2], 2], color=colors[c], lw='0.3')
        ax.plot3D(pts[[1, 3], 0], pts[[1, 3], 1], pts[[1, 3], 2], color=colors[c], lw='0.3')  # not in hull
        ax.plot3D(pts[[2, 3], 0], pts[[2, 3], 1], pts[[2, 3], 2], color=colors[c], lw='0.3')  # not in hull
        c += 1

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')


stone = Stone(random_points(5), 'random points')
# stone = generate_regular_stone(.2, .1, .1)


fig = plt.figure()
fig.suptitle('Collection of stones')

ax = fig.add_subplot(131, projection='3d')
stone.add_plot_to_ax(ax, orig=True, positive_eigenvec=False)
ax.set_title('orig')
ax = fig.add_subplot(132, projection='3d')
stone.add_plot_to_ax(ax, positive_eigenvec=False)
ax.set_title('rotated')


ax = fig.add_subplot(133, projection='3d')
# spatial.ConvexHull --> not good
# hull = spatial.ConvexHull(stone.vertices)
# ax.plot(stone.vertices[:, 0], stone.vertices[:, 1], stone.vertices[:, 2], 'o')
# print(len(hull.simplices), len(stone.vertices))
# for simplex in hull.simplices:
#     plt.plot(stone.vertices[simplex, 0], stone.vertices[simplex, 1], stone.vertices[simplex, 2], 'k-')

# spatial.Delaunay --> good, but returns tetrahedrons --> require surface triangles
tri = spatial.Delaunay(stone.vertices)
print(tri.convex_hull)
plot_tri_simple(ax, stone.vertices, tri)



plt.show()
