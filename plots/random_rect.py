# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.02.2021, random_rect.py
# Create and plot random regular cubes

from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt

from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import set_axes_equal

if TYPE_CHECKING:
    from trockenmauer.stone import Stone


"""
Generates regular cuboids with a length within ``[0.05, 0.3]``
and scales them randomly with a factor within ``[0.5, 2]``.
This results in a minimal length of 0.05 and a maximal length of 0.6.
"""

plt.rcParams.update({'axes.titlesize': 'small'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'small'})
plt.rcParams.update({'ytick.labelsize': 'small'})


def plot_stones(stones: List['Stone']):
    """
    Plot all stones in separate subplots

    :param stones: List of stones
    :return: plot window
    """
    # Caluculate the number of rows / columns of the subplots
    nrows = len(stones) // 4 + 1
    ncols = 4

    fig = plt.figure(figsize=(ncols*3.1, nrows*3))
    fig.suptitle('Collection of stones and their eigenvectors')

    for i, s in enumerate(stones):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        # axs[i // 3, i % 3] = s.plot()
        s.add_shape_to_ax(ax)
        s.add_labels_to_ax(ax, positive_eigenvec=False)
        ax.set_title(s.name)

        set_axes_equal(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    plt.show()


if __name__ == '__main__':
    # stones = [generate_regular_stone(0.2, 0.1, 0.1, name=str(i)) for i in range(10)]
    stoners = [generate_regular_stone(0.3, 0.15, 0.1, edge_noise=0.5, scale=[1, 2]) for i in range(10)]

    plot_stones(stoners)
