# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.2021, plot.py
#

from typing import List

import matplotlib.pyplot as plt

from .stone import Stone

plt.rcParams.update({'axes.titlesize': 'small'})
plt.rcParams.update({'axes.labelsize': 'small'})
plt.rcParams.update({'xtick.labelsize': 'small'})
plt.rcParams.update({'ytick.labelsize': 'small'})


def plot_stones(stones: List[Stone]):
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
        s.add_plot_to_ax(ax)
        ax.set_title(s.name)

    plt.show()
